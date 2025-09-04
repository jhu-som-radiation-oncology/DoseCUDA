from .plan import Plan, Beam, DoseGrid, VolumeObject, Prescription
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, RTIonPlanStorage
import datetime
import pkg_resources
import dose_kernels
import os
import SimpleITK as sitk


def get_roi_num(rt_ds, ROIName):
    for roi in rt_ds.StructureSetROISequence:
        if roi.ROIName == ROIName:
            return roi.ROINumber
    return None


def create_dose_reference(ds, prescriptions, rt_struct_file=None):
    # Initialize Dose Reference Sequence if not present
    if not hasattr(ds, 'DoseReferenceSequence'):
        ds.DoseReferenceSequence = []

    # Process RT structure file if provided
    rt_ds = pyd.dcmread(rt_struct_file, force=True) if rt_struct_file else None

    for i, rx in enumerate(prescriptions):
        # Determine ROI number
        if rt_struct_file:
            ptv_roi_num = get_roi_num(rt_ds, rx.ROIName)
            if ptv_roi_num is None:
                raise ValueError(f"No structure labelled '{rx.ROIName}' found in RT structure set.")
        else:
            ptv_roi_num = i + 1

        # Create Dose Reference dataset
        dr = Dataset()
        dr.DoseReferenceNumber = len(ds.DoseReferenceSequence) + 1
        dr.ReferencedROINumber = ptv_roi_num
        dr.DoseReferenceStructureType = 'VOLUME'
        dr.DoseReferenceDescription = f'{rx.ROIName} Prescription'
        dr.DoseReferenceType = 'TARGET'
        dr.TargetPrescriptionDose = rx.TargetPrescriptionDose
        dr.TargetUnderdoseVolumeFraction = rx.TargetUnderdoseVolumeFraction
        dr.add_new((0x4001, 0x0010), 'LO', 'RAYSEARCHLABS 2.0')
        dr.add_new((0x4001, 0x1011), 'UN', str(rx.TargetPrescriptionDose).encode())

        ds.DoseReferenceSequence.append(dr)

    # Set Referenced SOP Instance UID if RT structure file is provided
    if rt_struct_file:
        rss = ds.ReferencedStructureSetSequence[0]
        rss.ReferencedSOPInstanceUID = rt_ds.SOPInstanceUID

    return ds


class IMPTBeamModel():

    def __init__(self, dicom_rangeshifter_label, path_to_model, folder_rangeshifter_label):
        
        self.dicom_rangeshifter_label = dicom_rangeshifter_label

        # import the machine geometry
        machine_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "machine_geometry.csv"))

        self.VSADX = None
        self.VSADY = None

        for line in open(machine_geometry_path, "r"):
            if line.startswith("VSADX"):
                self.VSADX = float(line.split(',')[1])
            
            if line.startswith("VSADY"):
                self.VSADY = float(line.split(',')[1])

        if self.VSADX is None:
            raise Exception("VSADX not found in machine_geometry.csv")
        
        if self.VSADY is None:
            raise Exception("VSADY not found in machine_geometry.csv")


        # import LUT for this rangeshifter
        energy_list_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_rangeshifter_label, "energies.csv"))
        self.energy_table = pd.read_csv(energy_list_path)

        self.energy_labels = self.energy_table["energy_label"].to_numpy()
        energy_ids = self.energy_table["index"].to_numpy()
        self.reference_sigmas = self.energy_table["sigma"].to_numpy()

        self.divergence_params = []
        self.lut_depths = []
        self.lut_sigmas = []
        self.lut_idds = []

        for energy_label, energy_id in zip(self.energy_labels, energy_ids):

            lut_depths = []
            lut_sigmas = []
            lut_idds = []
            divergence_params = []
            
            lut_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_rangeshifter_label, "energy_%03d.csv" % energy_id))

            with open(lut_path, "r") as f:
                f.readline() # header
                line = f.readline()
                parts = line.split(",")
                divergence_params = [float(part) for part in parts]
                f.readline() # blank
                f.readline() # header
                k = 0 
                for line in f:
                    if k > 399:
                        break
                    k += 1
                    parts = line.split(",")
                    lut_depths.append(float(parts[0]))
                    lut_sigmas.append(float(parts[1]))
                    lut_idds.append(float(parts[2]))
            
            self.divergence_params.append(divergence_params)    
            self.lut_depths.append(lut_depths)
            self.lut_sigmas.append(lut_sigmas)
            self.lut_idds.append(lut_idds)

        self.divergence_params = np.array(self.divergence_params, dtype=np.single)
        self.lut_depths = np.array(self.lut_depths, dtype=np.single)
        self.lut_sigmas = np.array(self.lut_sigmas, dtype=np.single)
        self.lut_idds = np.array(self.lut_idds, dtype=np.single)

        # self.lut_depths = self.lut_depths[:, 0:399]
        # self.lut_sigmas = self.lut_sigmas[:, 0:399]
        # self.lut_idds = self.lut_idds[:, 0:399]

    def energyIDFromLabel(self, energy_label):

        energy_row = self.energy_table[self.energy_table["energy_label"] == energy_label]

        if len(energy_row) != 1:
            raise Exception("Energy ID not found for %.2f" % energy_label)

        return energy_row.index[0]


class IMPTDoseGrid(DoseGrid):

    def __init__(self):
        super().__init__()
        self.RLSP = []

    def RLSPFromHU(self, machine_name):

        rlsp_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", machine_name, "HU_RLSP.csv"))
        df_rlsp = pd.read_csv(rlsp_table_path)

        hu_curve = df_rlsp["HU"].to_numpy()
        rlsp_curve = df_rlsp["RLSP"].to_numpy()
        
        rlsp = np.array(np.interp(self.HU, hu_curve, rlsp_curve), dtype=np.single)
        
        return rlsp
    
    def computeIMPTPlan(self, plan, gpu_id=0):

        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.RLSP = self.RLSPFromHU(plan.machine_name)

        #check if spacing is isotropic
        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")
        
        rlsp_object = VolumeObject()
        rlsp_object.voxel_data = np.array(self.RLSP, dtype=np.single)
        rlsp_object.origin = np.array(self.origin, dtype=np.single)
        rlsp_object.spacing = np.array(self.spacing, dtype=np.single)

        for beam in plan.beam_list:

            try:           
                model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(beam.dicom_rangeshifter_label)
            except ValueError:
                print("Beam model not found for rangeshifter ID %s" % beam.dicom_rangeshifter_label)
                sys.exit(1)
            beam_model = plan.beam_models[model_index]

            beam_wet = dose_kernels.proton_raytrace_cuda(beam_model, rlsp_object, beam, gpu_id)

            wet_object = VolumeObject()
            wet_object.voxel_data = np.array(beam_wet, dtype=np.single)
            wet_object.origin = np.array(self.origin, dtype=np.single)
            wet_object.spacing = np.array(self.spacing, dtype=np.single)

            beam_dose = dose_kernels.proton_spot_cuda(beam_model, rlsp_object, wet_object, beam, gpu_id)

            self.beam_doses.append(beam_dose * plan.n_fractions)
            self.dose += beam_dose

        self.dose *= plan.n_fractions

    def computeIMPTWET(self, plan, gpu_id=0):

        self.RLSP = self.RLSPFromHU(plan.machine_name)

        rlsp_object = VolumeObject()
        rlsp_object.voxel_data = np.array(self.RLSP, dtype=np.single)
        rlsp_object.origin = np.array(self.origin, dtype=np.single)
        rlsp_object.spacing = np.array(self.spacing, dtype=np.single)

        beam_wets = []
        for beam in plan.beam_list:

            try:           
                model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(beam.dicom_rangeshifter_label)
            except ValueError:
                print("Beam model not found for rangeshifter ID %s" % beam.dicom_rangeshifter_label)
                sys.exit(1)
            beam_model = plan.beam_models[model_index]

            beam_wet = dose_kernels.proton_raytrace_cuda(beam_model, rlsp_object, beam, gpu_id)

            wet_object = VolumeObject()
            wet_object.voxel_data = np.array(beam_wet, dtype=np.single)
            wet_object.origin = np.array(self.origin, dtype=np.single)
            wet_object.spacing = np.array(self.spacing, dtype=np.single)

            beam_wets.append(wet_object)
        
        return beam_wets

    def computeIMPTPlanFromWET(self, plan, wet_object, beam, gpu_id=0):

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")
        
        rlsp_object = VolumeObject()
        rlsp_object.voxel_data = np.array(self.RLSP, dtype=np.single)
        rlsp_object.origin = np.array(self.origin, dtype=np.single)
        rlsp_object.spacing = np.array(self.spacing, dtype=np.single)

        try:           
            model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(beam.dicom_rangeshifter_label)
        except ValueError:
            print("Beam model not found for rangeshifter ID %s" % beam.dicom_rangeshifter_label)
            sys.exit(1)
        beam_model = plan.beam_models[model_index]

        beam_dose = dose_kernels.proton_spot_cuda(beam_model, rlsp_object, wet_object, beam, gpu_id)

        return beam_dose
    
    def writeWETNRRD(self, plan, wet_path, gpu_id=0):

        if not wet_path.endswith(".nrrd"):
            raise Exception("WET path must have .nrrd extension")

        self.RLSP = self.RLSPFromHU(plan.machine_name)
        fw = sitk.ImageFileWriter()

        rlsp_object = VolumeObject()
        rlsp_object.voxel_data = np.array(self.RLSP, dtype=np.single)
        rlsp_object.origin = np.array(self.origin, dtype=np.single)
        rlsp_object.spacing = np.array(self.spacing, dtype=np.single)

        for i, beam in enumerate(plan.beam_list):
            try:           
                model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(beam.dicom_rangeshifter_label)
            except ValueError:
                print("Beam model not found for rangeshifter ID %s" % beam.dicom_rangeshifter_label)
                sys.exit(1)
            beam_model = plan.beam_models[model_index]

            beam_wet = dose_kernels.proton_raytrace_cuda(beam_model, rlsp_object, beam, gpu_id)

            HU_img = sitk.GetImageFromArray(beam_wet)
            HU_img.SetOrigin(self.origin)
            HU_img.SetSpacing(self.spacing)

            fw.SetFileName(wet_path.replace(".nrrd", "_beam%02i.nrrd" % (i+1)))
            fw.Execute(HU_img)


class IMPTBeam(Beam):

    def __init__(self, beam=None):
        super().__init__()

        self.spot_list = []
        self.n_spots = 0
        self.dicom_rangeshifter_label = None

        if beam is not None:
            self.gantry_angle = beam.gantry_angle
            self.couch_angle = beam.couch_angle
            self.iso = beam.iso
            self.spot_list = beam.spot_list
            self.n_spots = beam.n_spots
            self.dicom_rangeshifter_label = beam.dicom_rangeshifter_label

    def addSpotData(self, cp, energy_id):

        spm = np.reshape(np.array(cp.ScanSpotPositionMap), (-1, 2))
        mus = np.array(cp.ScanSpotMetersetWeights)
        energy_id_array = np.full(mus.size, energy_id)

        spot_list = np.array(np.column_stack((spm, mus, energy_id_array)), dtype=np.single)
        self.spot_list = np.vstack((self.spot_list, spot_list)) if self.n_spots > 0 else spot_list
        self.n_spots += mus.size

    def changeSpotEnergy(self, energy_id):
        self.spot_list[:, 3] = energy_id

    def addSingleSpot(self, x, y, mu, energy_id):
        spot = np.array([x, y, mu, energy_id], dtype=np.single)
        self.spot_list = np.vstack((self.spot_list, spot)) if self.n_spots > 0 else spot
        self.n_spots += 1

    def resetSpots(self):
        self.spot_list = np.empty((0, 4), dtype=np.single)
        self.n_spots = 0


class IMPTPlan(Plan):

    def __init__(self, machine_name = "HitachiProbeatJHU"):
        super().__init__()
        self.machine_name = machine_name

        rangeshifter_list = pd.read_csv(pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", machine_name, "rangeshifter_labels.csv")))
        self.dicom_rangeshifter_label = rangeshifter_list["dicom_rangeshifter_label"]
        self.folder_rangeshifter_label = rangeshifter_list["folder_rangeshifter_label"]

        self.beam_models = []
        for d,f in zip(self.dicom_rangeshifter_label, self.folder_rangeshifter_label):
            self.beam_models.append(IMPTBeamModel(d, os.path.join("lookuptables", "protons", machine_name), f))

    def load_ct_series(self, ct_dir):
        """Read all .dcm files in ct_dir and sort by InstanceNumber."""
        files = [os.path.join(ct_dir, f)
                for f in os.listdir(ct_dir)
                if f.lower().endswith('.dcm')]
        dsets = [pyd.dcmread(f, force=True) for f in files]
        dsets.sort(key=lambda ds: ds.InstanceNumber)
        return dsets

    def writePlanDicom(self, ct_dir, output_file, rt_struct_file = None):
        """Build and save an RT Ion Plan DICOM for a proton pencil-beam scan."""
        # ————————————
        # File Meta Information
        # ————————————
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID    = RTIonPlanStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        # ————————————
        # Main RT ION PLAN Dataset
        # ————————————
        ds = FileDataset(output_file, {}, file_meta=meta, preamble=b"\0"*128)

        # Copy patient & study from first CT slice
        ct_series = self.load_ct_series(ct_dir)
        first = ct_series[0]
        ds.PatientName        = first.PatientName
        ds.PatientID          = first.PatientID
        ds.StudyInstanceUID   = first.StudyInstanceUID
        ds.SeriesInstanceUID  = generate_uid()
        ds.FrameOfReferenceUID= first.FrameOfReferenceUID

        # RT Ion Plan metadata
        ds.SOPClassUID = RTIonPlanStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality       = 'RTPLAN'

        ds.RTPlanLabel   = self.RTPlanLabel if self.RTPlanLabel else 'IMPT Plan'
        ds.RTPlanDate     = datetime.date.today().strftime('%Y%m%d')
        ds.RTPlanTime     = datetime.datetime.now().strftime('%H%M%S.%f')
        ds.RTPlanGeometry = 'PATIENT'
        ds.ApprovalStatus = 'UNAPPROVED'
        ds.InstanceCreationDate = ds.RTPlanDate
        ds.InstanceCreationTime = ds.RTPlanTime

        # Emulate RayStation top-level attributes
        ds.Manufacturer = 'RaySearch Laboratories'
        ds.PlanIntent = 'CURATIVE'
        ds.TreatmentProtocols = ['PencilBeamScanning'] 
        ds.PatientIdentityRemoved = 'YES'
        ds.DeidentificationMethod = 'RayStation Custom Export'

        # Required top-level attributes from RT General Plan Module
        ds.NumberOfBeams = len(self.beam_list)
        ds.NumberOfBrachyApplicationSetups = 0

        # Referenced Structure Set Sequence (required for RTPlanGeometry='PATIENT')
        ds.ReferencedStructureSetSequence = [Dataset()]
        rss = ds.ReferencedStructureSetSequence[0]
        rss.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        rss.ReferencedSOPInstanceUID = generate_uid()  # Replace with actual if available

        # check if Prescriptions is empty
        prescriptions = []
        if not self.Prescriptions:
            rx = Prescription()
            rx.TargetPrescriptionDose = self.n_fractions * 2.0 # assume 2 Gy per fraction as default
            rx.ROIName = "PTV"
            rx.TargetUnderdoseVolumeFraction = 2.0
            prescriptions.append(rx)
        else:
            for rx in self.Prescriptions:
                prescriptions.append(rx)

        # populate prescriptions
        ds = create_dose_reference(ds, prescriptions, rt_struct_file)

        # Fraction Group Sequence
        ds.FractionGroupSequence = [Dataset()]
        fg = ds.FractionGroupSequence[0]
        fg.FractionGroupNumber = 1
        fg.NumberOfFractionsPlanned = self.n_fractions
        fg.NumberOfBeams = len(self.beam_list)
        fg.NumberOfBrachyApplicationSetups = 0  # Critical addition to match RS and DICOM standard
        fg.ReferencedBeamSequence = []

        # Patient Setup Sequence
        ds.PatientSetupSequence = [Dataset()]
        ps = ds.PatientSetupSequence[0]
        ps.PatientPosition = 'HFS'
        ps.PatientSetupNumber = 1

        # ————————————
        # Ion Beam Sequence
        # ————————————
        ds.IonBeamSequence = []
        for i, beam in enumerate(self.beam_list):
            # Find the corresponding beam model
            try:
                model_index = list(self.dicom_rangeshifter_label.astype(str)).index(beam.dicom_rangeshifter_label)
            except ValueError:
                raise ValueError(f"Beam model not found for rangeshifter ID {beam.dicom_rangeshifter_label}")
            beam_model = self.beam_models[model_index]

            dcm_beam = Dataset()
            dcm_beam.BeamNumber = i + 1
            dcm_beam.BeamName = beam.BeamName
            dcm_beam.BeamDescription = beam.BeamDescription
            dcm_beam.BeamType = 'STATIC'
            dcm_beam.RadiationType = 'PROTON'
            dcm_beam.ScanMode = 'MODULATED_SPEC'
            dcm_beam.ModulatedScanModeType = 'STATIONARY'
            dcm_beam.TreatmentMachineName = 'ProBeat_noMRF_'
            dcm_beam.Manufacturer = 'Generic'
            dcm_beam.PrimaryDosimeterUnit = 'MU'
            dcm_beam.TreatmentDeliveryType = 'TREATMENT'
            dcm_beam.NumberOfWedges = 0
            dcm_beam.NumberOfCompensators = 0
            dcm_beam.NumberOfBoli = 0
            dcm_beam.NumberOfBlocks = 0
            dcm_beam.PatientSupportType = 'TABLE'
            dcm_beam.PatientSupportID = 'TABLE'

            # Virtual Source Axis Distances (at beam level)
            dcm_beam.VirtualSourceAxisDistances = [float(beam_model.VSADX), float(beam_model.VSADY)]

            # Snout Sequence
            dcm_beam.SnoutSequence = [Dataset()]
            snout = dcm_beam.SnoutSequence[0]
            snout.SnoutID = '0'

            # Range Shifter
            dcm_beam.NumberOfRangeShifters = 0
            if beam.dicom_rangeshifter_label:
                dcm_beam.NumberOfRangeShifters = 1
                dcm_beam.RangeShifterSequence = [Dataset()]
                rs = dcm_beam.RangeShifterSequence[0]
                rs.RangeShifterNumber = 1
                rs.RangeShifterID = beam.dicom_rangeshifter_label
                rs.RangeShifterType = 'BINARY'
                rs.AccessoryCode = beam.dicom_rangeshifter_label

            # Lateral Spreading Device Sequence 
            dcm_beam.NumberOfLateralSpreadingDevices = 1
            dcm_beam.LateralSpreadingDeviceSequence = [Dataset()]
            lsd = dcm_beam.LateralSpreadingDeviceSequence[0]
            lsd.LateralSpreadingDeviceNumber = 1
            lsd.LateralSpreadingDeviceID = '0'
            lsd.LateralSpreadingDeviceType = 'SCATTERER'

            # Range Modulator Sequence 
            dcm_beam.NumberOfRangeModulators = 1
            dcm_beam.RangeModulatorSequence = [Dataset()]
            rm = dcm_beam.RangeModulatorSequence[0]
            rm.RangeModulatorNumber = 0
            rm.RangeModulatorID = '0'
            rm.RangeModulatorType = 'FIXED'
            rm.RangeModulatorDescription = ''

            # Referenced Beam in Fraction Group
            rb = Dataset()
            rb.ReferencedBeamNumber = dcm_beam.BeamNumber
            total_beam_mu = np.sum(beam.spot_list[:, 2])
            rb.BeamMeterset = float(total_beam_mu)
            fg.ReferencedBeamSequence.append(rb)

            # Final Cumulative Meterset Weight
            dcm_beam.FinalCumulativeMetersetWeight = float(total_beam_mu)

            # Add RaySearch private creator and known attributes (emulate example)
            private_creator_tag = (0x4001, 0x0010)
            dcm_beam[private_creator_tag] = pyd.DataElement(private_creator_tag, 'LO', 'RAYSEARCHLABS 2.0')
            dcm_beam[(0x4001, 0x1002)] = pyd.DataElement((0x4001, 0x1002), 'UN', b'Constant 1.1')
            dcm_beam[(0x4001, 0x1012)] = pyd.DataElement((0x4001, 0x1012), 'UN', b'ProBeat_noMRF_')
            dcm_beam[(0x4001, 0x1033)] = pyd.DataElement((0x4001, 0x1033), 'UN', b'RBE_NOT_INCLUDED')
            dcm_beam[(0x4001, 0x1003)] = pyd.DataElement((0x4001, 0x1003), 'UN', b'20181030034922.000000 ')

            # Ion Control Point Sequence (pair per layer to emulate: first with data, second with 0 weights/same positions)
            unique_energy_ids = np.unique(beam.spot_list[:, 3]).astype(int)
            unique_energy_ids = unique_energy_ids[::-1]
            dcm_beam.IonControlPointSequence = []
            cumulative_mu = 0.0
            for layer_idx, energy_id in enumerate(unique_energy_ids):
                spots_in_layer = beam.spot_list[beam.spot_list[:, 3] == energy_id]
                layer_mu = np.sum(spots_in_layer[:, 2])
                # First CP for layer
                cp = Dataset()
                cp.ControlPointIndex = len(dcm_beam.IonControlPointSequence)
                cp.NominalBeamEnergyUnit = 'MEV'
                cp.NominalBeamEnergy = float(beam_model.energy_labels[energy_id])
                cp.GantryAngle = float(beam.gantry_angle)
                cp.GantryRotationDirection = 'NONE'
                cp.BeamLimitingDeviceAngle = 0.0
                cp.BeamLimitingDeviceRotationDirection = 'NONE'
                cp.PatientSupportAngle = float(beam.couch_angle)
                cp.PatientSupportRotationDirection = 'NONE'
                cp.TableTopEccentricAngle = float(beam.couch_angle)
                cp.TableTopEccentricRotationDirection = 'NONE'
                cp.TableTopPitchAngle = 0.0
                cp.TableTopPitchRotationDirection = 'NONE'
                cp.TableTopRollAngle = 0.0
                cp.TableTopRollRotationDirection = 'NONE'
                cp.GantryPitchAngle = 0.0
                cp.GantryPitchRotationDirection = 'NONE'
                cp.TableTopVerticalPosition = ''
                cp.TableTopLongitudinalPosition = ''
                cp.TableTopLateralPosition = ''
                cp.IsocenterPosition = [float(coord) for coord in beam.iso]
                cp.SnoutPosition = 391.0 
                cp.MetersetRate = 480.0 
                cp.CumulativeMetersetWeight = float(cumulative_mu)
                cp.ScanSpotTuneID = 'Standard' 
                cp.NumberOfScanSpotPositions = len(spots_in_layer)
                cp.ScanSpotPositionMap = spots_in_layer[:, 0:2].flatten().tolist()
                cp.ScanSpotMetersetWeights = spots_in_layer[:, 2].tolist()
                SpotWidth = beam_model.reference_sigmas[energy_id] * 2.0 * np.sqrt(2.0 * np.log(2.0)) * 10.0 
                cp.ScanningSpotSize = [SpotWidth, SpotWidth]
                cp.NumberOfPaintings = 1

                # Range Shifter Setting (per CP)
                if beam.dicom_rangeshifter_label:
                    cp.RangeShifterSetting = 'ON'
                # Lateral Spreading Device Settings Sequence
                cp.LateralSpreadingDeviceSettingsSequence = [Dataset()]
                lsds = cp.LateralSpreadingDeviceSettingsSequence[0]
                lsds.LateralSpreadingDeviceSetting = 'IN'
                lsds.IsocenterToLateralSpreadingDeviceDistance = 2275.0
                lsds.LateralSpreadingDeviceWaterEquivalentThickness = 0.0
                lsds.ReferencedLateralSpreadingDeviceNumber = 1
                # Range Modulator Settings Sequence
                cp.RangeModulatorSettingsSequence = [Dataset()]
                rms = cp.RangeModulatorSettingsSequence[0]
                rms.ReferencedRangeModulatorNumber = 0
                # Referenced Dose Reference Sequence
                cp.ReferencedDoseReferenceSequence = [Dataset(), Dataset()]
                for j, rdr in enumerate(cp.ReferencedDoseReferenceSequence):
                    rdr.ReferencedDoseReferenceNumber = j + 1
                    rdr.CumulativeDoseReferenceCoefficient = None 

                dcm_beam.IonControlPointSequence.append(cp)

                # Second CP for layer (emulate pair: same but weights 0)
                cp2 = Dataset()
                cp2.ControlPointIndex = len(dcm_beam.IonControlPointSequence)
                cp2.NominalBeamEnergyUnit = 'MEV'
                cp2.NominalBeamEnergy = float(beam_model.energy_labels[energy_id])
                cp2.GantryAngle = float(beam.gantry_angle)
                cp2.GantryRotationDirection = 'NONE'
                cp2.BeamLimitingDeviceAngle = 0.0
                cp2.BeamLimitingDeviceRotationDirection = 'NONE'
                cp2.PatientSupportAngle = float(beam.couch_angle)
                cp2.PatientSupportRotationDirection = 'NONE'
                cp2.TableTopEccentricAngle = float(beam.couch_angle)
                cp2.TableTopEccentricRotationDirection = 'NONE'
                cp2.TableTopPitchAngle = 0.0
                cp2.TableTopPitchRotationDirection = 'NONE'
                cp2.TableTopRollAngle = 0.0
                cp2.TableTopRollRotationDirection = 'NONE'
                cp2.GantryPitchAngle = 0.0
                cp2.GantryPitchRotationDirection = 'NONE'
                cp2.TableTopVerticalPosition = ''
                cp2.TableTopLongitudinalPosition = ''
                cp2.TableTopLateralPosition = ''
                cp2.IsocenterPosition = [float(coord) for coord in beam.iso]
                cp2.SnoutPosition = 391.0
                cp2.MetersetRate = 480.0
                cp2.CumulativeMetersetWeight = float(cumulative_mu + layer_mu)
                cp2.ScanSpotTuneID = 'Standard'
                cp2.NumberOfScanSpotPositions = len(spots_in_layer)
                cp2.ScanSpotPositionMap = spots_in_layer[:, 0:2].flatten().tolist()
                cp2.ScanSpotMetersetWeights = [0.0] * len(spots_in_layer)
                SpotWidth = beam_model.reference_sigmas[energy_id] * 2.0 * np.sqrt(2.0 * np.log(2.0)) * 10.0 
                cp2.ScanningSpotSize = [SpotWidth, SpotWidth]
                cp2.NumberOfPaintings = 1
                if beam.dicom_rangeshifter_label:
                    cp2.RangeShifterSetting = 'ON'
                cp2.LateralSpreadingDeviceSettingsSequence = cp.LateralSpreadingDeviceSettingsSequence  # Copy
                cp2.RangeModulatorSettingsSequence = cp.RangeModulatorSettingsSequence  # Copy
                cp2.ReferencedDoseReferenceSequence = cp.ReferencedDoseReferenceSequence  # Copy

                dcm_beam.IonControlPointSequence.append(cp2)

                cumulative_mu += layer_mu

            dcm_beam.NumberOfControlPoints = len(dcm_beam.IonControlPointSequence)
            ds.IonBeamSequence.append(dcm_beam)

        # Save the DICOM file
        ds.save_as(output_file, write_like_original=False)

    def readPlanDicom(self, plan_path):

        ds = pyd.dcmread(plan_path, force=True)
        n_beams = len(ds.IonBeamSequence)
        self.n_fractions = float(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        self.beam_list = []
        self.n_beams = 0

        for i in range(n_beams):

            ibs = ds.IonBeamSequence[i]
            
            beam = IMPTBeam()

            if ibs.NumberOfRangeShifters and ibs.RangeShifterSequence[0].RangeShifterID is not None:
                beam.dicom_rangeshifter_label = ibs.RangeShifterSequence[0].RangeShifterID
            else:
                beam.dicom_rangeshifter_label = '0'

            beam.gantry_angle = float(ibs.IonControlPointSequence[0].GantryAngle)
            beam.couch_angle = float(ibs.IonControlPointSequence[0].PatientSupportAngle)
            beam.iso = np.array(ibs.IonControlPointSequence[0].IsocenterPosition, dtype=np.single)

            if hasattr(ibs, 'BeamName'):
                beam.BeamName = ibs.BeamName

            if hasattr(ibs, 'BeamDescription'):
                beam.BeamDescription = ibs.BeamDescription

            for j in range(len(ibs.IonControlPointSequence)):
            
                if j % 2 == 0:
            
                    cp = ibs.IonControlPointSequence[j]

                    energy_id = self.beam_models[0].energyIDFromLabel(float(cp.NominalBeamEnergy))

                    beam.addSpotData(cp, energy_id)

            self.addBeam(beam)