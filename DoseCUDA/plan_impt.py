from .plan import Plan, Beam, DoseGrid, VolumeObject
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import dose_kernels
import os
import SimpleITK as sitk


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

    def __init__(self):
        super().__init__()

        self.spot_list = []
        self.n_spots = 0
        self.dicom_rangeshifter_label = None

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

    def readPlanDicom(self, plan_path):

        ds = pyd.dcmread(plan_path)
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
            
            for j in range(len(ibs.IonControlPointSequence)):
            
                if j % 2 == 0:
            
                    cp = ibs.IonControlPointSequence[j]

                    energy_id = self.beam_models[0].energyIDFromLabel(float(cp.NominalBeamEnergy))

                    beam.addSpotData(cp, energy_id)

            self.addBeam(beam)