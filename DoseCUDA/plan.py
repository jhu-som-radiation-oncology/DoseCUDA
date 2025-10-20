import numpy as np
import SimpleITK as sitk
import pydicom as pyd
import pydicom.uid
from datetime import datetime
import math


def _create_uid(pt_id: str):
    """Create a helpfully unique UID"""
    ts = datetime.now().timestamp()
    sub, sec = math.modf(ts)
    idstr = "".join(c for c in pt_id if c.isnumeric())
    pid = int(idstr) if idstr else 0
    uid = f"{pydicom.uid.PYDICOM_ROOT_UID}{pid}.{int(sec)}.{int(sub * 1e6)}"
    return pydicom.uid.UID(uid)


class VolumeObject:

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.voxel_data = []


class Prescription:

    def __init__(self):
        self.TargetPrescriptionDose = 0.0
        self.ROIName = None
        self.TargetUnderdoseVolumeFraction = 0.0


class DoseGrid:

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.size = np.array([0, 0, 0])
        self.HU = []
        self.dose = []
        self.beam_doses = []
        self.FrameOfReferenceUID = ""

    def loadCTNRRD(self, ct_path):
        fr = sitk.ImageFileReader()
        fr.SetFileName(ct_path)
        ct_img = fr.Execute()

        self.origin = np.array(ct_img.GetOrigin())
        self.spacing = np.array(ct_img.GetSpacing())
        self.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        self.size = np.array(self.HU.shape)

    def loadCTDCM(self, ct_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_path)

        dicom_names = list(dicom_names)
        dicom_names.sort(key=lambda x: pyd.dcmread(x, force=True).ImagePositionPatient[2])

        reader.SetFileNames(dicom_names)
        ct_img = reader.Execute()

        self.origin = np.array(ct_img.GetOrigin())
        self.spacing = np.array(ct_img.GetSpacing())
        self.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        self.HU = np.clip(self.HU, -1000.0, None)
        self.size = np.array(self.HU.shape)

    def resampleCT(self, new_spacing, new_size, new_origin):
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(new_origin)
        rf.SetOutputSpacing(new_spacing)
        rf.SetSize(new_size)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.origin = new_origin
        self.spacing = new_spacing
        self.size = np.array(self.HU.shape)

    def resampleCTfromSpacing(self, spacing):

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(self.origin)
        sp_new = (spacing, spacing, spacing)
        sz_new = (int(self.size[2] * self.spacing[0] / sp_new[0]),
                  int(self.size[1] * self.spacing[1] / sp_new[1]),
                  int(self.size[0] * self.spacing[2] / sp_new[2]))
        rf.SetOutputSpacing(sp_new)
        rf.SetSize(sz_new)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.spacing = sp_new
        self.size = np.array(self.HU.shape)

    def resampleCTfromReferenceDose(self, ref_dose):

        if not isinstance(ref_dose, pydicom.Dataset):
            ref_dose = pyd.dcmread(ref_dose, force=True)
        slice_thickness = float(ref_dose.GridFrameOffsetVector[1]) - float(ref_dose.GridFrameOffsetVector[0])
        ref_spacing = np.array([float(ref_dose.PixelSpacing[0]), float(ref_dose.PixelSpacing[1]), slice_thickness])
        ref_origin = np.array(ref_dose.ImagePositionPatient)

        ref_dose_img = sitk.GetImageFromArray(ref_dose.pixel_array)
        ref_dose_img.SetOrigin(ref_origin)
        ref_dose_img.SetSpacing(ref_spacing)

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        rf = sitk.ResampleImageFilter()
        rf.SetReferenceImage(ref_dose_img)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)

        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.size = np.array(self.HU.shape)
        self.origin = ref_origin
        self.spacing = ref_spacing

    def applyCouchModel(self, couch_wet=8.0):
        spacing = self.spacing[0]
        n_voxels = int(50.0 / spacing)
        hu_override_value = ((couch_wet / (n_voxels * spacing)) - 1.0) * 1000.0

        self.HU[:, -n_voxels:, :] = hu_override_value

    def streamDoseDCM(self, ref, dose_type="EFFECTIVE", individual_beams=False):
        """Flush the dose volume to a DICOM RTDose dataset"""
        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception(f"Unknown dose type: {dose_type}")

        if not isinstance(ref, pydicom.Dataset):
            ref = pyd.dcmread(ref, force=True)

        # Optional tags
        TAGS = [
            "PatientName",
            "PatientBirthDate",
            "PatientBirthTime",
            "PatientSex",
            "StudyDate",
            "StudyTime",
            "AccessionNumber",
            "StudyID",
            "StudyDescription",
            "ReferringPhysicianName",
            "PatientAge",
            "PatientSize",
            "PatientWeight",
            "BodyPartExamined",
            "FrameOfReferenceUID",
            "PositionReferenceIndicator"
        ]

        ts = datetime.now()
        template = pydicom.Dataset()
        template.SpecificCharacterSet  = r"ISO_IR 100"
        template.SOPClassUID           = pydicom.uid.RTDoseStorage
        template.Modality              = r"RTDOSE"
        template.Manufacturer          = r"SKCCC"
        template.ManufacturerModelName = r"DoseCUDA"
        template.SoftwareVersions      = None
        template.PatientID             = ref.PatientID
        template.StudyInstanceUID      = ref.StudyInstanceUID
        template.SeriesInstanceUID     = _create_uid(ref.PatientID)
        template.SeriesDescription     = f"{ref.get('SeriesDescription') or ''}_DoseCUDA"
        template.SeriesDate            = ts.date().strftime("%Y%m%d")
        template.SeriesTime            = ts.time().strftime("%H%M%S.%f")
        template.SeriesNumber          = None
        template.OperatorsName         = None

        for tag in TAGS:
            setattr(template, tag, ref.get(tag))

        template.InstanceCreationDate      = ts.date().strftime("%Y%m%d")
        template.InstanceCreationTime      = ts.time().strftime("%H%M%S.%f")
        template.SliceThickness            = self.spacing[0]
        template.ImagePositionPatient      = [x for x in self.origin]
        template.ImageOrientationPatient   = [1, 0, 0, 0, 1, 0]
        template.SamplesPerPixel           = 1
        template.PhotometricInterpretation = r"MONOCHROME2"
        template.NumberOfFrames            = int(self.size[0])
        template.FrameIncrementPointer     = (0x3004, 0x000c)
        template.GridFrameOffsetVector     = [self.spacing[0] * i for i in range(0, self.size[0])]
        template.Rows                      = int(self.size[1])
        template.Columns                   = int(self.size[2])
        template.PixelSpacing              = [self.spacing[1], self.spacing[2]]

        ptype = np.uint16
        signed = 1 if np.iinfo(ptype).min < 0 else 0
        template.BitsAllocated          = np.iinfo(ptype).bits
        template.BitsStored             = np.iinfo(ptype).bits
        template.HighBit                = np.iinfo(ptype).bits - 1 - signed
        template.PixelRepresentation    = signed

        template.DoseUnits         = r"GY"
        template.DoseType          = dose_type
        template.DoseSummationType = r"BEAM" if individual_beams else r"PLAN"
        template.TissueHeterogeneityCorrection = [r"IMAGE", r"ROI_OVERRIDE"]

        template.ReferencedRTPlanSequence = pydicom.Sequence([pydicom.Dataset()])
        if ref.SOPClassUID == pydicom.uid.RTDoseStorage:
            template.ReferencedRTPlanSequence[0].ReferencedSOPClassUID = ref.get("ReferencedRTPlanSequence", [{ }])[0].get("ReferencedSOPClassUID")
            template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = ref.get("ReferencedRTPlanSequence", [{ }])[0].get("ReferencedSOPInstanceUID")
        elif ref.SOPClassUID == pydicom.uid.RTIonPlanStorage or ref.SOPClassUID == pydicom.uid.RTPlanStorage:
            template.ReferencedRTPlanSequence[0].ReferencedSOPClassUID = ref.SOPClassUID
            template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = ref.SOPInstanceUID
        else:
            raise Exception("DICOM template SOP class is invalid")

        if individual_beams:
            template.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence = pydicom.Sequence([pydicom.Dataset()])
            template.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedFractionGroupNumber = 1
            template.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence = pydicom.Sequence([pydicom.Dataset()])
            beam_doses = self.beam_doses
        else:
            beam_doses = [self.dose]

        for inst, beam_dose in enumerate(beam_doses):
            template.SOPInstanceUID = _create_uid(template.PatientID)

            if individual_beams:
                template.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber = inst + 1
                template.InstanceNumber = inst + 1
            else:
                template.InstanceNumber = None

            scal = RBE * np.max(beam_dose) / float(np.iinfo(ptype).max)
            template.DoseGridScaling = f"{scal:16g}"
            template.PixelData = np.array(beam_dose * (RBE / scal), dtype=ptype).tobytes()
            yield template

    def writeDoseDCM(self, dose_path, ref_dose_path, dose_type="EFFECTIVE", individual_beams=False):

        if not dose_path.endswith(".dcm"):
            raise Exception("Dose path must have .dcm extension")
        else:
            print("test")

        for i, dose in enumerate(self.streamDoseDCM(ref_dose_path, dose_type, individual_beams)):
            path = dose_path
            if individual_beams:
                path = path.replace(".dcm", "_beam%02i.dcm" % (i + 1))
            dose.save_as(path, enforce_file_format=True, implicit_vr=True, little_endian=True)

    def writeDoseNRRD(self, dose_path, individual_beams=False, dose_type="EFFECTIVE"):

        if not dose_path.endswith(".nrrd"):
            raise Exception("Dose path must have .nrrd extension")

        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception("Unknown dose type: %s" % dose_type)

        fw = sitk.ImageFileWriter()
        dose_img = sitk.GetImageFromArray(np.array(self.dose * RBE, dtype=np.single))
        dose_img.SetOrigin(self.origin)
        dose_img.SetSpacing(self.spacing)

        if individual_beams:
            for i, beam_dose in enumerate(self.beam_doses):
                dose_img = sitk.GetImageFromArray(np.array(beam_dose * RBE, dtype=np.single))
                dose_img.SetOrigin(self.origin)
                dose_img.SetSpacing(self.spacing)
                fw.SetFileName(dose_path.replace(".nrrd", "_beam%02i.nrrd" % (i+1)))
                fw.Execute(dose_img)
        else:
            fw.SetFileName(dose_path)
            fw.Execute(dose_img)

    def writeCTNRRD(self, ct_path):

        if not ct_path.endswith(".nrrd"):
            raise Exception("CT path must have .nrrd extension")

        fw = sitk.ImageFileWriter()
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def writeCTNIFTI(self, ct_path):
        if not ct_path.endswith(".nii.gz"):
            raise Exception("CT path must have .nii.gz extension")

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(self.origin)
        HU_img.SetSpacing(self.spacing)

        fw = sitk.ImageFileWriter()
        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def createCubePhantom(self, size=[138, 138, 138], spacing=3.0):
        self.origin = np.array([-size[0] * spacing / 2.0, -size[1] * spacing / 2.0, -size[2] * spacing / 2.0])
        self.spacing = np.array([spacing, spacing, spacing])
        self.size = np.array(size)
        edge = round(10.0 / spacing)
        self.HU = np.ones(size, dtype=np.single) * -1000.0
        self.HU[edge:-edge, edge:-edge, edge:-edge] = 0.0


class Beam:

    def __init__(self):
        self.iso = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.BeamName = None
        self.BeamDescription = None


class Plan:

    def __init__(self):
        self.n_beams = 0
        self.n_fractions = 1
        self.beam_list = []
        self.RTPlanLabel = None
        self.Prescriptions = []

    def addPrescription(self, TargetPrescriptionDose, ROIName, TargetUnderdoseVolumeFraction):
        rx = Prescription()
        rx.TargetPrescriptionDose = TargetPrescriptionDose
        rx.ROIName = ROIName
        rx.TargetUnderdoseVolumeFraction = TargetUnderdoseVolumeFraction
        self.Prescriptions.append(rx)

    def addBeam(self, beam):
        if not beam.BeamName:
            beam.BeamName = f'PBS_Beam{self.n_beams + 1}'
        if not beam.BeamDescription:
            beam.BeamDescription = f'PBS_Beam {self.n_beams + 1}'
        self.beam_list.append(beam)
        self.n_beams += 1
