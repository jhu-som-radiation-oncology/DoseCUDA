import pydicom as pyd
import pydicom.uid
import numpy as np
import math
import copy
from datetime import datetime
from typing import Union


class RTDoseFactory:
    """
    A factory class designed to create multiple RTDose files from a given
    template
    """

    def _add_rtplan_tags(self, rp: pyd.Dataset):
        """
        Copy all necessary data from the RTPlan dataset into the template
        dataset
        """
        keys = [
            "StudyDate",
            "SeriesDate",
            "StudyTime",
            "SeriesTime",
            "AccessionNumber",
            "ReferringPhysicianName",
            "SeriesDescription",
            "OperatorsName",
            "PatientName",
            "PatientID",
            "PatientBirthDate",
            "PatientSex",
            "StudyInstanceUID",
            "StudyID",
            "SeriesNumber",
            "FrameOfReferenceUID",
            "PositionReferenceIndicator"
        ]
        for key in keys:
            self.template[key] = rp[key]


    def _add_const_tags(self):
        """
        Add to the template dataset any tags that do not depend on input data
        """
        tags = [
            ("SpecificCharacterSet",        r"ISO_IR 100"               ),
            ("SOPClassUID",                 pydicom.uid.RTDoseStorage   ),
            ("Modality",                    r"RTDOSE"                   ),
            ("Manufacturer",                r"SKCCC"                    ),
            ("ManufacturerModelName",       r"DoseCUDA"                 ),
            ("SoftwareVersions",            None                        ),
        ]
        for key, value in tags:
            setattr(self.template, key, value)


    def _create_uid(self):
        """
        Create a helpfully unique UID
        """
        ts = datetime.now().timestamp()
        sub, sec = math.modf(ts)
        idstr = "".join(c for c in self.template.PatientID if c.isnumeric())
        pid = int(idstr) if idstr else 0
        uid = f"{pydicom.uid.PYDICOM_ROOT_UID}{pid}.{int(sec)}.{int(sub * 1e6)}"
        return pydicom.uid.UID(uid)


    def __init__(self, rtplan: Union[str, pyd.Dataset]):
        """
        Create the factory from an RTPlan: either a path to the file on disk or
        the pydicom dataset in memory
        """
        if isinstance(rtplan, pyd.Dataset):
            ds = rtplan
        else:
            ds = pyd.dcmread(rtplan, force=True)

        self.template = pyd.Dataset()
        self._add_rtplan_tags(ds)
        self._add_const_tags()

        self.rp_cls_uid  = ds["SOPClassUID"].value
        self.rp_inst_uid = ds["SOPInstanceUID"].value


    def create_rtdose(self, beam_dose, origin, spacing, size, dose_type, inst_no = 0) -> pyd.Dataset:
        """
        Create an RTDose dataset

        :param beam_dose: A NumPy array of dose data in row-major order
        :param origin: A list of three floats to be used as ImagePositionPatient
        :param spacing: A list of three floats to be used as the pixel spacing
        :param size: The volume dimensions in voxels, row-major
        :param dose_type: The dose type string, 'EFFECTIVE' or 'PHYSICAL'
        :param inst_no: The beam number, or zero for composite dose
        """

        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception(f"Unknown dose type: {dose_type}")

        ds = copy.deepcopy(self.template)
        ts = datetime.now()

        ds.InstanceCreationDate = ts.date()
        ds.InstanceCreationTime = ts.time()
        ds.SOPInstanceUID       = self._create_uid()
        ds.SliceThickness       = spacing[0]

        ds.SeriesInstanceUID = f"{self.rp_inst_uid}.0"
        ds.InstanceNumber    = inst_no or 1

        ds.ImagePositionPatient      = [pt for pt in origin]
        ds.ImageOrientationPatient   = [1, 0, 0, 0, 1, 0]
        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = r"MONOCHROME2"
        ds.NumberOfFrames            = int(size[0])
        ds.FrameIncrementPointer     = ( 0x3004, 0x000c )
        ds.Rows                      = int(size[1])
        ds.Columns                   = int(size[2])
        ds.PixelSpacing              = [spacing[1], spacing[2]]

        # Always uint16 (could become an argument/class attribute later)
        ptype = np.uint16
        signed = 1 if np.iinfo(ptype).min < 0 else 0

        ds.BitsAllocated             = np.iinfo(ptype).bits
        ds.BitsStored                = np.iinfo(ptype).bits
        ds.HighBit                   = np.iinfo(ptype).bits - 1 - signed
        ds.PixelRepresentation       = signed

        ds.DoseUnits                 = r"GY"
        ds.DoseType                  = dose_type
        ds.DoseSummationType         = r"BEAM" if inst_no > 0 else r"PLAN"

        scal = float(np.iinfo(ptype).max) / np.max(beam_dose)
        scal = scal / RBE
        ds.GridFrameOffsetVector     = [spacing[0] * i for i in range(0, size[0])]
        ds.DoseGridScaling           = f"{1.0 / scal:16g}"
        ds.TissueHeterogeneityCorrection = [r"IMAGE", r"ROI_OVERRIDE"]   # Unsure

        ref_rtp_seq = pyd.Dataset()
        ref_rtp_seq.ReferencedSOPClassUID    = self.rp_cls_uid
        ref_rtp_seq.ReferencedSOPInstanceUID = self.rp_inst_uid

        if inst_no > 0:
            ref_beam_seq = pyd.Dataset()
            ref_beam_seq.ReferencedBeamNumber = inst_no

            ref_fx_gp_seq = pyd.Dataset()
            ref_fx_gp_seq.ReferencedBeamSequence = pyd.Sequence([ref_beam_seq])
            ref_fx_gp_seq.ReferencedFractionGroupNumber = 1

            ref_rtp_seq.ReferencedFractionGroupSequence = pyd.Sequence([ref_fx_gp_seq])

        ds.ReferencedRTPlanSequence = pyd.Sequence([ref_rtp_seq])

        ds.PixelData = np.array(beam_dose * scal * RBE, dtype=ptype).tobytes()

        return ds
