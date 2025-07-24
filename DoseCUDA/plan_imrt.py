from .plan import Plan, Beam, DoseGrid, VolumeObject
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import dose_kernels
from dataclasses import dataclass

@dataclass
class IMRTPhotonEnergy:

    def __init__(self, dicom_energy_label):
        self.dicom_energy_label = dicom_energy_label
        self.output_factor_equivalent_squares = None
        self.output_factor_values = None
        self.mu_calibration = None
        self.primary_source_distance = None
        self.scatter_source_distance = None
        self.mlc_distance = None
        self.scatter_source_weight = None
        self.electron_attenuation = None
        self.primary_source_size = None
        self.scatter_source_size = None
        self.profile_radius = None
        self.profile_intensities = None
        self.profile_softening = None
        self.spectrum_attenuation_coefficients = None
        self.spectrum_primary_weights = None
        self.spectrum_scatter_weights = None
        self.electron_source_weight = None
        self.has_xjaws = None
        self.has_yjaws = None
        self.electron_fitted_dmax = None
        self.jaw_transmission = None
        self.mlc_transmission = None
        self.mlc_index = None
        self.mlc_widths = None
        self.mlc_offsets = None
        self.n_mlc_pairs = None
        self.kernel = None

    def validate_parameters(self):
        """Validate that all required parameters are set"""
        required_params = [
            'output_factor_equivalent_squares', 'output_factor_values', 'mu_calibration',
            'primary_source_distance', 'scatter_source_distance', 'mlc_distance',
            'scatter_source_weight', 'electron_attenuation', 'primary_source_size',
            'scatter_source_size', 'profile_radius', 'profile_intensities',
            'profile_softening', 'spectrum_attenuation_coefficients', 'spectrum_primary_weights',
            'spectrum_scatter_weights', 'electron_source_weight', 'has_xjaws', 'has_yjaws',
            'electron_fitted_dmax', 'jaw_transmission', 'mlc_transmission'
        ]
        
        for param in required_params:
            if getattr(self, param) is None:
                raise Exception(f"{param} not set in beam model")
        
    def outputFactor(self, cp):
    
        min_y = 10000.0
        max_y = -10000.0
        max_x_diff = 0.0
        area = 0.0
        
        for i in range(cp.mlc.shape[0]):

            x1 = cp.mlc[i, 0]
            x2 = cp.mlc[i, 1]
            y_offset = cp.mlc[i, 2]
            y_width = cp.mlc[i, 3]

            area += (x2 - x1) * y_width

            if (x2 - x1) > 3.0 and (x2 - x1) > max_x_diff:
                max_x_diff = x2 - x1

            if (x2 - x1) > 3.0:
                if (y_offset - y_width / 2.0) < min_y:
                    min_y = (y_offset - y_width / 2.0)
                if (y_offset + y_width / 2.0) > max_y:
                    max_y = (y_offset + y_width / 2.0)

        perimeter = 2 * (max_y - min_y) + 2 * max_x_diff

        equivalent_square = 4 * area / perimeter

        output_factor = np.interp(equivalent_square, self.output_factor_equivalent_squares, self.output_factor_values)

        return output_factor
       

class IMRTControlPoint:

    def __init__(self, iso, mu, mlc, ga, ca, ta, xjaws, yjaws):
        self.iso = iso
        self.mu = mu
        self.mlc = mlc
        self.ga = ga
        self.ca = ca
        self.ta = ta
        self.xjaws = xjaws
        self.yjaws = yjaws


class IMRTDoseGrid(DoseGrid):
        
    def __init__(self):
        super().__init__()
        self.Density = []

    def DensityFromHU(self, machine_name):
                
        density_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "photons", machine_name, "HU_Density.csv"))
        df_density = pd.read_csv(density_table_path)

        hu_curve = df_density["HU"].to_numpy()
        density_curve = df_density["Density"].to_numpy()
        
        density = np.array(np.interp(self.HU, hu_curve, density_curve), dtype=np.single)
        
        return density

    def computeIMRTPlan(self, plan, gpu_id=0):
            
        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.Density = self.DensityFromHU(plan.machine_name)

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")
        
        density_object = VolumeObject()
        density_object.voxel_data = np.array(self.Density, dtype=np.single)
        density_object.origin = np.array(self.origin, dtype=np.single)
        density_object.spacing = np.array(self.spacing, dtype=np.single)

        for beam in plan.beam_list:
            beam_dose = np.zeros(self.Density.shape, dtype=np.single)
            
            try:           
                model_index = list(plan.dicom_energy_label.astype(str)).index(beam.dicom_energy_label)
            except ValueError:
                print("Beam model not found for beam energy %s" % beam.dicom_energy_label)
                sys.exit(1)
            beam_model = plan.beam_models[model_index]

            for cp in beam.cp_list:
                output_factor = beam_model.outputFactor(cp)
                cp_dose = dose_kernels.photon_dose_cuda(beam_model, density_object, cp, gpu_id)
                beam_dose += cp_dose * output_factor
                
            self.beam_doses.append(beam_dose)
            self.dose += beam_dose

        self.dose *= plan.n_fractions


class IMRTBeam(Beam):

    def __init__(self):
        super().__init__()
        self.cp_list = []
        self.n_cps = 0
        self.dicom_energy_label = None

    def addControlPoint(self, cp):
        self.cp_list.append(cp)
        self.n_cps += 1


class IMRTPlan(Plan):

    def __init__(self, machine_name = "VarianTrueBeamHF"):
        
        super().__init__()

        self.machine_name = machine_name

        energy_list = pd.read_csv(pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "photons", machine_name, "energy_labels.csv")))
        self.dicom_energy_label = energy_list["dicom_energy_label"]
        self.folder_energy_label = energy_list["folder_energy_label"]

        self.beam_models = []
        for d, f in zip(self.dicom_energy_label, self.folder_energy_label):
            beam_model = IMRTPhotonEnergy(d)
            self._load_beam_model_parameters(beam_model, machine_name, f)
            self.beam_models.append(beam_model)

    def _load_beam_model_parameters(self, beam_model, machine_name, folder_energy_label):
        """Load beam model parameters from lookup tables"""
        path_to_model = os.path.join("lookuptables", "photons", machine_name)
        
        # Load MLC geometry
        mlc_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "mlc_geometry.csv"))
        mlc_geometry = pd.read_csv(mlc_geometry_path)
        
        beam_model.mlc_index = mlc_geometry["mlc_pair_index"].to_numpy()
        beam_model.mlc_widths = mlc_geometry["width"].to_numpy()
        beam_model.mlc_offsets = mlc_geometry["center_offset"].to_numpy()
        beam_model.n_mlc_pairs = len(beam_model.mlc_index)

        # Load kernel
        kernel_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "kernel.csv"))
        kernel = pd.read_csv(kernel_path)
        beam_model.kernel = np.array(kernel.to_numpy(), dtype=np.single)

        # Load machine geometry
        machine_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "machine_geometry.csv"))
        self._load_machine_geometry(beam_model, machine_geometry_path)

        # Load beam parameters
        beam_parameter_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, folder_energy_label, "beam_parameters.csv"))
        self._load_beam_parameters(beam_model, beam_parameter_path)

        # Validate all parameters are loaded
        beam_model.validate_parameters()

    def _load_machine_geometry(self, beam_model, machine_geometry_path):
        """Load machine geometry parameters"""
        for line in open(machine_geometry_path):
            if line.startswith('primary_source_distance'):
                beam_model.primary_source_distance = float(line.split(',')[1])
            elif line.startswith('scatter_source_distance'):
                beam_model.scatter_source_distance = float(line.split(',')[1])
            elif line.startswith('mlc_distance'):
                beam_model.mlc_distance = float(line.split(',')[1])
            elif line.startswith('has_xjaws'):
                beam_model.has_xjaws = bool(line.split(',')[1])
            elif line.startswith('has_yjaws'):
                beam_model.has_yjaws = bool(line.split(',')[1])

    def _load_beam_parameters(self, beam_model, beam_parameter_path):
        """Load beam-specific parameters"""
        for line in open(beam_parameter_path):
            if line.startswith('output_factor_equivalent_squares'):
                beam_model.output_factor_equivalent_squares = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('output_factor_values'):
                beam_model.output_factor_values = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('mu_calibration'):
                beam_model.mu_calibration = float(line.split(',')[1])
            elif line.startswith('scatter_source_weight'):
                beam_model.scatter_source_weight = float(line.split(',')[1])
            elif line.startswith('electron_attenuation'):
                beam_model.electron_attenuation = float(line.split(',')[1])
            elif line.startswith('primary_source_size'):
                beam_model.primary_source_size = float(line.split(',')[1])
            elif line.startswith('scatter_source_size'):
                beam_model.scatter_source_size = float(line.split(',')[1])
            elif line.startswith('profile_radius'):
                beam_model.profile_radius = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('profile_intensities'):
                beam_model.profile_intensities = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('profile_softening'):
                beam_model.profile_softening = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_attenuation_coefficients'):
                beam_model.spectrum_attenuation_coefficients = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_primary_weights'):
                beam_model.spectrum_primary_weights = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('spectrum_scatter_weights'):
                beam_model.spectrum_scatter_weights = np.array(line.split(',')[1:], dtype=np.single)
            elif line.startswith('electron_source_weight'):
                beam_model.electron_source_weight = float(line.split(',')[1])
            elif line.startswith('electron_fitted_dmax'):
                beam_model.electron_fitted_dmax = float(line.split(',')[1])
            elif line.startswith('jaw_transmission'):
                beam_model.jaw_transmission = float(line.split(',')[1])
            elif line.startswith('mlc_transmission'):
                beam_model.mlc_transmission = float(line.split(',')[1])

    def readPlanDicom(self, plan):

        if not isinstance(plan, pyd.Dataset):
            ds = pyd.dcmread(plan, force=True)
        else:
            ds = plan

        total_mu = ds.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset

        self.n_fractions = float(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        self.beam_list = []
        self.n_beams = 0

        for beam in ds.BeamSequence:
            if beam.TreatmentDeliveryType == "TREATMENT":

                imrt_beam = IMRTBeam()
                
                for cp in beam.ControlPointSequence:
                    cpi = cp.ControlPointIndex
                    if cpi == 0:
                        imrt_beam.dicom_energy_label = cp.NominalBeamEnergy
                        imrt_beam.iso = np.array([cp.IsocenterPosition[0], cp.IsocenterPosition[1], cp.IsocenterPosition[2]], dtype=np.single)
                        cumulative_mu = 0.0
                        ca = cp.BeamLimitingDeviceAngle
                        ta = cp.PatientSupportAngle
                        ga = cp.GantryAngle

                        xjaws, yjaws, mlc = None, None, None
                        for ps in cp.BeamLimitingDevicePositionSequence:
                            if ps.RTBeamLimitingDeviceType == 'X':
                                xjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                            if ps.RTBeamLimitingDeviceType == 'Y':
                                yjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                            if ps.RTBeamLimitingDeviceType == 'MLCX':
                                mlc = np.array(cp.BeamLimitingDevicePositionSequence[2].LeafJawPositions, dtype=np.single)

                    else:

                        mu = cp.CumulativeMetersetWeight - cumulative_mu

                        if mlc is not None:
                            mlc = np.reshape(mlc, (2, self.beam_models[0].n_mlc_pairs))
                            mlc = np.array(np.vstack((mlc, self.beam_models[0].mlc_offsets.reshape(1, -1), self.beam_models[0].mlc_widths.reshape(1, -1))), dtype=np.single)
                            mlc = np.transpose(mlc)

                        else:

                            mlc = np.zeros((2, self.beam_models[0].n_mlc_pairs), dtype=np.single)

                            if xjaws is not None:
                                mlc[0, :] = xjaws[0]
                                mlc[1, :] = xjaws[1]

                            if yjaws is not None:
                                for i in range(self.beam_models[0].n_mlc_pairs):
                                    if(self.beam_models[0].mlc_offsets[i] < yjaws[0]) or (self.beam_models[0].mlc_offsets[i] > yjaws[1]):
                                        mlc[0, i] = 0.0
                                        mlc[1, i] = 0.0
                            
                            mlc = np.array(np.vstack((mlc, self.beam_models[0].mlc_offsets.reshape(1, -1), self.beam_models[0].mlc_widths.reshape(1, -1))), dtype=np.single)
                            mlc = np.transpose(mlc)

                        control_point = IMRTControlPoint(imrt_beam.iso, mu * total_mu, mlc, ga, ca, ta, xjaws, yjaws)
                        
                        imrt_beam.addControlPoint(control_point)

                        if hasattr(cp, 'GantryAngle'):
                            ga = cp.GantryAngle

                        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                            xjaws, yjaws, mlc = None, None, None
                            for ps in cp.BeamLimitingDevicePositionSequence:
                                if ps.RTBeamLimitingDeviceType == 'X':
                                    xjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                                if ps.RTBeamLimitingDeviceType == 'Y':
                                    yjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                                if ps.RTBeamLimitingDeviceType == 'MLCX':
                                    mlc = np.array(cp.BeamLimitingDevicePositionSequence[2].LeafJawPositions, dtype=np.single)

                        cumulative_mu = cp.CumulativeMetersetWeight
                
                self.addBeam(imrt_beam)

    def addSquareField(self, dicom_energy_label='6', dimx=10, dimy=10, mu=100, gantry_angle=0.0, collimator_angle=0.0, table_angle=0.0):
    

        imrt_beam = IMRTBeam()
        imrt_beam.dicom_energy_label = dicom_energy_label
        iso = np.array([0.0, 0.0, 0.0], dtype=np.single)

        dimx *= 10.0 # convert to mm
        dimy *= 10.0 # convert to mm

        # create a square field
        mlc = np.zeros((2, self.beam_models[0].n_mlc_pairs), dtype=np.single)
        mlc[0, :] = -dimx / 2.0
        mlc[1, :] = dimx / 2.0

        for i in range(self.beam_models[0].n_mlc_pairs):
            if(self.beam_models[0].mlc_offsets[i] < -(dimy / 2.0)) or (self.beam_models[0].mlc_offsets[i] > (dimy / 2.0)):
                mlc[0, i] = 0.0
                mlc[1, i] = 0.0

        mlc = np.array(np.vstack((mlc, self.beam_models[0].mlc_offsets.reshape(1, -1), self.beam_models[0].mlc_widths.reshape(1, -1))), dtype=np.single)
        mlc = np.transpose(mlc)

        jawx = np.array([-dimx / 2.0, dimx / 2.0], dtype=np.single)
        jawy = np.array([-dimy / 2.0, dimy / 2.0], dtype=np.single)


        cp = IMRTControlPoint(iso, mu, mlc, gantry_angle, collimator_angle, table_angle, jawx, jawy)
        imrt_beam.addControlPoint(cp)
        self.addBeam(imrt_beam)