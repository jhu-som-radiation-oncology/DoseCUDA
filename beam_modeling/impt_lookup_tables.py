import sys
import os
import pydicom as pyd
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian_2d(xy, a, x0, y0, sigma_x, sigma_y):
    x, y = xy
    return a * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


def fit_sigmas(dose_array, spacing):

    sigmas = np.zeros((dose_array.shape[1], ))
    depths = np.zeros((dose_array.shape[1], ))
    idd = np.zeros((dose_array.shape[1], ))

    for i in range(dose_array.shape[1]):

        slice = np.squeeze(dose_array[:, i, :])
        slice_max = np.max(slice)
        depths[i] = i * spacing[1]

        idd[i] = np.sum(slice) * spacing[0] * spacing[2]

        if slice_max > 0.01:

            slice = slice / slice_max

            # fit 2D gaussian to slice
            x = np.arange(0, slice.shape[1]) - slice.shape[1] / 2
            y = np.arange(0, slice.shape[0]) - slice.shape[0] / 2
            x, y = np.meshgrid(x, y)
            x = x.flatten()
            y = y.flatten()
            data = slice.flatten()
            popt, pcov = curve_fit(gaussian_2d, (x, y), data, p0=[1, 0, 0, 1, 1])

            # convert sigmas to mm
            sigma_x = popt[3] * spacing[0]
            sigma_y = popt[4] * spacing[2]

            sigmas[i] = np.mean([sigma_x, sigma_y])
    

    return sigmas, idd, depths


def plot_depth_sigma(all_sigmas, all_depths, r80s, output_path):

    for sigmas, depths, r80 in zip(all_sigmas, all_depths, r80s):
        depths = depths[sigmas > 0.1]
        sigmas = sigmas[sigmas > 0.1]
        surface_sigma = np.interp(r80 * 0.7, depths, sigmas)
        sigmas = sigmas - surface_sigma
        plt.plot(depths, sigmas)

    plt.xlabel("Depth (mm)")
    plt.ylabel("Sigma (mm)")
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def fit_model(sigmas, depths, distances, r80s):

    surface_sigmas = []
    ssds = []
    for sigma, depth, distance, r80 in zip(sigmas, depths, distances, r80s):
        surface_sigma = np.interp(r80 * 0.7, depth, sigma)
        ssd = np.interp(r80 * 0.7, depth, distance)
        surface_sigmas.append(surface_sigma)
        ssds.append(ssd)


    popt, pcov = curve_fit(lambda x, a, b, c: a * x ** 2 + b * x + c, ssds, surface_sigmas)
    print("Fit parameters: a = %.3E, b = %.3E, c = %.3E" % (popt[0], popt[1], popt[2]))

    x_fit = np.linspace(np.min(ssds)-50, np.max(ssds)+50, 100)
    y_fit = popt[0] * x_fit ** 2 + popt[1] * x_fit + popt[2]

    # plt.plot(ssds, surface_sigmas, 'o')
    # plt.plot(x_fit, y_fit)
    # plt.savefig(output_path)
    # plt.close()

    return popt
    

def plot_depth_idd(all_idds, all_depths, output_path):

    for idds,depths in zip(all_idds, all_depths):
        depths = depths[idds > 0.1]
        idds = idds[idds > 0.1]
        plt.plot(depths, idds)

    plt.xlabel("Depth (mm)")
    plt.ylabel("IDD (cGy*mm^2)")
    plt.savefig(output_path)
    plt.close()


def compute_R80s(all_idds, all_depths):

    r80s = []
    for idds, depths in zip(all_idds, all_depths):
        idds = np.flip(idds)
        depths = np.flip(depths)
        maxi = np.argmax(idds)
        idds = idds[0:maxi]
        depths = depths[0:maxi]
        r80 = np.interp(0.8 * np.max(idds), idds, depths)
        r80s.append(r80)

    return r80s


def print_lookup_tables(energy_index, fitting_params, beam_energies, r80s, all_sigmas, all_idds, all_depths, lookuptable_directory):

    output_path = os.path.join(lookuptable_directory, "energy_%03i.csv" % energy_index)

    for sigmas, depths, r80 in zip(all_sigmas, all_depths, r80s):
        depths = depths[sigmas > 0.1]
        sigmas = sigmas[sigmas > 0.1]
        surface_sigma = np.interp(r80 * 0.7, depths, sigmas)
        # sigmas = sigmas - surface_sigma

    depths_fit = np.linspace(0, 400.0, 401)
    idds_fit = []
    sigmas_fit = []
    for depth, idd, sigma, r80 in zip(all_depths, all_idds, all_sigmas, r80s):
        reference_sigma = np.interp(r80 * 0.7, depth, sigma)
        sigma = sigma - reference_sigma
        sigma_fit = np.interp(depths_fit, depth, sigma)
        sigmas_fit.append(sigma_fit)
        idd_fit = np.interp(depths_fit, depth, idd)
        idds_fit.append(idd_fit)

    idds_final = np.mean(idds_fit, axis=0)
    idd_max = np.max(idds_final)
    sigmas_final = np.mean(sigmas_fit, axis=0)
    sigmas_final[idds_final < (0.1 * idd_max)] = 3.0

    with open(output_path, 'w') as f:
        f.write("energy, R80,A,B,C\n")
        f.write("%s,%.2f,%.3E,%.3E,%.3E\n" % (beam_energies[0], np.mean(r80s), fitting_params[0], fitting_params[1], fitting_params[2]))
        f.write("\ndepth,sigma,idd\n")
        for d, i, s in zip(depths_fit, idds_final, sigmas_final):
            f.write("%.2f,%.2f,%.2f\n" % (d, s, i))


def import_dicom_data(dcm_directory):
    
    dcm_files = [f for f in os.listdir(dcm_directory) if f.endswith('.dcm')]
    plan_files = [f for f in dcm_files if 'RTPLAN' in pyd.dcmread(os.path.join(dcm_directory, f), force=True).Modality]
    dose_files = [f for f in dcm_files if 'RTDOSE' in pyd.dcmread(os.path.join(dcm_directory, f), force=True).Modality]

    if len(plan_files) != 1:
        print("Error: there should be exactly one RTPLAN file in the directory")
        sys.exit(1)

    plan = pyd.dcmread(os.path.join(dcm_directory, plan_files[0]), force=True)
    nbeams = len(plan.IonBeamSequence)

    if nbeams == 0:
        print("Error: there are no proton beams in the plan file")
        sys.exit(1)

    beam_numbers = []
    beam_isos = []
    beam_mus = []
    beam_energies = []
    print("Loading RT plan file: %s" % plan_files[0])

    for i in range(nbeams):

        beam = plan.IonBeamSequence[i]
        beam_name = beam.BeamName
        beam_type = beam.RadiationType

        if beam_type != 'PROTON':
            print("Error: beam %s is not a proton beam" % beam_name)
            continue

        beam_numbers.append(beam.BeamNumber)
        beam_isos.append(beam.IonControlPointSequence[0].IsocenterPosition)
        beam_mus.append(beam.IonControlPointSequence[0].ScanSpotMetersetWeights)
        beam_energies.append(beam.IonControlPointSequence[0].NominalBeamEnergy)

    beam_equivalent_depths = np.zeros((nbeams, ))
    beam_ssds = np.zeros((nbeams, ))
    for i in range(nbeams):

        referenced_beam_number = plan.FractionGroupSequence[0].ReferencedBeamSequence[i].ReferencedBeamNumber

        for j, beam_number in enumerate(beam_numbers):
            if beam_number == referenced_beam_number:
                beam_equivalent_depths[j] = plan.FractionGroupSequence[0].ReferencedBeamSequence[i].BeamDosePointEquivalentDepth
                beam_ssds[j] = plan.FractionGroupSequence[0].ReferencedBeamSequence[i].BeamDosePointSSD + (plan.FractionGroupSequence[0].ReferencedBeamSequence[i].BeamDosePointDepth -  plan.FractionGroupSequence[0].ReferencedBeamSequence[i].BeamDosePointEquivalentDepth)

    print("Loaded %i proton beams" % len(beam_numbers))

    dose_images = []
    dose_beam_numbers = []
    dose_sigmas = []
    dose_depths = []
    dose_idds = []
    dose_origins = []
    dose_sads = []
    dose_distances = []
    for dose_file in dose_files:
        
        print("Importing dose file: %s" % dose_file)
        dose = pyd.dcmread(os.path.join(dcm_directory, dose_file), force=True)

        if dose.DoseType != 'PHYSICAL':
            print("Error: dicom file %s is not a physical dose" % dose_file)
            continue

        if not hasattr(dose, 'InstanceNumber'):
            print("Error: dicom file %s does not have InstanceNumber tag" % dose_file)
            continue

        dose_beam_number = dose.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber
        if dose_beam_number not in beam_numbers:
            print("Error: beam number %i is not in the list of proton beams" % dose_beam_number)
            continue
        else:
            print("Beam number: %i" % dose_beam_number)

        dose_grid = dose.pixel_array * dose.DoseGridScaling
        dose_sitk = sitk.GetImageFromArray(dose_grid)
        dose_sitk.SetSpacing([float(dose.PixelSpacing[0]), float(dose.PixelSpacing[1]), float(dose.SliceThickness)])
        dose_sitk.SetOrigin(dose.ImagePositionPatient)

        dose_images.append(dose_sitk)
        dose_beam_numbers.append(dose_beam_number)

        origin_to_iso = beam_isos[beam_numbers.index(dose_beam_number)][1] - dose.ImagePositionPatient[1]
        edge_to_surface = origin_to_iso - beam_equivalent_depths[beam_numbers.index(dose_beam_number)]

        sigma, idd, depth = fit_sigmas(dose_grid, dose_sitk.GetSpacing())

        #normalize IDD by beam mu
        idd = idd / beam_mus[beam_numbers.index(dose_beam_number)]

        depth = depth - edge_to_surface
        distance = depth + beam_ssds[beam_numbers.index(dose_beam_number)]
        dose_sigmas.append(sigma)
        dose_depths.append(depth)
        dose_idds.append(idd)
        dose_origins.append(dose_sitk.GetOrigin())
        dose_distances.append(distance)

    return beam_energies, dose_idds, dose_depths, dose_sigmas, dose_distances


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python import_dose_files.py <lookuptable_directory>")
        sys.exit(1)

    df = pd.read_csv(os.path.join(sys.argv[1], "energies.csv"))

    for i in range(len(df)):

        dcm_directory = df.iloc[i]["dcm_directory"]
        target_energy = df.iloc[i]["energy_label"]
        energy_index = int(df.iloc[i]["index"])
        print("Processing energy %s..." % target_energy)

        beam_energies, dose_idds, dose_depths, dose_sigmas, dose_distances = import_dicom_data(dcm_directory)

        if target_energy == beam_energies[0]:
            print("Energy in dicoms matches energy label.")
        else:
            print("Warning! Energy in dicoms does not match energy label. Double check, or there might be errors in dose calc.")

        r80s = compute_R80s(dose_idds, dose_depths)
        fitting_params = fit_model(dose_sigmas, dose_depths, dose_distances, r80s, os.path.join(dcm_directory))
        print_lookup_tables(energy_index, fitting_params, beam_energies, r80s, dose_sigmas, dose_idds, dose_depths, sys.argv[1])

    print("Done")
