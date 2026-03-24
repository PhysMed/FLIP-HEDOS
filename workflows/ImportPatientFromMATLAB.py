# ========= Created by Marina Garcia-Cardosa in January 2024 ============
import numpy as np
from scipy.io import loadmat
import os
import glob


def import_patient_grid(patient_directory, patient_folder):
    pathXP = patient_directory + patient_folder + '/XP.mat'
    dictXP = loadmat(pathXP)
    pathYP = patient_directory + patient_folder + '/YP.mat'
    dictYP = loadmat(pathYP)
    pathZP = patient_directory + patient_folder + '/ZP.mat'
    dictZP = loadmat(pathZP)
    # print('Velocity coordinates loaded')
    return dictXP, dictYP, dictZP


def import_patient_coordinates(patient_directory, patient_folder):
    pathCoord = patient_directory + patient_folder + '/Coordenadas.mat'
    dictCoord = loadmat(pathCoord)
    # print('Dose coordinates loaded')
    return dictCoord


def import_patient_roi_arterial(patient_directory, patient_folder):
    pathROIarterial = patient_directory + patient_folder + '/ROIarterial.mat'
    dictROIarterial = loadmat(pathROIarterial)
    # print('ROIs loaded')
    return dictROIarterial


def import_patient_roi_venous(patient_directory, patient_folder):
    pathROIvenous = patient_directory + patient_folder + '/ROIvenous.mat'
    dictROIvenous = loadmat(pathROIvenous)
    # print('ROIs loaded')
    return dictROIvenous


def import_patient_trajectories(patient_directory, patient_folder):
    pathTraj = patient_directory + patient_folder + '/TrayectosVasos.mat'
    dictTraj = loadmat(pathTraj)
    # print('Trajectories loaded')
    return dictTraj


def import_patient_arterial_trajectories(patient_directory, patient_folder):
    pathArterialTraj = patient_directory + patient_folder + '/TrayectosVasosArterial.mat'
    dictArterialTraj = loadmat(pathArterialTraj)
    # print('Trajectories loaded')
    return dictArterialTraj


def import_patient_venous_trajectories(patient_directory, patient_folder):
    pathVenousTraj = patient_directory + patient_folder + '/TrayectosVasosVenous.mat'
    dictVenousTraj = loadmat(pathVenousTraj)
    # print('Trajectories loaded')
    return dictVenousTraj


def import_patient_energies(patient_directory, patient_folder):
    path_for_energies = patient_directory + patient_folder
    pattern = '*Energies*'
    energies_files = []
    for file in glob.glob(os.path.join(path_for_energies, pattern)):
        file_name = os.path.basename(file).lower()
        if 'energies' in file_name:
            energies_files.append(file)
    if len(energies_files) == 1:
        pathEnergies = patient_directory + patient_folder + '/Energies.mat'
        dictEnergies = loadmat(pathEnergies)
    else:
        dictEnergies = []
        for each_pack_energies in range(len(energies_files)):
            pathEnergies_beam = energies_files[each_pack_energies]
            if each_pack_energies == 0:
                dictEnergies = loadmat(pathEnergies_beam)
            else:
                dictEnergies_next_beam = loadmat(pathEnergies_beam)
                concatenate_dicts = np.concatenate((dictEnergies['Energies'], dictEnergies_next_beam['Energies']), axis=1)
                dictEnergies['Energies'] = concatenate_dicts
    return dictEnergies


def import_patient_times(patient_directory, patient_folder):
    pathTimes = patient_directory + patient_folder + '/Times.mat'
    dictTimes = loadmat(pathTimes)
    # print('Times loaded')
    return dictTimes['Times']


def import_patient_dose_rate_norm(patient_directory, patient_folder):
    pathDoseRateNorm = patient_directory + patient_folder + '/DoseRateNorm.mat'
    dictDoseRateNorm = loadmat(pathDoseRateNorm)
    # print('Normalized dose rate loaded')
    return dictDoseRateNorm
