# Under the GNU General Public License v3.0 (GPLv3):
# Copyright (C) 2026 PhysMed Research Group - University of Navarra
#
# This file includes code licensed under the MIT License:
# Copyright (c) 2023 MGH Radiation Oncology
#
# ========= Created by Chris Beekman et al. 2023 ===========
# ========= Modified by Marina Garcia-Cardosa since January 2024 to manage with data patients =========
# ========= Last update done in Novemeber 2024 by Marina Garcia-Cardosa to enhance and optimize it =========
from workflows import ImportPatientFromMATLAB, TimesFromPatientSpecific, BloodDoseFromDVHandPatientSpecific

# ==================== Patient directory ====================== #
# Specify your directory...
directory = '../input/patients'
print("Selected directory:", directory)
num_patient = input('Enter the patient number you want to upload: ')
folder = '/Patient' + num_patient
patient_directory = {
    'directory': directory,
    'folder': folder
}
# ============================================================== #

# ==================== Patient parameters ====================== #
gender = 'M'

print('Gender: ', gender)
if gender.lower() in ['m', 'male']:
    sheet_name = 'male_new'
    TBV = 5.3  # 5.3 L total simulation volume.
    litros_CO = float(input('Enter the number of liters per minute you want the CO to have (L): '))
    CO = litros_CO / 60  # 6.5 L/min cardiac output.
elif gender.lower() in ['f', 'female']:
    sheet_name = 'female_new'
    TBV = 3.9  # 3.9 L total simulation volume.
    CO = 5.9 / 60  # 5.9 L/min cardiac output.
else:
    raise ValueError('Cannot deduce gender.')

tumor_site = input('Enter the tumor location (thorax-abdomen / head_neck): ').lower()  # for example 'liver' or 'head_neck'
tumor_volume_fraction = 0.05
relative_blood_density = 1.0
relative_perfusion = 1.0
flip_site_arterial = input('Enter the arterial FLIP location (aorta / head_neck_arteries): ').lower() # 'aorta' # 'head_neck_arteries'
flip_site_venous = input('Enter the venous FLIP location (inferior_vena_cava / head_neck_veins): ').lower() # 'inferior_vana_cava' # 'head_neck_veins'
volume_arterial_compartment = float(input('Enter the volume of the arterial FLIP compartment (L): '))  # in liters (L)
flip_volume_arterial = (volume_arterial_compartment * 100) / TBV
volume_venous_compartment = float(input('Enter the volume of the venous FLIP compartment (L): '))  # in liters (L)
flip_volume_venous = (volume_venous_compartment * 100) / TBV
# Depending on the chosen patient organs and organs_DVH is different:
organs_specific = ['flip_arterial', 'flip_venous']
# Example of one patient (thorax-abdomen):
if num_patient == '19':
    organs = ['specific_vasculature', 'left_heart', 'right_heart', 'aorta', 'inferior_vena_cava', 'liver', 'kidney', 'stomach_oesophagus', 'pancreas']
    organs_DVH = ['flip_arterial', 'flip_venous', 'left_heart', 'right_heart', 'aorta', 'inferior_vena_cava', 'liver', 'kidney', 'stomach_oesophagus', 'pancreas']
# Example of another patient (head_neck):
elif num_patient == '20':
    organs = ['specific_vasculature', 'bone_muscle_skin', 'brain']
    organs_DVH = ['flip_arterial', 'flip_venous', 'bone_muscle_skin', 'brain']
else:
    print('This patient is not yet available for simulation :(')

patient_parameters = {
    'gender': gender,
    'sheet_name': sheet_name,
    'tumor_site': tumor_site,
    'tumor_volume_fraction': tumor_volume_fraction,
    'flip_site_arterial': flip_site_arterial,
    'flip_volume_arterial': flip_volume_arterial,
    'flip_site_venous': flip_site_venous,
    'flip_volume_venous': flip_volume_venous,
    'relative_blood_density': relative_blood_density,
    'relative_perfusion': relative_perfusion,
    'organs': organs,
    'organs_DVH': organs_DVH,
    'organs_specific': organs_specific,
    'TBV': TBV,
    'CO': CO
}
# ============================================================== #

# ==================== Treatment parameters ==================== #
# =========== this is an example in case that a patient has four beams:
# total_beam_on_time = 80
# start_times = [10, 40, 70, 100]
# beam_on_times = [20, 20, 20, 20]

# =========== this is an example in case that a patient has one beam:
# total_beam_on_time = 10
# start_times = [10]
# beam_on_times = [10]
# assert(sum(beam_on_times) == total_beam_on_time), 'Beam-on-time of separate fields should equal total beam-on-time.'
# assert(start_times[:-1] + beam_on_times[:-1] <= start_times[1:]), 'Cannot start new field before completing current.'

# =========== another example in case that a patient has one or several beams with patient-specific data:
modality = input('Select the treatment modality (proton or photon): ').lower() #'proton' # it can be proton or photon
nr_fractions = int(input('Enter the number of treatment fractions (integer): '))
first_treatment_field = 1
number_of_fields = int(input('Enter the number of fields/incidents in the treatment (integer): '))
time_between_fields = int(input('Enter the inter-beam/inter-arc time (integer): '))
name_all_trajectories = 'TrayectosVasos'
name_arterial_trajectories = 'TrayectosVasosArterial'
name_venous_trajectories = 'TrayectosVasosVenous'
types_of_trajectories = ['arterial', 'venous']
# importing dictionary of irradiation times
irrad_times = ImportPatientFromMATLAB.import_patient_times(patient_directory=directory, patient_folder=folder)
total_field_on_time = TimesFromPatientSpecific.total_field_on_time(irrad_times, number_of_fields, modality)
start_times = TimesFromPatientSpecific.start_field_times(irrad_times, number_of_fields, time_between_fields, modality)  # start times per energy layer for all the beams
field_on_times = TimesFromPatientSpecific.field_on_times(irrad_times, number_of_fields, modality)  # beam on times per energy layer for all the beams
assert(sum(field_on_times) == total_field_on_time), 'field-on-time of all the fields should be equal to total_field-on-time.'
assert(start_times[:-1] + field_on_times[:-1] <= start_times[1:]), 'Cannot start new field before completing current.'

treatment_parameters = {
    'modality': modality,
    'nr_fractions': nr_fractions,
    'first_treatment_field': first_treatment_field,
    'number_of_fields': number_of_fields,
    'name_all_trajectories': name_all_trajectories,
    'name_arterial_trajectories': name_arterial_trajectories,
    'name_venous_trajectories': name_venous_trajectories,
    'types_of_trajectories': types_of_trajectories,
    'total_field_on_time': total_field_on_time,
    'start_times': start_times,
    'field_on_times': field_on_times,
    'time_between_fields': time_between_fields
}

# =================== Simulation parameters ==================== #
voxel_size_mm3 = float(input('Enter the voxel size (mm³): '))
voxel_size_l = voxel_size_mm3 / 1000000  # in L
sample_size = round(patient_parameters['TBV'] / voxel_size_l)    # nr simulation particles
# sample_size = 10000
dt = 0.01  # in seconds
nr_steps = int(TimesFromPatientSpecific.calculate_nr_steps(irrad_times, number_of_fields, time_between_fields, dt, modality))  # nr time steps for all the beams (whole treatment)
weibull_shape = 2
generate_new = True
random_walk = False
accumulate = True
factor_arterial = int(input('Enter the arterial factor value (integer): '))
factor_venous = int(input('Enter the venous factor value (integer): '))

simulation_parameters = {
    'sample_size': sample_size,
    'nr_steps': nr_steps,
    'dt': dt,
    'weibull_shape': weibull_shape,
    'generate_new': generate_new,
    'random_walk': random_walk,
    'accumulate': accumulate,
    'voxel_size': voxel_size_l,
    'factor_arterial': factor_arterial,
    'factor_venous': factor_venous,
}

# ============================================================== #
BloodDoseFromDVHandPatientSpecific.blood_dose_distribution(simulation_parameters, patient_parameters, treatment_parameters, patient_directory)
