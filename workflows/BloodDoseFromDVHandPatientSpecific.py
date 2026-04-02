# Under the GNU General Public License v3.0 (GPLv3):
# Copyright (C) 2026 PhysMed Research Group - University of Navarra
#
# This file includes code licensed under the MIT License:
# Copyright (c) 2021-2023 MGH Radiation Oncology
#
# ========= Created by Marina Garcia-Cardosa in January 2024 ============
# ========= Modified by Marina Garcia-Cardosa:
#           * July 2024 to manage with proton and photon patients =========
#           * September-October 2024 to manage not only thorax-abdomen, but also head_neck patients patients =========
# ========= Last update done in October 2024 by Marina Garcia-Cardosa to enhance and optimize it =========

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.io
import datetime
import pickle
import gzip


from simulation import TemporalDistribution, DoseRateFromDVH, CompartmentDose, FlowModel
from PlotDoseDistribution import *


def blood_dose_distribution(simulation_params, patient_params, treatment_params, patient_directory):
    # ======= Step 1. Initialize simulation ============================= #
    if patient_params['tumor_site'] == 'head_neck':
        filename = 'input/phantom/ICRP89_compartment_model_with_head_neck.xlsx'
    else:
        filename = 'input/phantom/ICRP89_compartment_model.xlsx'

    model = FlowModel.ExpandFlowModelPatient(filename, patient_params, simulation_params, patient_directory, treatment_params)

    model.patient.load_patient_specific_data()

    # add the arterial compartment of the FLIP method
    if 'flip_arterial' in patient_params['organs_specific'] and patient_params['tumor_site'] == 'head_neck':
        model.replace_box_with_flip('flip_arterial', box_dict=patient_params)
    else:
        model.add_box_parallel_and_series_flip('flip_arterial', box_dict=patient_params)

    # add the venous compartment of the FLIP method
    if 'flip_venous' in patient_params['organs_specific'] and patient_params['tumor_site'] == 'head_neck':
        model.replace_box_with_flip('flip_venous', box_dict=patient_params)
    else:
        model.split_box_series_flip('flip_venous', box_dict=patient_params)
    # ============================================================== #

    # ======== Step 2. Generate distribution ======================= #
    blood = TemporalDistribution(model)
    if simulation_params['generate_new']:
        model.construct_weibull()
        blood.generate_from_weibull_fliphedos()

        # Could also do a Markov process, i.e. corresponding to exponential transit time distribution
        # This is the same as the above with Weibull shape_parameter=1.
    else:
        # load a blood_path.npy (in case you have it)
        blood.load('blood_path.npy')

    # blood.plot_time_distributions()
    # blood.plot_inflow_outflow()
    # blood.plot_volumes_over_time()
    # blood.plot_normalized_volume_over_time()
    blood.plot_final_blood_volumes()
    # blood.plot_volumes_over_time_flip()

    # ======== Step 3. Compute dose metrics for one fraction in patient-specific (FLIP) compartments ======================= #
    # It is kept the day in which the simulation was performed:
    day = datetime.datetime.now().strftime("%Y-%B-%d")
    print(f"The day the simulation was performed: {day}.\n")

    num_patient= patient_directory['folder']
    [doses_flip, bps_with_dose_flip] = blood.dose_metrics_flip()
    [unique_visits, percentages]= blood.analyze_bps_visits_in_flip()

    # Saving the dose metrics values:
    where_save_BPdoses_flip ='output'+ num_patient +'/bps_with_dose_'+ day + '_' + str(treatment_params['time_between_fields'])+'.npy'
    np.save(where_save_BPdoses_flip, bps_with_dose_flip)

    where_save_doses_flip ='output'+ num_patient +'/doses_'+ day + '_' + str(treatment_params['time_between_fields'])+'.npy'
    np.save(where_save_doses_flip, doses_flip)

    # Saving the necessary values to plot the visit histogram in another moment:
    where_save_unique_visits ='output'+ num_patient +'/unique_visits_'+ day + '_' + str(treatment_params['time_between_fields'])+'.npy'
    np.save(where_save_unique_visits, unique_visits)
    where_save_percentages='output'+ num_patient +'/percentages_'+ day + '_' + str(treatment_params['time_between_fields'])+'.npy'
    np.save(where_save_percentages, percentages)

    # ======== Step 4. Accumulate dose ============================= #
    dose = DoseRateFromDVH(n_fractions=treatment_params['nr_fractions'],
                           total_field_on_time=treatment_params['total_field_on_time'])

    compartment_ids = [[i for i, name in enumerate(blood.model.names) if organ in name] for organ in patient_params['organs_DVH']]

    print('Organ DVHs to compute dose:')
    dose_contributions = {}
    for organ, compartment_id in zip(patient_params['organs_DVH'], compartment_ids):
        blood_dose = CompartmentDose(blood.path, blood.model.dt)
        if ('specific_vasculature' in patient_params['organs']) and ('flip' in organ):
            blood_dose.dose_from_patient_specific_compartments(blood.model)
            dose_contributions['specific_vasculature'] = blood_dose.dose
        else:
            dose_rate_hist = dose.get_dose_rate_hist('input/patients'+ num_patient +'/DVHs/' + organ + '_DVH.csv')
            for start_time, field_on_time in zip(treatment_params['start_times'], treatment_params['field_on_times']):
                blood_dose.add_dose(dose_rate_hist, compartment_id, start_time=start_time, field_on_time=field_on_time)
            dose_contributions[organ] = blood_dose.dose
            print(" • Dose to {} computed".format(organ))

    blood_dose_total = CompartmentDose(blood.path, blood.model.dt)
    blood_dose_total.dose = sum(list(dose_contributions.values()))

    # To plot Blood DVH per organ in the same figure:
    plt.figure()
    for organ in patient_params['organs']:
        calculate_dvh_patient_specific(dose_contributions, blood.model.patient.NumParticles, organ)
    # Save the figure
    plt.savefig('output' + num_patient + '/BloodDVH_' + day + '.pdf')

    # It is shown the day and the time in the console message:
    hour_min = datetime.datetime.now().strftime("%H:%M")
    print(f"\nBlood DVH figure from Patient {num_patient} is saved today {day} at {hour_min}h.")

    # Dose metrics for one fraction taking into account all compartments
    # Access the dose array
    doses = blood_dose_total.dose
    # Filter values greater than 0.001 Gy
    dose_filtered = doses[doses > 0.001]
    # Calculate the average of the filtered values
    mean_doses = np.mean(dose_filtered)
    bps_dose = np.array(blood_dose_total.dose)
    total = len(bps_dose)
    percentage_higher_0001 = np.sum(bps_dose > 0.001) / total * 100
    percentage_higher_01 = np.sum(bps_dose > 0.1) / total * 100
    print(f"\nFLIP-HEDOS compartments:\n • BPs mean dose > 0.001 Gy: {mean_doses:.4f} Gy")
    print(f" • BPs percentage that have received > 0.001 Gy: {percentage_higher_0001:.2f} %")
    print(f" • BPs percentage that have received > 0.1 Gy: {percentage_higher_01:.2f} %\n")

    # Accumulate the dose along all treatment fractions...
    if simulation_params['accumulate']:
        blood_dose_total.repeat(treatment_params['nr_fractions'])
        print('Accumulation along all treatment fractions done!')
        # To plot histograms (all fractions all organs vs one fraction per organ):
        plot_dose_distribution(blood_dose_total, dose_contributions)
        # Save the figure
        plt.savefig('output'+ num_patient +'/BloodContributions_' + day + '.pdf')
        print(f"Blood contributions figure from Patient {num_patient} is saved today {day} at {hour_min}h.")
    # ============================================================== #

    # ======== Step 5. To save the blood matrix? ============================= #
    # In case you use the python console, is better to remove variables (to not overload python and your computer) before try to save <blood>.
    # delete_variables = input("Do you want to remove all variables except <blood> to proceed with saving the blood matrix?: (True o False): ").strip().lower()
    # if delete_variables in ["true", "yes", "1"]:  # Valores considerados como True
    #     del Chain, BloodDoseFromFields, Weibull, TimesFromPatientSpecific, MarkovChain, CompartmentDose, choose_color_plot, DoseRateFromDVH
    #     del ImportPatientFromMATLAB, plot_dose_distribution, Patient, calculate_dvh_patient_specific, DoseRate, BloodDoseFromPatientSpecific
    #     del plot_volumes, BloodDoseFromDVH, ExpandFlowModelPatient, ExpandFlowModel, TemporalDistribution
    #     del model, bps_with_dose, doses, field_on_times, irrad_times, organs, organs_DVH, organs_specific, patient_directory, patient_parameters, patient_params
    #     del simulation_parameters, simulation_params, start_times, treatment_parameters, treatment_params, types_of_trajectories
    #     print("Special variables have been removed from the workspace.")
    #
    #     # Save the compressed matrix with gzip
    #     where_save_blood='output'+ num_patient +'/bloodPath_FLIPHEDOS_' + day + '.pkl.gz'
    #     with gzip.open(where_save_blood, 'wb') as f:
    #         pickle.dump(blood, f)

        # # If we want to load the blood matrix, we have to uncomment the following code lines:
        # what_day = input("What is the day of the filename to load?: ")
        # where_save_blood = 'output' + num_patient + '/bloodPath_FLIPHEDOS_' + what_day + '.pkl.gz'
        # with gzip.open(where_save_blood, 'rb') as f:
        #     blood = pickle.load(f)
