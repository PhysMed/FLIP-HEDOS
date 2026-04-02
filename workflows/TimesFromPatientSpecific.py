# Under the GNU General Public License v3.0 (GPLv3):
# Copyright (C) 2026 PhysMed Research Group - University of Navarra
#
# ========= Created by Marina Garcia-Cardosa in January 2024 ============
# ========= Modify by Marina Garcia-Cardosa in July 2024 to manage with proton and photon patients =========
import numpy as np


def total_field_on_time(irrad_times, number_of_fields, modality):
    t_total_final = 0
    for field in range(number_of_fields):
        if field == 0:
            if modality == 'proton':
                times = irrad_times['Beam1']
            else:
                times = irrad_times['Arc1']
            for t in range(np.shape(times[0][0])[0]):
                t_total = times[0][0][t][1] - times[0][0][t][0]
                t_total_final = t_total_final + t_total
            del times
        elif field == 1:
            if modality == 'proton':
                times = irrad_times['Beam2']
            else:
                times = irrad_times['Arc2']
            for t in range(np.shape(times[0][0])[0]):
                t_total = times[0][0][t][1] - times[0][0][t][0]
                t_total_final = t_total_final + t_total
            del times
        elif field == 2:
            if modality == 'proton':
                times = irrad_times['Beam3']
            else:
                print(f'I am not prepared to manage more than three RT arcs.')
            for t in range(np.shape(times[0][0])[0]):
                t_total = times[0][0][t][1] - times[0][0][t][0]
                t_total_final = t_total_final + t_total
            del times
        elif field == 3:
            if modality == 'proton':
                times = irrad_times['Beam4']
            else:
                print(f'I am not prepared to manage more than four RT arcs.')
            for t in range(np.shape(times[0][0])[0]):
                t_total = times[0][0][t][1] - times[0][0][t][0]
                t_total_final = t_total_final + t_total
            del times
        else:
            print(f'I am not prepared to manage more than four fields.')

    return t_total_final


def start_field_times(irrad_times, number_of_fields, time_between_fields, modality):
    start_times = []
    for field in range(number_of_fields):
        if field == 0:
            if modality == 'proton':
                times = irrad_times['Beam1']
            else:
                times = irrad_times['Arc1']
            for t in range(np.shape(times[0][0])[0]):
                start_times.append(times[0][0][t][0])
            times = []
        elif field == 1:
            if modality == 'proton':
                times = irrad_times['Beam2']
            else:
                times = irrad_times['Arc2']
            jump_time = start_times[-1] + time_between_fields
            for t in range(np.shape(times[0][0])[0]):
                start_times.append(jump_time + times[0][0][t][0])
            times = []
        elif field == 2:
            if modality == 'proton':
                times = irrad_times['Beam3']
            else:
                print(f'I am not prepared to manage more than three RT arcs.')
            jump_time = start_times[-1] + time_between_fields
            for t in range(np.shape(times[0][0])[0]):
                start_times.append(jump_time + times[0][0][t][0])
            times = []
        elif field == 3:
            if modality == 'proton':
                times = irrad_times['Beam4']
            else:
                print(f'I am not prepared to manage more than four RT arcs.')
            jump_time = start_times[-1] + time_between_fields
            for t in range(np.shape(times[0][0])[0]):
                start_times.append(jump_time + times[0][0][t][0])
            times = []
        else:
            print(f'I am not prepared to manage more than four fields.')

    return start_times


def field_on_times(irrad_times, number_of_fields, modality):
    on_times = []
    for field in range(number_of_fields):
        if field == 0:
            if modality == 'proton':
                times = irrad_times['Beam1']
            else:
                times = irrad_times['Arc1']
        elif field == 1:
            if modality == 'proton':
                times = irrad_times['Beam2']
            else:
                times = irrad_times['Arc2']
        elif field == 2:
            if modality == 'proton':
                times = irrad_times['Beam3']
            else:
                print(f'I am not prepared to manage more than three RT arcs.')
        elif field == 3:
            if modality == 'proton':
                times = irrad_times['Beam4']
            else:
                print(f'I am not prepared to manage more than four RT arcs.')
        else:
            print(f'I am not prepared to manage more than four fields.')

        for t in range(np.shape(times[0][0])[0]):
                on_times.append(times[0][0][t][1] - times[0][0][t][0])
        del times

    return on_times


def calculate_nr_steps(irrad_times, number_of_fields, time_between_fields, dt, modality):
    nr_steps = []
    for field in range(number_of_fields):
        if field == 0:
            if modality == 'proton':
                times = irrad_times['Beam1']
            else:
                times = irrad_times['Arc1']
            nr_steps = np.round(np.round(times[0][0][-1][1], 2) / dt) + 1
            del times
        elif field == 1:
            if modality == 'proton':
                times = irrad_times['Beam2']
            else:
                times = irrad_times['Arc2']
            nr_steps = nr_steps + (np.round(np.round(np.round(times[0][0][-1][1], 2) + time_between_fields, 2) / dt) + 1)
            del times
        elif field == 2:
            if modality == 'proton':
                times = irrad_times['Beam3']
            else:
                print(f'I am not prepared to manage more than three RT arcs.')
            nr_steps = nr_steps + (np.round(np.round(np.round(times[0][0][-1][1], 2) + time_between_fields, 2) / dt) + 1)
            del times
        elif field == 3:
            if modality == 'proton':
                times = irrad_times['Beam4']
            else:
                print(f'I am not prepared to manage more than four RT arcs.')
            nr_steps = nr_steps + (np.round(np.round(np.round(times[0][0][-1][1], 2) + time_between_fields, 2) / dt) + 1)
            del times
        else:
            print(f'I am not prepared to manage more than four fields.')

    return nr_steps
