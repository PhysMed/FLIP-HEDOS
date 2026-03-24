# ========= Created by Marina Garcia-Cardosa in January 2024 ============
# ========= Modified by Marina Garcia-Cardosa in July 2024 to manage with proton and photon patients =========
# ========= Last update done in October 2024 by Marina Garcia-Cardosa to enhance and optimize it =========
from workflows.ImportPatientFromMATLAB import *
from FLIPmethod.ClassesForFLIPmethod import *
import numpy as np
import math
import copy
import pickle


class Patient:
    def __init__(self, patient_directory, treatment_parameters, simulation_parameters):
        # Initialization of simple attributes
        self.change_field = False
        self.organ_dose = []
        self.mean_organ_dose_arterial = []
        self.mean_organ_dose_venous = []
        self.modality = treatment_parameters['modality']
        self.time_between_fields = treatment_parameters['time_between_fields']
        self.first_treatment_field = treatment_parameters['first_treatment_field']
        self.number_of_fields = treatment_parameters['number_of_fields']
        self.field = None
        self.directory = patient_directory['directory']
        self.folder = patient_directory['folder']
        self.types_of_trajectories = treatment_parameters['types_of_trajectories']
        self.name_arterial_trajectories = treatment_parameters['name_arterial_trajectories']
        self.name_venous_trajectories = treatment_parameters['name_venous_trajectories']
        self.nr_steps = simulation_parameters['nr_steps']
        self.dt = simulation_parameters['dt']
        self.factor_arterial = simulation_parameters['factor_arterial']
        self.factor_venous = simulation_parameters['factor_venous']

        # Initialize attributes that will hold None or empty lists
        self.initialize_empty_attributes()

        # Load patient data (these could be moved to separate methods for clarity)
        self.dictXP, self.dictYP, self.dictZP = self.import_patient_grid()
        self.dictCoord = self.import_patient_coordinates()
        self.dictEnergies = self.import_patient_energies()
        self.dictTimes = self.import_patient_times()
        self.dictDoseRateNorm = self.import_patient_dose_rate_norm()

        # Calculate voxel-related attributes
        self.calculate_voxel_attributes()

        # Load arterial and venous trajectories, ROIs
        self.dictTraj_arterial = self.import_patient_arterial_trajectories()
        self.dictTraj_venous = self.import_patient_venous_trajectories()
        self.dictROI_arterial = self.import_patient_roi_arterial()
        self.dictROI_venous = self.import_patient_roi_venous()
        self.NumTrajectories_arterial = np.size(self.dictTraj_arterial['TrayectosVasosArterial'])
        self.NumTrajectories_venous = np.size(self.dictTraj_venous['TrayectosVasosVenous'])
        self.NumParticles = simulation_parameters['sample_size']

    def initialize_empty_attributes(self):
        """Initialize attributes that start with None or empty lists"""
        self.Particles = None
        self.TrajectoriesInDose = None
        self.doseInfo = None
        self.InstantDose = None
        self.PosI = self.PosJ = self.PosK = self.Which = None
        self.TrajectoriesInDose_arterial = None
        self.PosI_arterial = self.PosJ_arterial = self.PosK_arterial = self.Which_arterial = None
        self.TrajectoriesInDose_venous = None
        self.PosI_venous = self.PosJ_venous = self.PosK_venous = self.Which_venous = None
        self.ListActiveParticles = []
        self.ListActiveParticles_arterial = []
        self.ListActiveParticles_venous = []
        self.LastParticle = -1
        self.listIndex = []  # I clear it
        self.tIni = 0
        self.tFinal = None
        self.stepTime = None
        self.out_comp_flip = []
        self.listParticlesInTrajectories = []
        self.outflow_flip = []
        self.prob_enter_traj_arterial = []
        self.prob_enter_traj_venous = []
        self.step_range = []

    def calculate_voxel_attributes(self):
        """Calculate DX, DY, DZ, voxel size, and related attributes"""
        DX = (self.dictXP['XP'][:, 2:, :] - self.dictXP['XP'][:, :-2, :])
        DX = np.mean(DX) / 2
        DY = (self.dictYP['YP'][2:, :, :] - self.dictYP['YP'][:-2, :, :])
        DY = np.mean(DY) / 2
        DZ = (self.dictZP['ZP'][:, :, 2:] - self.dictZP['ZP'][:, :, :-2])
        DZ = np.mean(DZ) / 2
        self.DL = (np.abs(DX) + np.abs(DY) + np.abs(DZ)) / 3
        self.VolVoxel = np.abs(DX * DY * DZ)

    def import_patient_grid(self):
        return import_patient_grid(self.directory, self.folder)

    def import_patient_coordinates(self):
        return import_patient_coordinates(self.directory, self.folder)

    def import_patient_energies(self):
        return import_patient_energies(self.directory, self.folder)

    def import_patient_times(self):
        return import_patient_times(self.directory, self.folder)

    def import_patient_dose_rate_norm(self):
        return import_patient_dose_rate_norm(self.directory, self.folder)

    def import_patient_arterial_trajectories(self):
        return import_patient_arterial_trajectories(self.directory, self.folder)

    def import_patient_venous_trajectories(self):
        return import_patient_venous_trajectories(self.directory, self.folder)

    def import_patient_roi_arterial(self):
        return import_patient_roi_arterial(self.directory, self.folder)

    def import_patient_roi_venous(self):
        return import_patient_roi_venous(self.directory, self.folder)

    def cleanLastTimeFromTrajectories(self):
        for i in range(np.size(self.dictTraj[self.name_trajectories])):
            # Note: "ultimo" is "last" in english...
            self.dictTraj[self.name_trajectories]['ultimo_t'][0][i] = 0
        return self.dictTraj

    def load_change_field(self):
        self.first_treatment_field = self.first_treatment_field + 1
        self.change_field = False
        self.load_patient_specific_data()

    def load_patient_specific_data(self):
        # Updated October 2024
        CurrentField = self.first_treatment_field
        field_type = 'Beam' if self.modality == 'proton' else 'Arc'
        print(f'It is working {field_type}{CurrentField} in {self.modality} RT modality. \n')

        if CurrentField == 1:
            # Clear particles and initialize them
            self.Particles = [Particle() for _ in range(self.NumParticles)]

        # Recalculate trajectory positions in the dose matrix
        XD, YD, ZD = self.dictCoord['Coordenadas']['x'][0][0], self.dictCoord['Coordenadas']['y'][0][0], \
        self.dictCoord['Coordenadas']['z'][0][0]
        CoorX, CoorY, CoorZ = [np.min(XD), np.max(XD)], [np.min(YD), np.max(YD)], [np.min(ZD), np.max(ZD)]
        dX, dY, dZ = np.diff(CoorX) / (XD.shape[1] - 1), np.diff(CoorY) / (YD.shape[0] - 1), np.diff(CoorZ) / (
                    ZD.shape[2] - 1)
        dD = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)

        # Extract energy layers for the current field
        energy_key = 'Beam' if self.modality == 'proton' else 'Arc'
        self.listIndex = [i for i in range(np.size(self.dictEnergies['Energies'][energy_key]))
                          if self.dictEnergies['Energies'][energy_key][0][i][0][0] == CurrentField]

        # Clear doseInfo and initialize it
        self.doseInfo = [SeveralDoseInfo() for _ in self.listIndex]

        for i, idx in enumerate(self.listIndex):
            energy_data = self.dictEnergies['Energies']
            if self.modality == 'proton':
                self.doseInfo[i].Field = energy_data['Beam'][0][idx][0][0]
            else:
                self.doseInfo[i].Field = energy_data['Arc'][0][idx][0][0]

            self.doseInfo[i].AngleGantry = energy_data['AngleGantry'][0][idx][0][0]
            self.doseInfo[i].Dose = energy_data['Dose'][0][idx][:][:]

            if self.modality == 'proton':
                self.doseInfo[i].Segment = energy_data['Segment'][0][idx][0][0]
                self.doseInfo[i].NumSpots = energy_data['NumSpots'][0][idx][0][0]

        if CurrentField == 1:
            self.field = f'{field_type}{self.first_treatment_field}'
            self.tFinal = round(self.dictTimes[self.field][0][0][-1][1] * 100) / 100
            self.stepTime = [x * 0.01 for x in range(int((self.tFinal - self.tIni) / 0.01) + 1)]
            self.listParticlesInTrajectories = [None] * len(self.stepTime)
            self.step_range = [0, len(self.stepTime) - 1]
        else:
            self.step_range[0] = self.step_range[-1] + int(self.time_between_fields / self.dt) + 1
            self.field = f'{field_type}{self.first_treatment_field}'
            self.tFinal = round(self.dictTimes[self.field][0][0][-1][1] * 100) / 100
            self.stepTime = [x * 0.01 for x in range(int((self.tFinal - self.tIni) / 0.01) + 1)]
            self.step_range[-1] = self.step_range[0] + len(self.stepTime) - 1
            self.listParticlesInTrajectories = [None] * len(self.stepTime)

        for idx, trajectory_type in enumerate(self.types_of_trajectories):
            # Load trajectory-specific data
            if 'arterial' in trajectory_type:
                self.dictTraj = self.dictTraj_arterial
                self.dictROI = self.dictROI_arterial
                self.NumTrajectories = self.NumTrajectories_arterial
                self.name_trajectories = self.name_arterial_trajectories
            elif 'venous' in trajectory_type:
                self.dictTraj = self.dictTraj_venous
                self.dictROI = self.dictROI_venous
                self.NumTrajectories = self.NumTrajectories_venous
                self.name_trajectories = self.name_venous_trajectories
            else:
                self.dictTraj = import_patient_trajectories(self.directory, self.folder)
                self.NumTrajectories = np.size(self.dictTraj['TrayectosVasos'])

            print(f'Type of trajectory: {trajectory_type} loaded.')
            self.TrajectoriesInDose = [TrajectoryInDose() for _ in range(self.NumTrajectories)]

            # Calculate positions in dose matrix
            print('Recalculating positions of the trajectories in the dose matrix')
            for i in range(self.NumTrajectories):
                pos_j = np.round((self.dictTraj[self.name_trajectories]['x'][0][i] - CoorX[0]) / dX + 1).astype(int)
                pos_i = np.round((self.dictTraj[self.name_trajectories]['y'][0][i] - CoorY[0]) / dY + 1).astype(int)
                pos_k = np.round((self.dictTraj[self.name_trajectories]['z'][0][i] - CoorZ[0]) / dZ + 1).astype(int)

                # Assign trajectory coordinates within dose space
                self.TrajectoriesInDose[i].i, self.TrajectoriesInDose[i].j, self.TrajectoriesInDose[
                    i].k = pos_i, pos_j, pos_k

                # Correct coordinates that are out of bounds
                invalid = np.where(
                    (pos_j < 1) | (pos_i < 1) | (pos_k < 1) | (pos_j > XD.shape[1]) | (pos_i > YD.shape[0]) | (
                                pos_k > ZD.shape[2]))[0]
                self.TrajectoriesInDose[i].i[invalid], self.TrajectoriesInDose[i].j[invalid], \
                self.TrajectoriesInDose[i].k[invalid] = -1, -1, -1

            if CurrentField == 1:
                prob_enter_traj = self.calculate_prob_enter_trajectories(trajectory_type)
                if 'arterial' in trajectory_type:
                    self.prob_enter_traj_arterial = prob_enter_traj
                elif 'venous' in trajectory_type:
                    self.prob_enter_traj_venous = prob_enter_traj

            # Store trajectory data based on type
            if 'arterial' in trajectory_type:
                self.TrajectoriesInDose_arterial, self.PosI_arterial, self.PosJ_arterial, self.PosK_arterial, self.Which_arterial = \
                    self.TrajectoriesInDose, pos_i, pos_j, pos_k, invalid
            elif 'venous' in trajectory_type:
                self.TrajectoriesInDose_venous, self.PosI_venous, self.PosJ_venous, self.PosK_venous, self.Which_venous = \
                    self.TrajectoriesInDose, pos_i, pos_j, pos_k, invalid

        # Clear unused variables
        self.TrajectoriesInDose = self.PosI = self.PosJ = self.PosK = self.Which = self.dictTraj = self.dictROI = self.NumTrajectories = self.name_trajectories = None

    def calculate_prob_enter_trajectories(self, type_trajectory):
        # Updated October 2024
        # it is launched a message about the type of trajectory that has been loaded.
        print(f'Probability of enter in {type_trajectory} trajectories loaded.')
        print('\n')

        # Determine the probability of entering in each trajectory for the blood particles:
        vel_ini_traj = []
        # Change over time
        delta_t = 0.01
        for i in range(np.size(self.dictROI['PIS']['x'][0][0])):
            r = np.where(np.round(self.dictTraj[self.name_trajectories]['x'][0][i].flatten(), 4) == np.round(self.dictROI['PIS']['x'][0][0][0][i], 4))[0]
            c = np.where(np.round(self.dictTraj[self.name_trajectories]['y'][0][i].flatten(), 4) == np.round(self.dictROI['PIS']['y'][0][0][0][i], 4))[0]
            s = np.where(np.round(self.dictTraj[self.name_trajectories]['z'][0][i].flatten(), 4) == np.round(self.dictROI['PIS']['z'][0][0][0][i], 4))[0]
            index = []
            index = np.intersect1d(np.intersect1d(r, c), s)[0]
            if np.size(index) != 0:
                # Initial positions and Initial + 1 positions
                x1, y1, z1 = self.dictTraj[self.name_trajectories]['x'][0][i][index][0], self.dictTraj[self.name_trajectories]['y'][0][i][index][0], self.dictTraj[self.name_trajectories]['z'][0][i][index][0]
                x2, y2, z2 = self.dictTraj[self.name_trajectories]['x'][0][i][index+1][0], self.dictTraj[self.name_trajectories]['y'][0][i][index+1][0], self.dictTraj[self.name_trajectories]['z'][0][i][index+1][0]
                # Calculate velocities in each spatial coordinate
                vx = (x2 - x1) / delta_t
                vy = (y2 - y1) / delta_t
                vz = (z2 - z1) / delta_t
                # Calculate total velocity
                vel_ini_traj.append(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
        # Calculate the probability based on the initial velocity at the start of each trajectory
        prob_enter_traj = []
        sum_vel = sum(vel_ini_traj[:])
        for i in range(len(vel_ini_traj)):
            p = vel_ini_traj[i] / sum_vel
            prob_enter_traj.append(p)

        return prob_enter_traj

    def calculate_mtt_patient(self, prob_enter_traj):
        mean_transition_time = []
        for i in range(self.NumTrajectories):
            mean_transition_time.append(np.size(self.dictTraj[self.name_trajectories]['t'][0][i]) * self.dt)

        mtt_flip = np.average(mean_transition_time, weights=prob_enter_traj)

        return mtt_flip

    def fill_trajectories(self, num_arterial_bp_in_flip, num_venous_bp_in_flip):
        # Updated function October 2024
        # Mapping trajectory types to their corresponding variables
        parts_map = {
            'arterial': {
                'dictTraj': self.dictTraj_arterial,
                'NumTrajectories': self.NumTrajectories_arterial,
                'name_trajectories': self.name_arterial_trajectories,
                'ListActiveParticles': self.ListActiveParticles_arterial,
                'prob_enter_traj': self.prob_enter_traj_arterial,
                'num_bp_in_flip': num_arterial_bp_in_flip
            },
            'venous': {
                'dictTraj': self.dictTraj_venous,
                'NumTrajectories': self.NumTrajectories_venous,
                'name_trajectories': self.name_venous_trajectories,
                'ListActiveParticles': self.ListActiveParticles_venous,
                'prob_enter_traj': self.prob_enter_traj_venous,
                'num_bp_in_flip': num_venous_bp_in_flip
            }
        }

        # Travel through the parts of the vascular system
        for part_type, part_data in parts_map.items():
            dictTraj = part_data['dictTraj']
            NumTrajectories = part_data['NumTrajectories']
            name_trajectories = part_data['name_trajectories']
            ListActiveParticles = part_data['ListActiveParticles']
            prob_enter_traj = part_data['prob_enter_traj']
            num_bp_in_flip = part_data['num_bp_in_flip']

            # Assign the blood particles to the trajectories
            for bp in num_bp_in_flip:
                # Select a trajectory randomly based on probability
                num_traj_choice = np.random.choice(range(NumTrajectories), p=prob_enter_traj)

                # Number of nodes in the chosen trajectory
                NumNodesTraj = np.size(dictTraj[name_trajectories]['t'][0][num_traj_choice])

                # Choose a random position within the trajectory
                pos_choice_in_traj = np.random.choice(range(NumNodesTraj))

                # Configure the attributes of the last particle
                # Note: "ultimo" is "last" in english...
                dictTraj[name_trajectories]['ultimo_t'][0][num_traj_choice] = 0
                ListActiveParticles.append(bp)
                particle = self.Particles[bp]
                particle.Flag = 0  # The particle is in an irradiable area
                particle.Trajectory = num_traj_choice
                particle.PosTraj = pos_choice_in_traj
                particle.Dose = 0  # Initial dose
                particle.Index = bp

            # Save the active particles in the correct place (variable)
            if part_type == 'arterial':
                self.ListActiveParticles_arterial = ListActiveParticles
            elif part_type == 'venous':
                self.ListActiveParticles_venous = ListActiveParticles

            print(f'Trajectories from {part_type} part of the vasculature are filled.')

    def UpdateDosePerControlPoint(self, time, factor):
        NumControlPoint = len(self.dictDoseRateNorm['DoseRateNorm'][self.field][0][0][0][:])
        is_not_in_any = 1
        patient_step_time = self.dt / factor

        for i in range(NumControlPoint):
            current_instant_dose = copy.deepcopy(self.doseInfo[i])
            doseRate = self.dictDoseRateNorm['DoseRateNorm'][self.field][0][0][0][i]

            if round(self.dictTimes[self.field][0][0][i][0], 10) < round((time + patient_step_time), 10) and round(self.dictTimes[self.field][0][0][i][0], 10) >= round(time, 10):
                temporal_step = (time + patient_step_time) - self.dictTimes[self.field][0][0][i][0]
                is_not_in_any = 0
                #print(f'It starts an energy layer in time {time}')
                break

            if round(self.dictTimes[self.field][0][0][i][1], 10) >= round(time, 10) and round(self.dictTimes[self.field][0][0][i][1], 10) < round((time + patient_step_time), 10):
                temporal_step = self.dictTimes[self.field][0][0][i][1] - time
                is_not_in_any = 0
                #print(f'It ends an energy layer in time {time}')
                break

            if round(time, 10) < round(self.dictTimes[self.field][0][0][i][1], 10) and round(time, 10) > round(self.dictTimes[self.field][0][0][i][0], 10):
                temporal_step = patient_step_time  # 1 / 100 centesimas de segundo
                is_not_in_any = 0
                #print(f'In the middle of the energy layer in time {time}')
                break

        if is_not_in_any:
            doseRate = 0
            temporal_step = 0

        self.InstantDose = copy.deepcopy(current_instant_dose)
        self.InstantDose.Dose = self.InstantDose.Dose * doseRate * temporal_step
        return self.InstantDose

    def bp_travelling_along_simulation(self, index_c, index_c_arterial, index_c_venous, step):
        # Updated function: 22 october 2024
        # Set corresponding trajectory and particle lists based on the index_c
        if index_c == index_c_arterial:
            self.dictTraj, self.name_trajectories = self.dictTraj_arterial, self.name_arterial_trajectories
            self.TrajectoriesInDose, self.ListActiveParticles = self.TrajectoriesInDose_arterial, self.ListActiveParticles_arterial
            flip = 'flip_arterial'
            flow_factor = self.factor_arterial
            # print(f' Length ListActiveParticles_arterial: {len(self.ListActiveParticles)} at step: {step}')
        elif index_c == index_c_venous:
            self.dictTraj, self.name_trajectories = self.dictTraj_venous, self.name_venous_trajectories
            self.TrajectoriesInDose, self.ListActiveParticles = self.TrajectoriesInDose_venous, self.ListActiveParticles_venous
            flip = 'flip_venous'
            flow_factor = self.factor_venous
            #print(f' Length ListActiveParticles_venous: {len(self.ListActiveParticles)} at step: {step}')

        NumParticlesToRadiate = len(self.ListActiveParticles)

        if self.step_range[0] <= step <= self.step_range[1]:
            special_step = step - self.step_range[0]
            time_now = np.round(self.stepTime[special_step], 2)

            # Update the dose per control point and log time if close to an integer
            self.InstantDose = self.UpdateDosePerControlPoint(time=time_now, factor=flow_factor)
            if abs(time_now - round(time_now)) < 1e-4:
                print(
                    f't = {time_now:.2f} of {self.tFinal - self.tIni:.2f} {self.field} of {self.number_of_fields} - {flip} compartment')

            # Update the number of particles in the trajectory at the current step
            if self.listParticlesInTrajectories[special_step] is None:
                self.listParticlesInTrajectories[special_step] = NumParticlesToRadiate
            else:
                self.listParticlesInTrajectories[special_step] += NumParticlesToRadiate

        self.out_comp_flip = []

        # Radiate and move particles
        for _ in np.arange(0, self.dt, self.dt / flow_factor):
            for k in range(NumParticlesToRadiate):
                particle = self.Particles[self.ListActiveParticles[k]]
                where = particle.PosTraj

                # Access coordinates
                traj = self.TrajectoriesInDose[particle.Trajectory]
                CoordI, CoordJ, CoordK = traj.i[where][0] - 1, traj.j[where][0] - 1, traj.k[where][0] - 1

                # Update dose if within bounds
                if CoordI != -1 and (self.step_range[0] <= step <= self.step_range[1]):
                    particle.Dose += self.InstantDose.Dose[CoordI, CoordJ, CoordK]

                # Move particle forward
                particle.PosTraj += 1

            # Identify and remove particles that are out of bounds
            to_delete = [
                p for p in self.ListActiveParticles if self.Particles[p].PosTraj >= len(
                    self.dictTraj[self.name_trajectories]['x'][0][self.Particles[p].Trajectory])
            ]

            for p in to_delete:
                particle = self.Particles[p]
                particle.Flag = 1
                self.out_comp_flip.append(p)
                # particle.When_out.append(step)
                self.ListActiveParticles.remove(p)

            # Update NumParticlesToRadiate after deletion
            NumParticlesToRadiate = len(self.ListActiveParticles)

        self.outflow_flip.append(len(self.out_comp_flip))

        # Check if it's time to change the field
        if self.first_treatment_field < self.number_of_fields and self.step_range[0] <= step <= self.step_range[1]:
            if time_now == np.round(self.stepTime[-1], 2):
                self.change_field = True

        # Clear variables and update lists for next iteration
        self.dictTraj, self.name_trajectories, self.TrajectoriesInDose = None, None, None
        if index_c == index_c_arterial:
            self.ListActiveParticles_arterial = self.ListActiveParticles
        elif index_c == index_c_venous:
            self.ListActiveParticles_venous = self.ListActiveParticles
        self.ListActiveParticles = []

    def add_blood_particles(self, index_in_flip_to_use, index_c, index_c_arterial, index_c_venous, step):
        # Updated october 2024
        # Selection of the variables depending on if it is arterial or venous
        if index_c == index_c_arterial:
            part_data = {
                'dictTraj': self.dictTraj_arterial,
                'NumTrajectories': self.NumTrajectories_arterial,
                'name_trajectories': self.name_arterial_trajectories,
                'ListActiveParticles': self.ListActiveParticles_arterial,
                'prob_enter_traj': self.prob_enter_traj_arterial
            }
            # print(f' Length in_flip_to_use: {len(index_in_flip_to_use)} at step: {step}')
        elif index_c == index_c_venous:
            part_data = {
                'dictTraj': self.dictTraj_venous,
                'NumTrajectories': self.NumTrajectories_venous,
                'name_trajectories': self.name_venous_trajectories,
                'ListActiveParticles': self.ListActiveParticles_venous,
                'prob_enter_traj': self.prob_enter_traj_venous
            }

        # Destructuring variables to simplify access
        dictTraj = part_data['dictTraj']
        NumTrajectories = part_data['NumTrajectories']
        name_trajectories = part_data['name_trajectories']
        ListActiveParticles = part_data['ListActiveParticles']
        prob_enter_traj = part_data['prob_enter_traj']

        for bp in index_in_flip_to_use:
            # Random selection of a trajectory
            num_traj_choice = np.random.choice(NumTrajectories, p=prob_enter_traj)

            # Record the time of the last particle on the selected trajectory
            # Note: "ultimo" is "last" in english...
            dictTraj[name_trajectories]['ultimo_t'][0][num_traj_choice] = np.round(step, 2)

            # Update the list of active particles
            ListActiveParticles.append(bp)
            self.LastParticle = bp

            # Update the particle's attributes
            particle = self.Particles[bp]
            particle.Flag = 0  # Puede ser irradiada
            particle.Trajectory = num_traj_choice  # Trayectoria seleccionada
            particle.PosTraj = 0  # Posici�n inicial en la trayectoria
            particle.Dose = particle.Dose  # Dosis recibida
            particle.Index = bp

        # Update the active particles according to type (arterial or venous)
        if index_c == index_c_arterial:
            self.ListActiveParticles_arterial = ListActiveParticles
            # print(f' Length ListActiveParticles_arterial in add: {len(self.ListActiveParticles_arterial)} at step: {step}')
        elif index_c == index_c_venous:
            self.ListActiveParticles_venous = ListActiveParticles
            #print(f' Length ListActiveParticles_venous in add: {len(self.ListActiveParticles_venous)} at step: {step}')

        # Reset the list of active particles for the next iteration
        self.ListActiveParticles = []

    def mean_organ_dose(self, index_c, index_c_arterial, index_c_venous, step):
        if index_c == index_c_arterial:
            self.dictTraj = self.dictTraj_arterial
            self.name_trajectories = self.name_arterial_trajectories
            self.TrajectoriesInDose = self.TrajectoriesInDose_arterial
            self.organ_dose = self.mean_organ_dose_arterial
            self.NumTrajectories = self.NumTrajectories_arterial
        if index_c == index_c_venous:
            self.dictTraj = self.dictTraj_venous
            self.name_trajectories = self.name_venous_trajectories
            self.TrajectoriesInDose = self.TrajectoriesInDose_venous
            self.organ_dose = self.mean_organ_dose_venous
            self.NumTrajectories = self.NumTrajectories_venous

        time_now = np.round(self.stepTime[step], 2)  # stepTime[t]
        self.InstantDose = self.UpdateDosePerControlPoint(time=time_now, factor=self.factor)

        organ_dose = 0
        # I am going to access to each trajectory:
        for t in range(np.size(self.dictTraj[self.name_trajectories])):
            # I want to know the number of nodes that the trajectory has:
            NumNodesTraj = np.size(self.dictTraj[self.name_trajectories]['t'][0][t])
            # Knowing the number of nodes, I want to go one by one checking how much dose is receiving:
            for n in range(NumNodesTraj):
                # I access the i, j and k coordinates, knowing the node (n)
                c_i = self.TrajectoriesInDose[t].i[n][0]-1
                c_j = self.TrajectoriesInDose[t].j[n][0]-1
                c_k = self.TrajectoriesInDose[t].k[n][0]-1
                # The dose is going to be updated
                if c_i != -1:
                    organ_dose = organ_dose + self.InstantDose.Dose[c_i, c_j, c_k]
        self.organ_dose.append(organ_dose)

        # I clear:
        self.dictTraj = None
        self.name_trajectories = None
        self.TrajectoriesInDose = None
        self.NumTrajectories = None
        if index_c == index_c_arterial:
            self.mean_organ_dose_arterial = self.organ_dose
        if index_c == index_c_venous:
            self.mean_organ_dose_venous = self.organ_dose
        self.organ_dose = []
