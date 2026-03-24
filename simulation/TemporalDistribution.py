# ========= Created by Chris Beekman et al. 2023 ============
# ========= Modified by Marina Garcia-Cardosa since January 2024 to manage with data patients =========
# ========= Last update done in Novemeber 2024 by Marina Garcia-Cardosa to enhance and optimize it =========
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objs as go
import pandas as pd
import time
import seaborn as sns
import os


class TemporalDistribution:
    def __init__(self, model):
        self.model = model
        self.ttd = {}
        self.rtd = {}
        self.path = None
        self.tv = None

    def generate_from_markov(self):
        """
        Generate a temporal distribution from a pre-built markov chain.
        """
        t = time.process_time()
        compartment_id = self.model.cum_volume.searchsorted(
            np.random.uniform(size=self.model.sample_size)).astype(np.uint8)
        self.path = self.model.chain.walk(self.model.nr_steps, compartment_id)
        print(f'Time to generate simulation distribution: {time.process_time()-t:.6f} seconds')

    def generate_from_weibull(self):
        """
        Generate a temporal distribution from a pre-built chain using Weibull distribution.
        """
        # it starts to count time...
        start_time = time.process_time()
        # it is obtained the id of each compartment respect the sorted ascend array of cum_volume
        compartment_id = self.model.cum_volume.searchsorted(
            np.random.uniform(size=self.model.sample_size)).astype(np.uint8)
        # the BPs start to travel along the different compartments...
        self.path = self.model.chain.walk_v1(self.model.nr_steps, compartment_id)
        # self.path = self.model.chain.walk_v2(self.model.nr_steps, compartment_id)
        # it is printed how long the simulation took
        print(f'Time to generate temporal distribution: {time.process_time()-start_time:.2f} seconds')

    def generate_from_weibull_fliphedos(self):
        """
        Generate a temporal distribution from a pre-built chain using Weibull distribution.
        """
        # it starts to count time...
        start_time = time.process_time()
        # it is obtained the id of each compartment respect the sorted ascend array of cum_volume
        compartment_id = self.model.cum_volume.searchsorted(
            np.random.uniform(size=self.model.sample_size)).astype(np.uint8)
        # the BPs start to travel along the different compartments...
        index_c_arterial = np.where(np.array(self.model.names) == 'flip_arterial')[0][0]
        index_c_venous = np.where(np.array(self.model.names) == 'flip_venous')[0][0]
        self.path = self.model.chain.walk_v1_fliphedos(self.model.patient.nr_steps, compartment_id, index_c_arterial, index_c_venous, self.model.patient)
        # it is printed how long the simulation took
        print(f'Time to generate temporal distribution: {time.process_time()-start_time:.2f} seconds')

    def temporal_volume(self):
        """
        Calculate volume changes in time. # of BP x # of time-steps
        """
        start_time = time.process_time()
        self.tv = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=0, arr=self.path)
        self.tv = self.tv[:len(self.model.names)] / self.model.sample_size
        print(f'Time to get temporal volumes: {time.process_time() - start_time:.2f} seconds')

    def save(self, f_name):
        """
        Save simulation path for potential re-use
        """
        np.save(f_name, self.path)

    def load(self, f_name):
        """
        Load simulation distribution
        """
        self.path = np.load(f_name)

    def _particle_entry_exit(self, compartment_id):
        # which rows (=particle ids) pass given compartment at some point during simulated time:
        rows = np.amax(self.path == compartment_id, axis=1)
        # for these particle ids, when is it in the compartment:
        in_comp = np.array(self.path[rows] == compartment_id, dtype=np.int32)
        # when does it enter and leave the compartment?:
        diff = np.concatenate([in_comp[:, 0][:, None], np.diff(in_comp, axis=1), -in_comp[:, -1][:, None]], axis=1)
        particle_id, t_entry = np.where(diff == 1)
        t_exit = np.where(diff == -1)[1]
        return in_comp, particle_id, t_entry, t_exit

    def _get_time_distributions(self, name, nr_particles_passed):
        """
        Due to finite width of time window, this is biased toward shorter transition/recurrence times as their
        probability of falling within the window in greater. I think the correct term for this is "right censoring".
        To quantify this a bit, we print the percentage of transit/recurrence times that were contained
        in the simulated time window.
        For this we need an estimate of particles that have past the compartment during the simulation.
        This is given by the "nr_particles_passed" parameter.
        """
        # find the time indices where the particle enters and exits the compartment.
        # get transition and recurrence times by subtraction.
        compartment_id = self.model.names.index(name)
        in_comp, particle_id, t_entry, t_exit = self._particle_entry_exit(compartment_id)
        # Calculate tts
        ttd = (t_exit - t_entry) * self.model.dt
        # beginning and tail are potentially cut off due to specific time window, discard these.
        self.ttd[name] = ttd[in_comp[particle_id, 0] + in_comp[particle_id, -1] == 0]
        # Calculate rts, deleting entries that do not correspond to the same particle.
        rtd = (t_entry[1:] - t_exit[:-1]) * self.model.dt
        self.rtd[name] = rtd[np.where(particle_id[1:] == particle_id[:-1])]

        print('Only the shortest {:.3f}% of transit times captured in {:}, MTT therefore underestimation.'.format(
            self.ttd[name].size / nr_particles_passed * 100, name))
        print('Only the shortest {:.3f}% of recurrence times captured in {:}, MRT therefore underestimation.'.format(
            self.rtd[name].size / nr_particles_passed * 100, name))

    def _transition_recurrence_time(self, expected_nr_particles, names=None):
        """
        Calculate transition time tt-distribution (ttd), and recurrence time rt-distribution (rtd)
        """
        if names is None:
            names = self.model.names

        start_time = time.process_time()
        for name in names:
            nr_particles_passed = expected_nr_particles[self.model.names.index(name)]
            self._get_time_distributions(name, nr_particles_passed)
        print(f'Time to get transition times: {time.process_time() - start_time:.6f} seconds')

    def _plot_hist(self, names, time_distribution, name_of_mean):
        _, axes = plt.subplots(nrows=len(names), ncols=1, figsize=(6, 4 * len(names)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for ax, name in zip(axes, names):
            v_max = 3 * np.percentile(time_distribution[name], 90)
            n_bins = int(v_max // self.model.dt)
            ax.hist(time_distribution[name], bins=np.linspace(0, v_max, n_bins),
                    density=True, label=name)
            ax.legend()
            ax.set_title(name_of_mean + ' = {:.3f}'.format(np.mean(time_distribution[name])))
        plt.xlabel('time (s)')
        plt.show()

    def plot_time_distributions(self, names):
        """
        This plots both the simulated transit time distribution and the recurrence time distribution.
        Note that both are right-censored; their mean will therefore be an underestimation.
        """
        assert(isinstance(names, list)), '"names" should be a list.'
        # obtain the expected nr of particles crossing through each organ in the given time window.
        volume_passed = self.model.flows * self.path.shape[1] * self.model.dt
        expected_nr_particles = volume_passed / self.model.particle_volume
        # calculate transit and recurrence time distributions.
        self._transition_recurrence_time(expected_nr_particles, names)

        self._plot_hist(names, time_distribution=self.ttd, name_of_mean='MTT')
        self._plot_hist(names, time_distribution=self.rtd, name_of_mean='MRT')

    def plot_inflow_outflow(self, names):
        """
        This plots both the flow into and out off a compartment.
        Clearly in equilibrium these should be the same and equal the intended compartmental flow.
        """
        for name in names:
            compartment_id = self.model.names.index(name)
            in_comp, particle_id, t_entry, t_exit = self._particle_entry_exit(compartment_id)
            ti, count = np.unique(t_entry, return_counts=True)
            plt.plot(ti[1:-1] * self.model.dt, count[1:-1] * self.model.particle_volume * 1000 / self.model.dt,
                     label=name + ' -- inflow')
            ti, count = np.unique(t_exit, return_counts=True)
            plt.plot(ti[1:-1] * self.model.dt, count[1:-1] * self.model.particle_volume * 1000 / self.model.dt,
                     label=name + ' -- outflow')
        plt.xlim([0, self.path.shape[1] * self.model.dt])
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('flow (mL/s)')
        plt.show()

    def plot_volumes_over_time(self):
        """
        This plots the (simulation) volumes of the all compartments over time.
        Again, in equilibrium, these lines should be straight (with some stochastic noise)
        and reflect the intended simulation volume in each compartment.
        """
        if self.tv is None:
            self.temporal_volume()
        ti = np.arange(self.tv.shape[1]) * self.model.dt
        for i, name in enumerate(self.model.names):
            plt.plot(ti, self.tv[i], label=name)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Total blood volume percentage (%)')
        plt.show()

    def plot_volumes_over_time_flip(self):
        """
        This plots the (simulation) volumes of the FLIP compartments over time.
        Again, in equilibrium, these lines should be straight (with some stochastic noise,
        they may fluctuate) and reflect the intended simulation volume in each compartment.
        """
        # Find the indices for ‘flip_arterial’ and ‘flip_venous’ in a more compact way
        flip_comps = [
            self.model.names.index('flip_arterial'),
            self.model.names.index('flip_venous')
        ]

        # Vectorize the operation of finding the 'flip' compartments in self.path
        flip_indices = np.isin(self.path, flip_comps)

        # Calculate the volumes of both compartments using a vectorized approach
        vol_in_comps = np.sum(flip_indices, axis=0) * self.model.particle_volume

        # Separate the volumes of the arterial and venous compartments
        vol_in_comp_arterial = vol_in_comps[self.path == flip_comps[0]]
        vol_in_comp_venous = vol_in_comps[self.path == flip_comps[1]]

        # Create an interactive chart with Plotly
        fig = go.Figure()

        # Add the lines for the arterial and venous flips
        fig.add_trace(go.Scatter(x=np.arange(self.model.nr_steps), y=vol_in_comp_arterial,
                                 mode='lines', name='flip_arterial', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=np.arange(self.model.nr_steps), y=vol_in_comp_venous,
                                 mode='lines', name='flip_venous', line=dict(color='blue')))

        # Configure the figure layout
        fig.update_layout(
            title='FLIP arterial and venous volume',
            xaxis_title='Time (s)',
            yaxis_title='Blood volume (L)',
            legend_title='FLIP compartments',
            hovermode='x'
        )

        # Show the figure
        fig.show()

    def dose_metrics_flip(self):
        """
        Analyzes the radiation dose received by blood particles (BPs) in
        patient-specific (FLIP) compartments.
        * Note: The threshold of 0.001 Gy is used to exclude near-zero doses.
        """
        # Obtain all particles dose in patient-specific (FLIP) compartments
        doses = [self.model.patient.Particles[i].Dose for i in range(self.model.patient.NumParticles)]

        # Print mean dose
        print('FLIP compartments:\n • BPs mean dose: {:.5f} Gy'.format(np.mean(doses)))

        # Filter particles with doses > 0.001 Gy
        bps_with_dose = [dose for dose in doses if dose > 0.001]

        # Print the average dose for points with doses > 0.001 Gy
        print(' • Mean dose among BPs that have received > 0 Gy: {:.5f} Gy'.format(np.mean(bps_with_dose)))
        return doses, bps_with_dose

    def analyze_bps_visits_in_flip(self):
        """
        Analyzes how many times each blood particle (BP) visits the patient-specific
        vasculature compartments (arterial and venous) and computes the distribution
        of visits.
        """
        # Find indexes for patient-specific compartments
        index_c_arterial = self.model.names.index('flip_arterial')
        index_c_venous = self.model.names.index('flip_venous')

        # Create the matrix to store information about visits
        num_bps = len(self.path)
        matrix_visits_dose = np.zeros((num_bps, 1))

        # Go through each BP path and count visits to each compartment
        for i, path in enumerate(self.path):
            cont_art = (path[0] == index_c_arterial) + np.sum(
                (np.array(path[:-1]) != index_c_arterial) & (np.array(path[1:]) == index_c_arterial))
            cont_ven = (path[0] == index_c_venous) + np.sum(
                (np.array(path[:-1]) != index_c_venous) & (np.array(path[1:]) == index_c_venous))

            # Save the total number of visits and the BP dose
            matrix_visits_dose[i, 0] = cont_art + cont_ven

        # Extract the number of visits from the first column
        visits = matrix_visits_dose[:, 0]

        # Count the frequency of each number of visits
        unique_visits, counts = np.unique(visits, return_counts=True)

        # Convert to percentage
        total_particles = num_bps
        percentages = (counts / total_particles) * 100

        # Print results
        print(f' • {np.round(percentages[0], 2)} % of BPs that have not visited the patient-specific vasculature.')
        print(f' • {np.round(percentages[1], 2)} % of BPs that have visited the patient-specific vasculature once.')
        print(f' • {np.round(percentages[2], 2)} % of BPs that have visited the patient-specific vasculature twice.')
        print(f' • {np.round(np.sum(percentages[3:]), 2)} % of BPs that have visited the patient-specific vasculature more than twice.')
        print(f' • The average visits number is {np.round(np.mean(visits), 2)} visits.\n')

        self.plot_visits_histogram(unique_visits, percentages)

        return unique_visits, percentages

    def plot_visits_histogram(self, unique_visits, percentages):
        """
        Plots a histogram (bar figure) showing the distribution of visit counts
        and their corresponding percentages for blood particles.
        * Notes:
            - The function assumes that `unique_visits` and `percentages` have the same length.
            - Saving functionality is available but currently commented out.
        """
        # Create the histogram
        plt.figure(figsize=(8, 5))
        plt.bar(unique_visits, percentages, color='b', alpha=0.7)

        # Labels, grid, ticks y title
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2.5))
        plt.xlabel('Number of visits (#)', fontsize=14)
        plt.ylabel('BPs percentage (%)', fontsize=14)
        plt.title('Number of visits done by blood particles \n in FLIP (patient-specific) compartments')
        # plt.yticks(np.arange(0, 31, 5))
        # plt.ylim([0, 30])
        plt.xticks(unique_visits)  # Ensure that each number of visits has a mark on the x-axis.
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.4)

        # # In case you want to save the plot, uncomment the following lines:
        # keep = int(input('Do you want to save the figure? (Yes->1 / No->0): '))
        # if keep == 1:
        #     name_figure = input('Write the name of the figure you would like to save: ')
        #     plt.savefig(name_figure + '.pdf')

        # It shows the plot
        plt.show(block=False)

    def plot_particle_trajectories(self, time_steps=None):
        """
        Visualize the trajectory of individual particles over time.

        path_matrix: a matrix of size (num_particles, num_time_steps), where each cell indicates the compartment at each time step.
        particle_indices: a list of the indices of the particles to be visualized.
        time_steps: the number of time steps to display (optional).
        """
        plt.figure(figsize=(10, 6))

        if time_steps is None:
            time_steps = self.path.shape[1]  # Use all times unless another value is specified

        idx=0
        trajectory = self.path[idx, :time_steps]  # Get the particle's trajectory
        plt.plot(range(time_steps), trajectory, marker='o', linestyle='-', label=f'BP {idx}')

        plt.xlabel('Time')
        plt.ylabel('Compartment')
        plt.title('Trajectories of selected particles')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    def plot_normalized_volume_over_time(self, names):
        """
        This zooms in on one or multiple volumes and plots them normalized.
        Hence, in equilibrium, these lines should wiggle about the value 1.
        """
        if self.tv is None:
            self.temporal_volume()
        # normalize volumes:
        normalized_volumes = self.model.volumes / np.sum(self.model.volumes)
        print('Sum target volumes = {:.3f}'.format(np.sum(normalized_volumes)))
        final_volumes = self.tv[:, -1]
        print('Sum final volumes = {:.3f}'.format(np.sum(final_volumes)))

        _, axes = plt.subplots(nrows=len(names), ncols=1, figsize=(6, 4 * len(names)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for ax, name in zip(axes, names):
            idx = self.model.names.index(name)
            ax.plot(np.arange(self.tv[idx].size) * self.model.dt, self.tv[idx] / normalized_volumes[idx], label=name)
            ax.legend()
            print('{} -- Fraction of target volume = {:.3f}'.format(name, final_volumes[idx] / normalized_volumes[idx]))
        plt.xlabel('time (s)')
        plt.suptitle('Normalized volume over time')
        plt.show()

    def plot_final_blood_volumes(self):
        """
        This plots the final volumes + reference values.
        Perhaps better to take an average of the final x steps to get rid of some stochasticity.
        """
        bins = self.model.size
        # fig, ax = plt.subplots(1, 1, figsize=(5, 8), dpi=300)
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.path[:, -1], bins=bins, range=[-0.5, bins - 0.5],
                weights=[1/self.model.sample_size] * self.model.sample_size, density=False,
                label='simulation', orientation='horizontal', rwidth=0.7)
        ax.plot(self.model.volumes / np.sum(self.model.volumes), np.arange(bins), 'r*', label='reference')
        ax.set_yticks(list(range(0, bins)), self.model.names)
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel('Percentage of total simulation volume')
        ax.invert_yaxis()
        #plt.subplots_adjust(bottom=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

