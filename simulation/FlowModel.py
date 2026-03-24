# ========= Created by Chris Beekman et al. 2023 ============
# ========= Modified by Marina Garcia-Cardosa since January 2024 to manage with data patients =========
# ========= Last update done in Novemeber 2024 by Marina Garcia-Cardosa to enhance and optimize it =========
import numpy as np
import networkx as nx
import pandas as pd

from simulation import Chain, MarkovChain
from workflows.PatientSpecificDataFLIPmethod import Patient


class FlowModel:
    def __init__(self, filename, patient_params, simulation_params):
        """
        Constructor for CompartmentNetwork
        total_vol : (L)
        total_flow : (L/s)
        dt : time step in seconds
        """
        self.total_volume = patient_params['TBV']
        self.total_flow = patient_params['CO']
        self.sample_size = simulation_params['sample_size']
        self.nr_steps = simulation_params['nr_steps']
        self.dt = simulation_params['dt']
        self.weibull_shape = simulation_params['weibull_shape']

        self.df = self._read_excel_file(filename, sheetname=patient_params['sheet_name'])
        self.size = self.df.index.name
        self.names = list(self.df.index.values[:self.size])
        # these are given as percentages
        self.flows = np.array(self.df.flow_sum[:self.size].values, dtype=np.float64) / 100 * self.total_flow
        self.volumes = np.array(self.df.volume[:self.size].values, dtype=np.float64) / 100 * self.total_volume
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)
        self.particle_volume = self.total_volume / self.sample_size

        # get the rate matrix:
        self.k_matrix = self._get_rate_matrix()
        # convert k_matrix to a graph:
        self._rate_matrix_to_graph()
        self._get_mtts()
        # initialize probability matrix
        self.prob = None

        # initialize chain:
        self.chain = None
        print('Compartmental simulation initialized. \n')

    def _read_excel_file(self, filename, sheetname):
        df = pd.read_excel(filename, sheet_name=sheetname, engine="openpyxl", index_col=0)
        df.fillna(0, inplace=True)
        return df

    def _get_rate_matrix(self):
        k_matrix = np.array(self.df.values[:self.size, :self.size], dtype=np.float64) / 100 * self.total_flow
        k_matrix /= self.volumes[:, None]
        return k_matrix   # 1 / min

    def _get_mtts(self):
        # sort (graph does not have an order necessarily).
        idx = np.array([list(self.G.nodes.keys()).index(name) for name in self.names], dtype=int)
        self.mtt = 1 / np.sum(self.k_matrix, axis=1)[idx]

    def _rate_matrix_to_graph(self):
        # condense flow kinetics in a directed graph with rates k
        adjacency = np.array(self.k_matrix, dtype=[('k', np.float32)])
        self.G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph())
        node_attr = {i: volume for i, volume in enumerate(self.volumes)}
        nx.set_node_attributes(self.G, node_attr, 'V')
        mapping = {i: name for i, name in enumerate(self.names)}
        nx.relabel_nodes(self.G, mapping, copy=False)

    def _graph_to_rate_matrix(self):
        self.k_matrix = nx.to_numpy_array(self.G, weight='k')

    def _get_transition_matrix(self):
        # convert the graph into a transition matrix.
        self._graph_to_rate_matrix()
        self.prob = self.k_matrix * self.dt
        # sort (graph does not have an order necessarily).
        idx = np.array([list(self.G.nodes.keys()).index(name) for name in self.names], dtype=int)
        self.prob = self.prob[idx][:, idx]
        assert(np.array(np.sum(self.prob, axis=1) < np.ones(self.prob.shape[0])).all()), \
            'time step size is too large; leaving probabilities > 1 encountered.'
        # probability of staying
        np.fill_diagonal(self.prob, 1.0 - np.sum(self.prob, axis=1))

    def construct_weibull(self):
        self._get_transition_matrix()
        # Construct jumping process using Weibull distribution
        self.chain = Chain(self.names, self.prob, self.mtt, dt=self.dt, k=self.weibull_shape)

    def construct_markov(self):
        self._get_transition_matrix()
        # Construct markov chain
        self.chain = MarkovChain(self.names, self.prob)


class ExpandFlowModel(FlowModel):
    """
    Inherits from FlowModel. The idea is that we can dynamically add a tumor box.
    In this implementation, the tumor box get added in parallel to the tumor-site from which it 'steals' simulation.
    How much? That is given by the volume fraction (size)
    and the relative simulation density and perfusion of tumor vs tumor-site.
    """
    def __init__(self, filename, patient_params, simulation_params):
        # inherit from base class:
        FlowModel.__init__(self, filename, patient_params, simulation_params)

    def _add_box(self, name, site, blood_volume_fraction):
        idx = self.names.index(site)
        self.size += 1
        self.names.insert(idx + 1, name)

        # adjust volume of original site and added box:
        orig_volume = self.volumes[idx]
        self.volumes[idx] = (1 - blood_volume_fraction) * orig_volume
        self.volumes = np.insert(self.volumes, idx + 1, blood_volume_fraction * orig_volume)
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)
        self.G.nodes[site]['V'] = self.volumes[idx]
        self.G.add_node(name, V=self.volumes[idx + 1])
        return idx

    def split_box_parallel(self, name, box_dict):
        site = box_dict['tumor_site']
        volume_fraction = box_dict['tumor_volume_fraction']
        relative_blood_density = box_dict['relative_blood_density']
        relative_perfusion = box_dict['relative_perfusion']

        blood_volume_fraction = volume_fraction * relative_blood_density
        blood_flow_fraction = volume_fraction * relative_perfusion

        assert (blood_volume_fraction < 1.0), \
            'Cannot steal more than 100% of the original simulation volume.'
        assert(blood_flow_fraction < 1.0), \
            'Cannot steal more than 100% of the original flow.'
        idx = self._add_box(name, site, blood_volume_fraction)

        # adjust flow original site and added box:
        orig_flow = self.flows[idx]
        self.flows[idx] = (1 - blood_flow_fraction) * orig_flow
        self.flows = np.insert(self.flows, idx + 1, blood_flow_fraction * orig_flow)

        # adjust network:
        for prev_comp in self.G.predecessors(site):
            orig_rate = self.G.edges[(prev_comp, site)]['k']
            # rate changes as flow since the simulation volume of the predecessor of the site remains equal.
            site_rate = (1 - blood_flow_fraction) * orig_rate
            box_rate = blood_flow_fraction * orig_rate
            self.G.edges[(prev_comp, site)]['k'] = site_rate
            self.G.add_edge(prev_comp, name, k=box_rate)
        for next_comp in self.G.successors(site):
            orig_rate = self.G.edges[(site, next_comp)]['k']
            # Now both the flow and volume are different...
            site_rate = (1 - blood_flow_fraction) / (1 - blood_volume_fraction) * orig_rate
            box_rate = blood_flow_fraction / blood_volume_fraction * orig_rate
            self.G.edges[(site, next_comp)]['k'] = site_rate
            self.G.add_edge(name, next_comp, k=box_rate)

        # the rates have changed so we have to update the rate_matrix and MTTs:
        self._graph_to_rate_matrix()
        self._get_mtts()


class ExpandFlowModelPatient(FlowModel):
    """
    Inherits from FlowModel. The idea is that we can add boxes in parallel or in series, or in both configurations,
    to represent new organs or specific regions of existing organs, depending on our needs.
    """
    def __init__(self, filename, patient_params, simulation_params, patient_directory, treatment_params):
        # inherit from base class:
        FlowModel.__init__(self, filename, patient_params, simulation_params)
        self.patient = Patient(patient_directory, treatment_params, simulation_params)

    def _add_box(self, name, site, blood_volume_fraction):
        idx = self.names.index(site)
        self.size += 1
        self.names.insert(idx + 1, name)

        # adjust volume of original site and added box:
        orig_volume = self.volumes[idx]
        self.volumes[idx] = (1 - blood_volume_fraction) * orig_volume
        self.volumes = np.insert(self.volumes, idx + 1, blood_volume_fraction * orig_volume)
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)
        self.G.nodes[site]['V'] = self.volumes[idx]
        self.G.add_node(name, V=self.volumes[idx + 1])
        return idx

    def _add_new_box(self, name, site, blood_volume):
        idx = self.names.index(site)
        self.size += 1
        self.names.insert(idx + 1, name)

        # adjust volume of new added box:
        self.volumes = np.insert(self.volumes, idx + 1, blood_volume)
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)
        self.G.add_node(name, V=self.volumes[idx + 1])
        return idx

    def split_box_parallel(self, name, box_dict):
        if name == 'flip_arterial':
            site = box_dict['flip_site_arterial']
            volume_fraction = box_dict['flip_volume_arterial']
            relative_blood_density = box_dict['relative_blood_density']
            relative_perfusion = box_dict['relative_perfusion']
        if name == 'flip_venous':
            site = box_dict['flip_site_venous']
            volume_fraction = box_dict['flip_volume_venous']
            relative_blood_density = box_dict['relative_blood_density']
            relative_perfusion = box_dict['relative_perfusion']

        blood_volume_fraction = volume_fraction * relative_blood_density
        blood_flow_fraction = volume_fraction * relative_perfusion

        assert (blood_volume_fraction < 1.0), \
            'Cannot steal more than 100% of the original simulation volume.'
        assert(blood_flow_fraction < 1.0), \
            'Cannot steal more than 100% of the original flow.'
        idx = self._add_box(name, site, blood_volume_fraction)

        # adjust flow original site and added box:
        orig_flow = self.flows[idx]
        self.flows[idx] = (1 - blood_flow_fraction) * orig_flow
        self.flows = np.insert(self.flows, idx + 1, blood_flow_fraction * orig_flow)

        # adjust network:
        for prev_comp in self.G.predecessors(site):
            orig_rate = self.G.edges[(prev_comp, site)]['k']
            # rate changes as flow since the simulation volume of the predecessor of the site remains equal.
            site_rate = (1 - blood_flow_fraction) * orig_rate
            box_rate = blood_flow_fraction * orig_rate
            self.G.edges[(prev_comp, site)]['k'] = site_rate
            self.G.add_edge(prev_comp, name, k=box_rate)
        for next_comp in self.G.successors(site):
            orig_rate = self.G.edges[(site, next_comp)]['k']
            # Now both the flow and volume are different...
            site_rate = (1 - blood_flow_fraction) / (1 - blood_volume_fraction) * orig_rate
            box_rate = blood_flow_fraction / blood_volume_fraction * orig_rate
            self.G.edges[(site, next_comp)]['k'] = site_rate
            self.G.add_edge(name, next_comp, k=box_rate)

        # the rates have changed so we have to update the rate_matrix and MTTs:
        self._graph_to_rate_matrix()
        self._get_mtts()

    def split_box_parallel_flip(self, name, mtt_flip, box_dict):
        if name == 'flip_arterial':
            site = box_dict['flip_site_arterial']
            volume_fraction = box_dict['flip_volume_arterial']
            relative_blood_density = box_dict['relative_blood_density']
            if 'mtt_aorta' in box_dict:
                relative_perfusion = box_dict['mtt_aorta'] / mtt_flip
            if 'mtt_head_neck_arteries' in box_dict:
                relative_perfusion = box_dict['mtt_head_neck_arteries'] / mtt_flip
        if name == 'flip_venous':
            site = box_dict['flip_site_venous']
            volume_fraction = box_dict['flip_volume_venous']
            relative_blood_density = box_dict['relative_blood_density']
            if 'mtt_ivc' in box_dict:
                relative_perfusion = box_dict['mtt_ivc'] / mtt_flip
            if 'mtt_head_neck_veins' in box_dict:
                relative_perfusion = box_dict['mtt_head_neck_veins'] / mtt_flip

        blood_volume_fraction = volume_fraction * relative_blood_density
        blood_flow_fraction = volume_fraction * relative_perfusion

        assert (blood_volume_fraction < 1.0), \
            'Cannot steal more than 100% of the original simulation volume.'
        assert(blood_flow_fraction < 1.0), \
            'Cannot steal more than 100% of the original flow.'
        idx = self._add_box(name, site, blood_volume_fraction)

        # adjust flow original site and added box:
        orig_flow = self.flows[idx]
        self.flows[idx] = (1 - blood_flow_fraction) * orig_flow
        self.flows = np.insert(self.flows, idx + 1, blood_flow_fraction * orig_flow)

        # adjust network:
        for prev_comp in self.G.predecessors(site):
            orig_rate = self.G.edges[(prev_comp, site)]['k']
            # rate changes as flow since the simulation volume of the predecessor of the site remains equal.
            site_rate = (1 - blood_flow_fraction) * orig_rate
            box_rate = blood_flow_fraction * orig_rate
            self.G.edges[(prev_comp, site)]['k'] = site_rate
            self.G.add_edge(prev_comp, name, k=box_rate)
        for next_comp in self.G.successors(site):
            orig_rate = self.G.edges[(site, next_comp)]['k']
            # Now both the flow and volume are different...
            site_rate = (1 - blood_flow_fraction) / (1 - blood_volume_fraction) * orig_rate
            box_rate = blood_flow_fraction / blood_volume_fraction * orig_rate
            self.G.edges[(site, next_comp)]['k'] = site_rate
            self.G.add_edge(name, next_comp, k=box_rate)

        # the rates have changed so we have to update the rate_matrix and MTTs:
        self._graph_to_rate_matrix()
        self._get_mtts()

    def split_box_series_flip(self, name, box_dict):
        if name == 'flip_arterial':
            site = box_dict['flip_site_arterial']
            volume_fraction = box_dict['flip_volume_arterial']
            relative_blood_density = box_dict['relative_blood_density']
        if name == 'flip_venous':
            site = box_dict['flip_site_venous']
            volume_fraction = box_dict['flip_volume_venous']
            relative_blood_density = box_dict['relative_blood_density']

        blood_volume_fraction = volume_fraction * relative_blood_density
        blood_volume = (blood_volume_fraction * self.total_volume) / 100
        idx = self._add_new_box(name, site, blood_volume)

        # adjust site volume
        names = list(self.names)
        id_site = names.index(site)
        orig_volume = self.volumes[id_site]
        self.volumes[id_site] = orig_volume - blood_volume
        self.G.nodes[site]['V'] = self.volumes[id_site]
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)

        # adjust flow original site and added box:
        names = list(self.names)
        idx = names.index(site)
        orig_flow = self.flows[idx]
        self.flows[idx] = orig_flow
        self.flows = np.insert(self.flows, idx + 1, orig_flow)

        # adjust network connections:
        next_comp = list(self.G.successors(site))[0]

        # update the rate between the previous compartment and the original site
        # it remains the same because is in series...

        # add the new connection in series between the original site and the new compartment
        # rate_between = self.flows[idx + 1] / self.volumes[idx + 1]
        orig_rate_out = self.G.edges[(site, next_comp)]['k']
        rate_to_flip = self.flows[idx] / self.volumes[idx]
        self.G.add_edge(site, name, k=rate_to_flip)

        # now adjust the connection from the new compartment to the next one
        orig_rate_out = self.G.edges[(site, next_comp)]['k']
        rate_from_flip = self.flows[idx + 1] / self.volumes[idx + 1]
        self.G.add_edge(name, next_comp, k=rate_from_flip)

        # the edge that connects site->next_comp has to be removed because it has been added a serial box in the middle
        self.G.remove_edge(site, next_comp)

        # The Excel table is modified to add 'flip_venous':
        # Create a new row as a dictionary or a list
        new_row = pd.Series([0] * (len(self.df.columns) - 1) + [self.total_flow * 60], index=self.df.columns)
        # Insert the new row after 'left_heart' or 'inferior_vena_cava'
        index_position = self.df.index.get_loc(site) + 1  # Find the position after 'aorta' or 'inferior_vena_cava'
        # Split the DataFrame and insert the new row
        self.df = pd.concat(
            [self.df.iloc[:index_position], pd.DataFrame([new_row], index=[name]), self.df.iloc[index_position:]])
        # Create a new column with values (for example: all zeros)
        new_column = np.zeros(len(self.df), dtype=float)
        # Get the position of the 'left_heart' or 'inferior_vena_cava' column
        col_position = self.df.columns.get_loc(site) + 1  # Find the position after ‘left_heart’ or 'inferior_vena_cava'
        # Insert the new column at the desired position
        self.df.insert(col_position, name, new_column)
        # Modify the values in the 'inferior_vena_cava' row:
        orig_value_excel = self.df.loc[site, next_comp]
        self.df.loc[site, name] = orig_value_excel
        self.df.loc[site, next_comp] = 0
        self.df.loc[site, 'flow_sum'] = orig_value_excel
        self.df.loc[site, 'volume'] = np.round((self.volumes[idx] * 100) / self.total_volume,3)
        self.df.loc[site, 'MTT estimate'] = self.volumes[idx] / self.flows[idx]
        # Modify the values in the 'flip_venous' row:
        self.df.loc[name, next_comp] = orig_value_excel
        self.df.loc[name, 'flow_sum'] = orig_value_excel
        self.df.loc[name, 'volume'] = np.round((self.volumes[idx + 1] * 100) / self.total_volume,3)
        self.df.loc[name, 'MTT estimate'] = self.volumes[idx + 1] / self.flows[idx + 1]
        # Modify the values in the column:
        self.df.loc['sum', name] = orig_value_excel

        # get the rate matrix:
        self.k_matrix = None
        self.k_matrix = self._get_rate_matrix()
        # convert k_matrix to a graph:
        self._rate_matrix_to_graph()
        self._get_mtts()

    def add_box_parallel_and_series_flip(self, name, box_dict):
        if name == 'flip_arterial':
            site = box_dict['flip_site_arterial']
            volume_fraction = box_dict['flip_volume_arterial']
            relative_blood_density = box_dict['relative_blood_density']
        if name == 'flip_venous':
            site = box_dict['flip_site_venous']
            volume_fraction = box_dict['flip_volume_venous']
            relative_blood_density = box_dict['relative_blood_density']

        blood_volume_fraction = volume_fraction * relative_blood_density

        # add flip box after site
        names = list(self.names)
        id_site_orig = names.index(site)
        orig_volume = self.volumes[id_site_orig]
        orig_flow = self.flows[id_site_orig]
        blood_volume = (blood_volume_fraction * self.total_volume) / 100
        orig_volume_fraction = (orig_volume * 100) / self.total_volume
        self._add_new_box(name, site, blood_volume)

        blood_fraction_volume_sa = 12 / 100  # sa = superior_aorta
        blood_fraction_volume_ra = 88 / 100  # ra = rest of the aorta

        prev_comp = list(self.G.predecessors(site))[0]
        next_comp = list(self.G.successors(site))[0]
        name_parallel_box = 'superior_aorta'
        # add superior_aorta after prev_comp (which is 'left_heart') and before 'aorta' (which plays the role of rest of the aorta)
        id_prev_comp = self._add_new_box(name_parallel_box, prev_comp, orig_volume * blood_fraction_volume_sa)

        # adjust site volume
        names = list(self.names)
        id_site = names.index(site)
        self.volumes[id_site] = (orig_volume * blood_fraction_volume_ra) - blood_volume
        self.G.nodes[site]['V'] = self.volumes[id_site]
        self.cum_volume = np.cumsum(self.volumes) / np.sum(self.volumes)

        # adjust flow original site and added box:
        names = list(self.names)
        id_site = names.index(site)
        self.flows[id_site_orig] = orig_flow * blood_fraction_volume_ra
        self.flows = np.insert(self.flows, id_prev_comp + 1, orig_flow * blood_fraction_volume_sa)
        self.flows = np.insert(self.flows, id_site + 1, orig_flow * blood_fraction_volume_ra)

        # adjust network connections:
        orig_rate_in = self.G.edges[(prev_comp, site)]['k']
        rate_in_up = orig_rate_in * blood_fraction_volume_sa
        rate_in_down = orig_rate_in * blood_fraction_volume_ra

        # adjust network rates - in rates:
        self.G.edges[(prev_comp, site)]['k'] = rate_in_down
        self.G.add_edge(prev_comp, name_parallel_box, k=rate_in_up)
        rate_to_flip = self.flows[id_site] / self.volumes[id_site]
        self.G.add_edge(site, name, k=rate_to_flip)

        # adjust network rates - out rates:
        orig_rate_out = self.G.edges[(site, next_comp)]['k']
        rate_out_up = self.flows[id_prev_comp + 1] / self.volumes[id_prev_comp + 1]
        rate_out_down = self.flows[id_site + 1] / self.volumes[id_site + 1]
        self.G.add_edge(name_parallel_box, next_comp, k=rate_out_up)
        self.G.add_edge(name, next_comp, k=rate_out_down)

        # the edge that connects site->next_comp has to be removed because it has been added a serial box in the middle
        self.G.remove_edge(site, next_comp)

        # The Excel table is modified to add 'superior aorta':
        # Create a new row as a dictionary or a list
        new_row = pd.Series([0] * (len(self.df.columns) - 1) + [self.total_flow * 60], index=self.df.columns)
        # Insert the new row after 'left_heart'
        index_position = self.df.index.get_loc(prev_comp) + 1  # Find the position after 'left_heart'
        # Split the DataFrame and insert the new row
        self.df = pd.concat([self.df.iloc[:index_position], pd.DataFrame([new_row], index=[name_parallel_box]), self.df.iloc[index_position:]])
        # Create a new column with values (for example: all zeros)
        new_column = np.zeros(len(self.df), dtype=float)
        # Get the position of the ‘left_heart’ column
        col_position = self.df.columns.get_loc(prev_comp) + 1  # Find the position after 'left_heart'
        # Insert the new column at the desired position
        self.df.insert(col_position, name_parallel_box, new_column)
        # Changing the values in the row:
        self.df.loc[name_parallel_box, next_comp] = blood_fraction_volume_sa * 100
        self.df.loc[name_parallel_box, 'flow_sum'] = blood_fraction_volume_sa * 100
        self.df.loc[name_parallel_box, 'volume'] = np.round((self.volumes[id_prev_comp + 1] * 100) / self.total_volume, 3)
        self.df.loc[name_parallel_box, 'MTT estimate'] = self.volumes[id_prev_comp + 1] / self.flows[id_prev_comp + 1]
        # Changing the values in the column:
        self.df.loc[prev_comp, name_parallel_box] = blood_fraction_volume_sa * 100
        self.df.loc['sum', name_parallel_box] = blood_fraction_volume_sa * 100
        # Changing the values in the left_heart row:
        self.df.loc[prev_comp, site] = blood_fraction_volume_ra * 100

        # The Excel table is modified to add 'flip_arterial':
        # Create a new row as a dictionary or a list
        new_row = pd.Series([0] * (len(self.df.columns) - 1) + [6.5], index=self.df.columns)
        # Insert the new row after 'aorta'
        index_position = self.df.index.get_loc(site) + 1  # Find the position after 'aorta'
        # Split the DataFrame and insert the new row
        self.df = pd.concat([self.df.iloc[:index_position], pd.DataFrame([new_row], index=[name]), self.df.iloc[index_position:]])
        # Create a new column with values (for example: all zeros)
        new_column = np.zeros(len(self.df), dtype=float)
        # Get the position of the 'left_heart' column
        col_position = self.df.columns.get_loc(site) + 1  # Find the position after 'aorta'
        # Insert the new column at the desired position
        self.df.insert(col_position, name, new_column)
        # Modify the values of the row
        self.df.loc[self.names[id_site+1], next_comp] = blood_fraction_volume_ra * 100
        self.df.loc[self.names[id_site + 1], 'flow_sum'] = blood_fraction_volume_ra * 100
        self.df.loc[self.names[id_site + 1], 'volume'] = np.round((self.volumes[id_site + 1] * 100) / self.total_volume,3)
        self.df.loc[self.names[id_site + 1], 'MTT estimate'] = self.volumes[id_site + 1] / self.flows[id_site + 1]
        # Modify the values of the column
        self.df.loc['sum', name] = blood_fraction_volume_ra * 100

        # The Excel table for the 'aorta' is modified:
        # Change the values in the row
        self.df.loc[site, name] = blood_fraction_volume_ra * 100
        self.df.loc[site, next_comp] = 0
        self.df.loc[site, 'flow_sum'] = blood_fraction_volume_ra * 100
        self.df.loc[site, 'volume'] = np.round((self.volumes[id_site] * 100) / self.total_volume,3)
        self.df.loc[site, 'MTT estimate'] = self.volumes[id_site + 1] / self.flows[id_site + 1]
        # Modify the values in the column:
        self.df.loc['sum', site] = blood_fraction_volume_ra * 100
        # Modify the values in the 'large_arteries' column:
        self.df.loc['sum', next_comp] = 100

        # get the rate matrix (13nov2024):
        self.k_matrix = None
        self.k_matrix = self._get_rate_matrix()
        # convert k_matrix to a graph:
        self._rate_matrix_to_graph()
        self._get_mtts()

    def replace_box_with_flip(self, name, box_dict):
        if name == 'flip_arterial':
            site = box_dict['flip_site_arterial']
            volume_fraction = box_dict['flip_volume_arterial']
            relative_blood_density = box_dict['relative_blood_density']
        if name == 'flip_venous':
            site = box_dict['flip_site_venous']
            volume_fraction = box_dict['flip_volume_venous']
            relative_blood_density = box_dict['relative_blood_density']

        blood_volume_fraction = volume_fraction * relative_blood_density

        # find the position
        idx = self.names.index(site)
        orig_flow = self.flows[idx]
        # change box name
        orig_name = site
        self.names[idx] = name
        self.df.index.values[idx] = name
        # the flow remains the same:
        self.flows[idx] = orig_flow
        # update the volume of the patient-specific compartment:
        orig_volume = self.volumes[idx]
        self.volumes[idx] = (self.total_volume * blood_volume_fraction) / 100
        # update the volume of the previous patient-specific compartment:
            # this is done because the both volumes are related large_arteries/large_veins
            # with head_neck_arteries/head_neck_veins
        orig_total_volume_previous = orig_volume + self.volumes[idx - 1]
        new_volume_previous = orig_total_volume_previous - self.volumes[idx]
        self.volumes[idx-1] = np.round(new_volume_previous, 3)

        prev_comp = list(self.G.predecessors(orig_name))[0]
        orig_rate_in = self.G.edges[(prev_comp, orig_name)]['k']
        next_comp = list(self.G.successors(orig_name))[0]
        orig_rate_out = self.G.edges[(orig_name, next_comp)]['k']

        # change compartment original name (site) to flip name (name) in the graph
        self.G = nx.relabel_nodes(self.G, {site: name})

        # adjust network with rate in:
        # rate does not have to change, we're simply replacing the compartment.

        # adjust network with rate out from head-neck arteries or veins:
        list_edges_from_head_neck_a_v = list(self.G.edges(self.names[idx]))
        if len(list_edges_from_head_neck_a_v) == 1:
            flow_value = 1
            rate_from_head_neck_a_v = (self.flows[idx] / self.volumes[idx]) * flow_value
            self.G.edges[list_edges_from_head_neck_a_v[0]]['k'] = np.round(rate_from_head_neck_a_v, 3)
        else:
            for j in range(len(list_edges_from_head_neck_a_v)):
                flow_value = np.round(self.df.loc[list_edges_from_head_neck_a_v[j]] / 100, 3)
                rate_from_head_neck_a_v = (self.flows[idx] / self.volumes[idx]) * flow_value
                self.G.edges[list_edges_from_head_neck_a_v[j]]['k'] = np.round(rate_from_head_neck_a_v, 3)
        # adjust network with rate out from large arteries or veins:
        list_edges_from_large_a_v = list(self.G.edges(self.names[idx-1]))
        if len(list_edges_from_large_a_v) == 1:
            flow_value = 1
            rate_from_previous_large_a_v = (self.flows[idx-1] / self.volumes[idx-1]) * flow_value
            self.G.edges[list_edges_from_large_a_v[0]]['k'] = np.round(rate_from_previous_large_a_v, 3)
        else:
            for i in range(len(list_edges_from_large_a_v)):
                flow_value = np.round(self.df.loc[list_edges_from_large_a_v[i]] / 100, 3)
                rate_from_previous_large_a_v = (self.flows[idx-1] / self.volumes[idx-1]) * flow_value
                self.G.edges[list_edges_from_large_a_v[i]]['k'] = np.round(rate_from_previous_large_a_v, 3)

        self.df.loc[self.names[idx - 1], 'volume'] = np.round((self.volumes[idx - 1] * 100) / self.total_volume, 3)
        self.df.loc[self.names[idx], 'volume'] = np.round((self.volumes[idx] * 100) / self.total_volume, 3)
        self.df.loc[self.names[idx - 1], 'MTT estimate'] = self.volumes[idx-1]/self.flows[idx-1]
        self.df.loc[self.names[idx], 'MTT estimate'] = self.volumes[idx]/self.flows[idx]
        self.G.nodes[self.names[idx-1]]['V'] = self.volumes[idx-1]
        self.G.nodes[name]['V'] = self.volumes[idx]

        # get the rate matrix:
        self.k_matrix = None
        self.k_matrix = self._get_rate_matrix()
        # convert k_matrix to a graph:
        self._rate_matrix_to_graph()
        self._get_mtts()
