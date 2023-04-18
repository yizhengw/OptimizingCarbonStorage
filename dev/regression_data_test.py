import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
import pickle
# import pandas as pd
#
# from sklearn.model_selection import train_test_split, GridSearchCV

DATA_PATH = '/home/jmern91/storage/CCS/data/data_regression_temp/'
# DATA_PATH = '/home/jmern/Storage/CCS/data/data_regression_temp/'
case_name = 'eng_geo_no_global_rate02' # case name 'eng_geo_no_global_rate02' injects 0.01 Mega tons/well/year

# these are the indices for the current (09/20/2021) MRST simulation. TODO: need to make that loadable
schedule_idx = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
numb_sim = 1000 # number of simulations

def save_pickle(file_name, obj):
    open_file = open(file_name, "wb")
    pickle.dump(obj, open_file)
    open_file.close()

def load_pickle(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


def reward(mass_fractions_selection, injector_bhps_selection, pressure_map_first_layer_selection, schedule_idx):
    unique_elements = np.unique(schedule_idx)
    reward_pressure = []
    reward_exited = []
    reward_free = []
    reward_trapped = []
    reward_initial_pressure = []

    # mass_fractions_selection = mass_fractions_selection[1:,:]

    for i in unique_elements:
        mass_fractions_temp = mass_fractions_selection[schedule_idx <= i, :]
        injector_bhps_temp = injector_bhps_selection[schedule_idx <= i, :]
        pressure_map_first_layer_temp = pressure_map_first_layer_selection[schedule_idx <= i, :]

        existed_vol = mass_fractions_temp[:, 7][-1] / np.mean(mass_fractions_temp[:, 7])
        free_plume_vol = mass_fractions_temp[:, 6][-1] / np.mean(mass_fractions_temp[:, 6])
        trapped_vol = np.sum(mass_fractions_temp[:, 0:5], axis=1)[-1] / np.mean(
            np.sum(mass_fractions_temp[:, 0:5], axis=1))
        pressure = np.maximum(np.max(injector_bhps_temp), np.max(pressure_map_first_layer_temp))
        initial_pressure = np.max(pressure_map_first_layer_temp[1, :])

        reward_pressure.append(pressure)
        reward_initial_pressure.append(initial_pressure)
        reward_exited.append(existed_vol)
        reward_free.append(free_plume_vol)
        reward_trapped.append(trapped_vol)

    return [reward_pressure, reward_initial_pressure, reward_exited, reward_free, reward_trapped]

# ################### I think this loads from the Sherlock server and isn't needed  ####################################
#
# exited_vol_collection_mean = []
# exited_vol_collection_max = []
#
# mean_bhp_inj_collection = []
#
# max_bhp_inj_collection = []
#
# mean_sat_inj_collection = []
#
# max_sat_inj_collection = []
#
# reward_collection = []
# files_not_run = []
# perm_observations = []
#
# # load maps
# parameters = np.load(SIM_PATH + case_name + '/paramters/Parameters.npy')  # load all of them
# geomaps = SIM_PATH + case_name + '/paramters/maps_collection.mat'
# maps_collection_array = sio.loadmat(geomaps)["maps"]
#
# well_xs = np.array(parameters[5:, :][np.arange(0, 8, 2), :], dtype="int")
# well_ys = np.array(parameters[5:, :][np.arange(1, 8, 2), :], dtype="int")
# well_locs = np.array([well_xs, well_ys]).T  # 1000 by 4 by 2
#
# print(well_locs.shape)
#
# unique_elements = np.unique(schedule_idx)
#
# for i in range(numb_sim):
#
#     SIM_FOLDER = SIM_PATH + case_name + '/simulations/sim_' + str(i + 1) + '/'
#
#     # loading variables
#     try:
#         mass_fractions = np.load(SIM_FOLDER + 'mass_fractions.npy')
#         mass_fractions = mass_fractions[1:, :]  # take out the first 0 (added for plotting)
#         injector_bhps = np.load(SIM_FOLDER + 'injector_bhps.npy')
#         pressure_map_first_layer = np.load(SIM_FOLDER + 'pressure_map_first_layer.npy')
#
#         obs_well_sat = np.load(SIM_FOLDER + 'observation_sat.npy')
#
#         # compute reward
#         reward_collection.append(
#             np.array(reward(mass_fractions, injector_bhps, pressure_map_first_layer, schedule_idx)))
#
#         perm_observations.append(
#             [maps_collection_array[i, ele[0] - 1, ele[1] - 1, 0] for ele in well_locs[i, :, :]])  # 4 by 2)
#
#         exited_vol_collection_mean_temp = []
#         exited_vol_collection_max_temp = []
#
#         mean_bhp_inj_collection_temp = []
#
#         max_bhp_inj_collection_temp = []
#
#         mean_sat_inj_collection_temp = []
#
#         max_sat_inj_collection_temp = []
#
#         for k in unique_elements:
#             mass_fractions_temp = mass_fractions[schedule_idx <= k, :]
#
#             mass_fractions_temp = mass_fractions[schedule_idx <= k, :]
#             injector_bhps_temp = injector_bhps[schedule_idx <= k, :]
#             pressure_map_first_layer_temp = pressure_map_first_layer[schedule_idx <= k, :]
#             obs_well_sat_temp = obs_well_sat[schedule_idx <= k, :]
#
#             exited_vol_collection_mean_temp.append(mass_fractions_temp[:, 7][-1] / np.mean(mass_fractions_temp[:, 7]))
#             exited_vol_collection_max_temp.append(mass_fractions_temp[:, 7][-1])
#
#             # compute reward
#             # reward_collection.append(np.array(reward(mass_fractions_temp, injector_bhps_temp, pressure_map_first_layer_temp,schedule_idx)))
#
#             # get data for JYM surrogate model
#
#             mean_bhp_inj = np.mean(injector_bhps_temp, axis=0)
#             mean_bhp_inj_collection_temp.append(mean_bhp_inj)
#
#             max_bhp_inj = np.max(injector_bhps_temp, axis=0)
#             max_bhp_inj_collection_temp.append(max_bhp_inj)
#
#             mean_sat_inj = np.mean(obs_well_sat_temp, axis=0)
#             mean_sat_inj_collection_temp.append(mean_sat_inj)
#
#             max_sat_inj = np.max(obs_well_sat_temp, axis=0)
#             max_sat_inj_collection_temp.append(max_sat_inj)
#
#         mean_bhp_inj_collection.append(np.asarray(mean_bhp_inj_collection_temp))
#         max_bhp_inj_collection.append(np.asarray(max_bhp_inj_collection_temp))
#
#         mean_sat_inj_collection.append(np.asarray(mean_sat_inj_collection_temp))
#         max_sat_inj_collection.append(np.asarray(max_sat_inj_collection_temp))
#
#         exited_vol_collection_mean.append(np.asarray(exited_vol_collection_mean_temp))
#         exited_vol_collection_max.append(np.asarray(exited_vol_collection_max_temp))
#
#
#
#
#     except IOError:
#
#         print("Could not read MRST simulation file nr " + str(i + 1))
#         files_not_run.append(i)
#
# perm_observations = np.asarray(perm_observations)
#
# #################################################################################################################
# in general: all lists have a length equal to the number of simulations - each element of the list has 4 values,
# equivalent to the 4 periods of the MRST simulation

# put 'False' only after reading data direclty from the Oak drive
loading = True

# paramter names
parameter_names = ['c_rock', 'srw', 'src', 'pe', 'muw', 'x_injector_1', 'y_injector_1', 'x_injector_2', 'y_injector_2',
                   'x_injector_3', 'y_injector_3', 'x_observation_1', 'y_observation_1']

if loading:  # loading variables

    index_good_runs = np.load(DATA_PATH + 'index_good_runs_' + case_name + '.npy')  # index of MRST simulations that finished.

    exited_vol_collection_mean = load_pickle(DATA_PATH + 'exited_vol_collection_mean_' + case_name + '.pkl')
    exited_vol_collection_max = load_pickle(DATA_PATH + 'exited_vol_collection_max_' + case_name + '.pkl')

    reward_collection = load_pickle(DATA_PATH + 'reward_collection_' + case_name + '.pkl')

    mean_bhp_inj_collection = load_pickle(DATA_PATH + 'mean_bhp_inj_collection_' + case_name + '.pkl')
    max_bhp_inj_collection = load_pickle(DATA_PATH + 'max_bhp_inj_collection_' + case_name + '.pkl')

    mean_sat_inj_collection = load_pickle(DATA_PATH + 'mean_sat_inj_collection_' + case_name + '.pkl')
    max_sat_inj_collection = load_pickle(DATA_PATH + 'max_sat_inj_collection_' + case_name + '.pkl')

    # perm found at well location 1:n - observation well is the last
    perm_observations = np.load(DATA_PATH + 'perm_observations_' + case_name + '.npy')

    # paramters: (number of run simulations x number of parameters)
    parameters_cleaned = np.load(DATA_PATH + 'paramters_cleaned_' + case_name + '.npy')
    # parameters = np.load(DATA_PATH + 'parameters/Parameters.npy')  # load all of them
    # parameters = parameters.T
    # parameters = parameters[0:numb_sim, :]  # get the ones where we have simulations
    # parameters_cleaned = parameters[index_good_runs, :]  # clean them

else:  # saving variables

    index_all_sim = np.arange(numb_sim)
    index_good_runs = np.delete(index_all_sim, files_not_run)
    np.save('index_good_runs_' + case_name + '.npy', index_good_runs)

    save_pickle('exited_vol_collection_mean_' + case_name + '.pkl', exited_vol_collection_mean)
    save_pickle('exited_vol_collection_max_' + case_name + '.pkl', exited_vol_collection_max)

    save_pickle('reward_collection_' + case_name + '.pkl', reward_collection)

    save_pickle('mean_bhp_inj_collection_' + case_name + '.pkl', mean_bhp_inj_collection)
    save_pickle('max_bhp_inj_collection_' + case_name + '.pkl', max_bhp_inj_collection)

    save_pickle('mean_sat_inj_collection_' + case_name + '.pkl', mean_sat_inj_collection)
    save_pickle('max_sat_inj_collection_' + case_name + '.pkl', max_sat_inj_collection)

    np.save('perm_observations_' + case_name + '.npy', perm_observations)

    # parameters

    parameters = np.load(SIM_PATH + case_name + '/paramters/Parameters.npy')  # load all of them
    parameters = parameters.T
    parameters = parameters[0:numb_sim, :]  # get the ones where we have simulations
    paramters_cleaned = parameters[index_good_runs, :]  # clean them
    np.save('paramters_cleaned_' + case_name + '.npy', paramters_cleaned)

import pdb; pdb.set_trace()
print()