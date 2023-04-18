import numpy as np
import pickle

def load_pickle(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

def load_data(data_path, case_name):
    parameter_names = ['c_rock', 'srw', 'src', 'pe', 'muw', 'x_injector_1', 'y_injector_1', 'x_injector_2', 'y_injector_2',
                   'x_injector_3', 'y_injector_3', 'x_observation_1', 'y_observation_1']

    index_good_runs = np.load(data_path + 'index_good_runs_' + case_name + '.npy')  # index of MRST simulations that finished.

    exited_vol_collection_mean = load_pickle(data_path + 'exited_vol_collection_mean_' + case_name + '.pkl')
    exited_vol_collection_max = load_pickle(data_path + 'exited_vol_collection_max_' + case_name + '.pkl')

    reward_collection = load_pickle(data_path + 'reward_collection_' + case_name + '.pkl')

    mean_bhp_inj_collection = load_pickle(data_path + 'mean_bhp_inj_collection_' + case_name + '.pkl')
    max_bhp_inj_collection = load_pickle(data_path + 'max_bhp_inj_collection_' + case_name + '.pkl')

    mean_sat_inj_collection = load_pickle(data_path + 'mean_sat_inj_collection_' + case_name + '.pkl')
    max_sat_inj_collection = load_pickle(data_path + 'max_sat_inj_collection_' + case_name + '.pkl')

    # perm found at well location 1:n - observation well is the last
    perm_observations = np.load(data_path + 'perm_observations_' + case_name + '.npy')

    # paramters: (number of run simulations x number of parameters)
    parameters_cleaned = np.load(data_path + 'paramters_cleaned_' + case_name + '.npy')

    reward_collection_arr = np.array(reward_collection)
    index_no_nan = np.argwhere(np.isnan((reward_collection_arr)))

    index_good_runs = np.delete(index_good_runs, index_no_nan)

    exited_vol_collection_mean = np.delete(np.array(exited_vol_collection_mean), index_no_nan, 0)
    exited_vol_collection_max = np.delete(np.array(exited_vol_collection_max), index_no_nan, 0)
    reward_collection = np.delete(np.array(reward_collection), index_no_nan, 0)
    mean_bhp_inj_collection = np.delete(np.array(mean_bhp_inj_collection), index_no_nan, 0)
    max_bhp_inj_collection = np.delete(np.array(max_bhp_inj_collection), index_no_nan, 0)
    mean_sat_inj_collection = np.delete(np.array(mean_sat_inj_collection), index_no_nan, 0)
    max_sat_inj_collection = np.delete(np.array(max_sat_inj_collection), index_no_nan, 0)
    perm_observations = np.delete(np.array(perm_observations), index_no_nan, 0)
    parameters_cleaned = np.delete(np.array(parameters_cleaned), index_no_nan, 0)

    results = {
                "index_good_runs":index_good_runs,
                "exited_vol_collection_mean":exited_vol_collection_mean,
                "exited_vol_collection_max":exited_vol_collection_max,
                "reward_collection":reward_collection,
                "mean_bhp_inj_collection":mean_bhp_inj_collection,
                "max_bhp_inj_collection":max_bhp_inj_collection,
                "mean_sat_inj_collection":mean_sat_inj_collection,
                "max_sat_inj_collection":max_sat_inj_collection,
                "perm_observations":perm_observations,
                "parameters_cleaned":parameters_cleaned,
                "parameter_names":parameter_names
    }
    return results

if __name__ == "__main__":
    # data_path = '/home/jmern91/storage/CCS/data/data_regression_temp/'
    data_path = '/home/jmern/Storage/CCS/data/data_regression_temp/'
    case_name = 'eng_geo_no_global_rate02'  # case name 'eng_geo_no_global_rate02' injects 0.01 Mega tons/well/year
    results = load_data(data_path, case_name)
    import pdb; pdb.set_trace()
    print()