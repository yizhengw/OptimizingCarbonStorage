% you may need to change your python version in matlab
% into python 3 or above. To find this executable python path
% you may type the following commands in the terminal (only applied in macbook)
% conda env list
% conda activate ENVIRONMENT-NAME
% which python
% conda deactivate

% Run the next 2 lines within matlab when running it the first time
% pyenv('Version','/Users/markuszechner/opt/anaconda3/bin/python')
% clear classes

clear all;
close all;

CLUSTER = true;

if CLUSTER
    
    i = str2num(getenv('SLURM_ARRAY_TASK_ID'));
    
    fprintf('Running MRST simulation number %d ', i);
    
    
    case_name = 'eng_geo_no_global';
    
    SIM_DATA_PARAMS = ['/oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/' case_name '/paramters/'];
    SIM_DATA = ['/oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/' case_name '/simulations/'];
    
    %TODO: change to cluster path
    %TODO also save the structs with all the well and grid solutions - for
    %testing
    addpath('/home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab/');
    addpath('/home/users/mzechner/code-dev/MRST/mrst-2021a/modules/co2lab/');
    addpath('/home/users/mzechner/code-dev/MRST/mrst-2021a/');
    % TODO where is the custom start up file located - add path
    addpath('/home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/npy-matlab-master/npy-matlab');
else
    
    SIM_DATA_PARAMS = '/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab/SISL/CCS_reservoir_simulation_synthetic_map/sim_data/parameters/'
    SIM_DATA = '/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab/SISL/CCS_reservoir_simulation_synthetic_map/sim_data/'
    
    addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core/matlab');
    addpath('/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab');
    addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/npy-matlab-master/npy-matlab');
end


% Everytime matlab is started those 2 matlab files have to be run
% to load all the required matlab modules for the siumlation
run('startup.m')
run('startuplocal_custom.m')

% Importing the python functions that generate the geo maps
% IMPORTANT: when you change something in the python file it is
% necessary to restart matlab and reload the python functions!
% Make sure the pwd is the one where 'map_solver' is located
% mod = py.importlib.import_module('map_solver');

% geo function:
% input: 5 variables
% 1: total number of realizations
% 2: the realization index that you want to output (between 1 to total_num_realization)
% 3: number of row (x) of the map
% 4: number of col (y) of the map
% 5: number of layers (z) of the map

% Loading maps in case we do NOT use python to generate the geology
% this file is on our google drive
% Load maps and get the one for this particular simulation
%geo_maps = readNPY([SIM_DATA_PARAMS 'mapcollections.npy']);
geo_maps = load([SIM_DATA_PARAMS 'maps_collection.mat']);
GEO_MAPS = geo_maps.maps;

% Load Parameters
load([SIM_DATA_PARAMS 'PriorModelParameters.mat'],'PriorModelParameters')


numb_sim = 5000; % number of geo maps
NL=80;
NW=80;
ND=8;


% mean_poro = zeros(numb_sim,3);
%
% box_lenth=3; % to calculate mean poro around well

%% Load Prior Sampling
% the sampling is done in a different matlab script so that we dont
% overwrite the parameters each time we run the simulation master script
% Be careful when running the sampling so that it doesn't overwrite the
% paramters we used to run the simulations
% 'numb_sim' in this file has to be smaller or equal the one in the
% sample script

% Load Parameters
load([SIM_DATA_PARAMS 'PriorModelParameters.mat'],'PriorModelParameters')


%% Running the simulations
tic
% 1: usage of python script to generate geology
% We set a seed in the python function so that it will generate the
% same models even though it is called multiple times
% res_map_python = py.map_solver.main(numb_sim,i-1,100,100,10);
% p = double(res_map_python);

% 2: Read geo maps generated in SGEMS

P = squeeze(GEO_MAPS(i,:,:,:));
K = P.^3.*(1e-5)^2./(0.81*72*(1-P).^2);


% poro mean around the well is only used for sensitivity analysis
% we can extract it later from the saved geo model as well
%     mean_well1 = get_mean_poro_well(p, PriorModelParameters.('x_injector_1')(i), PriorModelParameters.('y_injector_1')(i), box_lenth);
%     mean_well2 = get_mean_poro_well(p, PriorModelParameters.('x_injector_2')(i), PriorModelParameters.('y_injector_2')(i), box_lenth);
%     mean_well3 = get_mean_poro_well(p, PriorModelParameters.('x_injector_3')(i), PriorModelParameters.('y_injector_3')(i), box_lenth);
%     mean_poro(i,:) = [mean_well1 mean_well2 mean_well3];


% run_reservoir_sim returns the bottom hole pressures of all
% three injectors
run_reservoir_sim_with_eng_params_new_schedule(P,K,NL,NW,ND, ...
    PriorModelParameters,i,SIM_DATA);


T = toc;
fprintf('The MRST simulations took %8.2f seconds.n', T);


