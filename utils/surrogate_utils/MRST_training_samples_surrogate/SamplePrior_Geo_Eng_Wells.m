%% Sampling the prior for eng paramters

clear all;
close all;

% Number of Simulations
numb_sim = 10000;


CLUSTER = true;

if CLUSTER
    case_name = 'eng_geo_no_global';
    SIM_DATA_PARAMS = ['/oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/' case_name '/paramters/'];
    addpath('/home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab');
    addpath('/home/users/mzechner/code-dev/CCS_dev/CCS-core');
    addpath('/home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/npy-matlab-master/npy-matlab');
else
    SIM_DATA_PARAMS = '/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab/SISL/CCS_reservoir_simulation_synthetic_map/sim_data/parameters/'
    addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core/matlab');
    addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core');
    addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/npy-matlab-master/npy-matlab');
end



Normal = 0;
Uniform = 1;
UniformCategorical=2;

PriorParameterDistribution = struct();

% rock compressibility
PriorParameterDistribution.('c_rock') = [2e-5 8e-5 Uniform];

% residual water
PriorParameterDistribution.('srw') = [0.15 0.35 Uniform];

% residual CO2
PriorParameterDistribution.('src') = [0.15 0.35 Uniform];

% capillary entry pressure
PriorParameterDistribution.('pe') = [2 20 Uniform];

% brine viscosity
PriorParameterDistribution.('muw') = [6e-4 9e-4 Uniform];

% Well Coordinates

% Injector 1
PriorParameterDistribution.('x_injector_1') = [1 80 UniformCategorical];
PriorParameterDistribution.('y_injector_1') = [1 80 UniformCategorical];

% injector 2
PriorParameterDistribution.('x_injector_2') = [1 80 UniformCategorical];
PriorParameterDistribution.('y_injector_2') = [1 80 UniformCategorical];

% injector 3
PriorParameterDistribution.('x_injector_3') = [1 80 UniformCategorical];
PriorParameterDistribution.('y_injector_3') = [1 80 UniformCategorical];

% Observation Well 1
PriorParameterDistribution.('x_observation_1') = [1 80 UniformCategorical];
PriorParameterDistribution.('y_observation_1') = [1 80 UniformCategorical];

%% Generate prior models by sampling from prior parameters
ParameterNames = fieldnames(PriorParameterDistribution);
PriorModelParameters = struct();


% Iterate over each uncertain reservoir parameter
for i = 1:numel(ParameterNames)
    ParameterName = ParameterNames{i};
    ParameterRange = PriorParameterDistribution.(ParameterName);

    % Sample from uniform/normal distributions
    if (ParameterRange(3) == Normal)
        Value = ParameterRange(1) + ParameterRange(2)*randn(numb_sim,1);
    elseif(ParameterRange(3) == Uniform)
        Value = ParameterRange(1) + rand(numb_sim,1)*...
            (ParameterRange(2) - ParameterRange(1));

    elseif(ParameterRange(3) == UniformCategorical)

        Value = randsample(ParameterRange(1):ParameterRange(2),numb_sim, true);
    end

    % Store sampled value
    PriorModelParameters.(ParameterName) = Value;

end

% Save Parameters
if not(isfolder(SIM_DATA_PARAMS))
    mkdir(SIM_DATA_PARAMS)
end

save([SIM_DATA_PARAMS 'PriorModelParameters.mat'],'PriorModelParameters')

% save as npy file
Param_temp = struct2cell(PriorModelParameters);

Parameters = zeros(length(fieldnames(PriorModelParameters)),numb_sim);
for i=1:size(Param_temp)

    Parameters(i,:) = Param_temp{i};

end

writeNPY(Parameters, [SIM_DATA_PARAMS 'Parameters.npy']);
