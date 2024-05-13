
%% handling I/O from Julia 

% Important: last well in W_COORDS is the observation well!
% inj_rate has to be set to 0 for that well!

% check number of wells
[~, n_wells] = size(W_COORDS);

% Injection Rates
% rate is later an input
rate = [0.5;...
        0.5;...
        0.5;...
        0];

inj_rate = zeros(n_wells,1);

for i = 1:n_wells
    inj_rate(i) = rate(i) * mega * 1e3 / year / co2_rho;
end

% Reservoir Model Paramters that are part of the belief
c_rock  = PARAMETERS(1) / barsa;
srw     = PARAMETERS(2); % residual water
src     = PARAMETERS(3); % residual CO2
pe      = PARAMETERS(4) * kilo * Pascal; % capillary entry pressure
muw     = PARAMETERS(5) * Pascal * second; % brine viscosity

% grid properties
P=POROSITY;
K=PERMEABILITY;

% check schedule varibles

assert(TOTAL_TIME>INJECTION_STOP,'Injection stop needs to be smaller than total time!')

%% Make geo model
% takes porosity and perm map and dims as input - structur is 
% computed and returned

[G, Gt, rock, rock2D, bcIxVE] = makeSlopingAquifer_custom_JULIA(P,K,NL, NW, ND)


% Build Well struct
W = [];
perf_interval=1:8;

for i = 1:n_wells
    x = W_COORDS(1, i);
    y = W_COORDS(2, i);
    wc_global = false(G.cartDims);
    wc_global(x, y, perf_interval) = true;
    wc = find(wc_global(G.cells.indexMap));
    
    W = addWell(W, G, rock, wc, ...
    'type', 'rate', ...  % inject at constant rate
    'val', inj_rate(i), ... % volumetric injection rate
    'comp_i', [0 1],...
    'Name', ['Well ' num2str(i)]);    % inject CO2, not water
    
end



%% Grid and rock for VE simulation

% Make top surface grid used for VE simulation
[Gt, G, transMult] = topSurfaceGrid(G);

% Computing vertically averaged rock object
rock2D = averageRock(rock, Gt);

%% Initial state

% Gt.cells.z gives the caprock depth of each cell in the 2D grid.
initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Fluid model

invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D = @(s) max((s-src)./(1-src), 0).^2; % uses CO2 saturation

% Use full EOS rather than compressibilities and include capillary fringe model
% based on upscaled sampled capillary pressure.
% See makeVEFluid.m for a description of various fluid models which are
% available.
% A description of the different models can be found in the paper
% "Fully-Implicit Simulation of Vertical-Equilibrium Models with Hysteresis and
% Capillary Fringe" (Nilsen et al., Computational Geosciences 20, 2016).

fluid = makeVEFluid(Gt, rock, 'P-scaled table'             , ...
                    'co2_mu_ref'  , muco2, ...%6e-5 * Pascal * second , ...
                    'wat_mu_ref'  , muw, ...%8e-4 * Pascal * second , ...
                    'co2_rho_ref' , co2_rho                , ...
                    'wat_rho_ref' , rhow                   , ...
                    'co2_rho_pvt' , [co2_c, p_ref]         , ...
                    'wat_rho_pvt' , [wat_c, p_ref]         , ...
                    'residual'    , [srw, src]             , ...
                    'pvMult_p_ref', p_ref                  , ...
                    'pvMult_fac'  , c_rock                 , ...
                    'invPc3D'     , invPc3D                , ...
                    'kr3D'        , kr3D                   , ...
                    'transMult'   , transMult);

%% Bounday Conditions

% hydrostatic pressure conditions for open boundary faces
p_bc = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D = addBC([], bcIxVE, 'pressure', p_bc);
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

%% CCS schedule

W2D  = convertwellsVE(W, G, Gt, rock2D);

schedule = createWellSchedule_CCS(TOTAL_TIME, INJECTION_STOP, NSTEP_INJECTION, ... 
    NSTEP_SHUT_IN, W2D, INTERVAL, 'bc', bc2D);



%% REservoir Model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);

%% Simulate
[wellSol, states] = simulateScheduleAD(initState, model, schedule);


%% Process Results

% 
injector_bhps = zeros(length(wellSol), length(wellSol{1}));
observation_sat = zeros(length(wellSol), 1);

obs_well_index = sub2ind([NL,NW],[W_COORDS(1, end)],[W_COORDS(2, end)]);


for time_step = 1:length(wellSol)

    injector_bhps(time_step,:) = cell2mat({wellSol{time_step}.bhp});

    observation_sat(time_step,:) = cell2mat({states{time_step}.s(obs_well_index,2)})

end



%% Extracting Data

sat_map_MATLAB = zeros(size(states,1),NL,NW,ND);
sat_map_all_layers_reconstructed = zeros(size(states,1),NL*NW*ND);
sat_map_all_first_layer = zeros(size(states,1),NL*NW*1);

pressure_map_MATLAB = zeros(size(states,1),NL,NW,1);
pressure_map_first_layer = zeros(size(states,1),NL*NW*1);

% % generating a new folder for each simulation:
%
% SAVE_PATH_SIM = [SIM_DATA 'sim_' num2str(i) '/'];
% if not(isfolder(SAVE_PATH_SIM))
%     mkdir(SAVE_PATH_SIM)
% end

for k=1:size(states,1)

    % Saturations
    [h, h_max] = upscaledSat2height(states{k}.s(:,2), states{k}.sGmax, Gt, ...
        'pcWG', fluid.pcWG, ...
        'rhoW', fluid.rhoW, ...
        'rhoG', fluid.rhoG, ...
        'p', states{end}.pressure);


    sat_map = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);

    sat_map_reshaped=reshape(sat_map,NL,NW,ND);

    sat_map_MATLAB(k,:,:,:)= sat_map_reshaped;

    sat_map_all_layers_reconstructed(k,:)= sat_map;

    sat_map_all_first_layer(k,:)= reshape(sat_map_reshaped(:,:,1), 1,NL*NW);


    % Pressure
    pressure_map = states{k}.pressure;

    pressure_map_reshaped=reshape(pressure_map,NL,NW,1);

    pressure_map_MATLAB(k,:,:,:)= pressure_map_reshaped;
    pressure_map_first_layer(k,:)= pressure_map;

end

% Running the trap analysis to get the trapped masses of CO2
ta = trapAnalysis(Gt, false);
reports = makeReports(Gt, {initState states{:}}, model.rock, model.fluid, ...
                      schedule, [srw, src], ta, []);

%mass_hist = cell2mat({reports.masses}.');

mass_hist = reshape([reports.masses]',8, [])';

% names = {'Dissolved'           , 'Structural residual' , ...
%     'Residual'            , 'Residual in plume'  , ...
%     'Structural subscale' , 'Structural plume'   , ...
%     'Free plume'          , 'Exited'};

% Permuting volumes to comply with the ordering we want for reporting
P = [1 2 3 4 6 5 7 8];
mass_fractions = mass_hist(:,P);

time_days = cell2mat({reports.t}.');

% returning the schedule indices for splitted rewards:
schedule_idx = schedule.step.control;


%% saving all variables

% % Saturations
% save([SAVE_PATH_SIM 'sat_map_MATLAB.mat'],'sat_map_MATLAB');
% writeNPY(sat_map_all_layers_reconstructed, [SAVE_PATH_SIM 'sat_map_all_layers_reconstructed.npy']);
% writeNPY(sat_map_all_first_layer, [SAVE_PATH_SIM 'sat_map_all_first_layer.npy']);
%
% % Pressure
% save([SAVE_PATH_SIM 'pressure_map_MATLAB.mat'],'pressure_map_MATLAB');
% writeNPY(pressure_map_first_layer, [SAVE_PATH_SIM 'pressure_map_first_layer.npy']);
%
% % Trapped Masses
% writeNPY(mass_fractions, [SAVE_PATH_SIM 'mass_fractions.npy']);
%
% % Time vector in days
% writeNPY(time_days, [SAVE_PATH_SIM 'time_days.npy']);
%
% % Bottom hole pressures of all 3 injector wells
% % using well solutions NOT block pressure
% writeNPY(injector_bhps, [SAVE_PATH_SIM 'injector_bhps.npy']);
%
% % Sautrations versus time of the Observations well
% % using the block saturations
% writeNPY(observation_sat, [SAVE_PATH_SIM 'observation_sat.npy']);
