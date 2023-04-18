
%function [trapping_fractions(2), max_pressure] = runJohansenVE(x_pos, y_pos)

%% Basic VE example
% This example shows how to setup a basic VE simulation in mrst. We use the
% Johansen formation and create a VE grid from a 3D mrst grid. We demonstrate the
% use of an EOS and a model with a capillary fringe. We also analyse the
% evolution of trapping mechanisms throughout the simulation.
%
% This example is described in more detail in section 3.1 of
% "Simplified models for numerical simulation of geological CO2 storage"
% (O. Andersen, 2017)
%
% For comparison example3D.m shows the fully 3D version of this model.

%% Load modules

clear all; close all;

mrstModule add ad-core;
mrstModule add ad-props;


%% Setup model parameters

gravity on;
g       = gravity;
rhow    = 1000;
co2     = CO2props();
p_ref   = 30 *mega*Pascal;
t_ref   = 94+273.15;
co2_rho = co2.rho(p_ref, t_ref);
co2_c   = co2.rhoDP(p_ref, t_ref) / co2_rho;
wat_c   = 0; 
c_rock  = 4.35e-5 / barsa;
srw     = 0.27; % residual water
src     = 0.20; % residual CO2
pe      = 5 * kilo * Pascal; % capillary entry pressure
muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

%% Injection Rates

rate = [0.5;...
        0.5;...
        0.5;...
        0];
    
NL=80;
NW=80;
ND=8;

W_COORDS=[1 10 20 30;1 10 20 30];

[~, n_wells] = size(W_COORDS);

inj_rate = zeros(n_wells,1);

for i = 1:n_wells
    inj_rate(i) = rate(i) * mega * 1e3 / year / co2_rho;
end

%% Model building

% Load Johansen model
%[G, rock, bcIx, ~, ~, bcIxVE] = makeJohansenVEgrid();

[G, Gt, rock, rock2D, bcIxVE, pp] = makeSlopingAquifer_custom3()
%[G, Gt, rock, rock2D, bcIx] = makeSlopingAquifer()

% % Specify well information
% wc_global1 = false(G.cartDims);
% wc_global1(x_pos_1, y_pos_1, 1:8) = true;
% wc1 = find(wc_global1(G.cells.indexMap));
% %wc = [3715, 10210, 16022, 21396, 26770]';
% 
% %3.5
% 
% 
% 
% W = addWell([], G, rock, wc1, ...
%             'type', 'rate', ...  % inject at constant rate
%             'val', inj_rate_1, ... % volumetric injection rate
%             'comp_i', [0 1]);    % inject CO2, not water
% 
% 
% wc_global2 = false(G.cartDims);
% wc_global2(x_pos_2, y_pos_2, 1:8) = true;
% wc2 = find(wc_global2(G.cells.indexMap));
% W = addWell(W, G, rock, wc2, ...
%     'type', 'rate', ...  % inject at constant rate
%     'val', inj_rate_2, ... % volumetric injection rate
%     'comp_i', [0 1]);    % inject CO2, not water
% 
% wc_global3 = false(G.cartDims);
% wc_global3(x_pos_3, y_pos_3, 1:8) = true;
% wc3 = find(wc_global3(G.cells.indexMap));
% W = addWell(W, G, rock, wc3, ...
%     'type', 'rate', ...  % inject at constant rate
%     'val', inj_rate_3, ... % volumetric injection rate
%     'comp_i', [0 1]);    % inject CO2, not water
% 
% wc_global4 = false(G.cartDims);
% wc_global4(x_pos_4, y_pos_4, 1:8) = true;
% wc4 = find(wc_global4(G.cells.indexMap));
% W = addWell(W, G, rock, wc4, ...
%     'type', 'rate', ...  % inject at constant rate
%     'val', inj_rate_4, ... % volumetric injection rate
%     'comp_i', [0 1]);    % inject CO2, not water

%% test for schedule

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
        
%% Grid and rock
% Make top surface grid used for VE simulation
[Gt, G, transMult] = topSurfaceGrid(G);

% % Shift G 100 meters down, and plot both grids for comparison
% GG = G;
% GG.nodes.coords(:,3) = GG.nodes.coords(:,3) + 100;
% figure; 
% plotGrid(GG, 'facecolor', 'green'); 
% plotGrid(Gt, 'facecolor', 'red');
% view(-65,33);
% set(gcf, 'position', [531 337 923 356]); axis tight;
% 
% 
% % % Let us plot the well on the grid to check that we got the position right.
% plotCellData(Gt, Gt.cells.z, 'edgealpha', 0.4);
% plotWell(Gt.parent, W, 'color', 'k');
% set(gcf, 'position', [10, 10, 800, 600]); view(76, 66);
% 

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

%% Set up schedule
W2D  = convertwellsVE(W, G, Gt, rock2D);

% hydrostatic pressure conditions for open boundary faces
p_bc = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D = addBC([], bcIxVE, 'pressure', p_bc); 
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);


%% CCS schedule

TOTAL_TIME = 530;
INJECTION_STOP=30;
assert(TOTAL_TIME>INJECTION_STOP,'Injection stop needs to be smaller than total time!')

NSTEP_INJECTION = 30; 
NSTEP_SHUT_IN = 20;
INTERVAL = 5;

schedule = createWellSchedule_CCS(TOTAL_TIME, INJECTION_STOP, NSTEP_INJECTION, ... 
    NSTEP_SHUT_IN, W2D, INTERVAL, 'bc', bc2D);






%% SCHEDULE

% Setting up two copies of the well and boundary specifications. 
% Modifying the well in the second copy to have a zero flow rate.
% schedule.control    = struct('W', W2D, 'bc', bc2D);
% schedule.control(2) = struct('W', W2D, 'bc', bc2D);
% schedule.control(3) = struct('W', W2D, 'bc', bc2D);
% schedule.control(4) = struct('W', W2D, 'bc', bc2D);
% 
% % 1st control: only 1 well
% schedule.control(1).W(2).val = inj_rate(1);
% schedule.control(1).W(2).val = 0;
% schedule.control(1).W(3).val = 0;
% 
% % 2nd control: 2 wells
% schedule.control(2).W(2).val = inj_rate(1);
% schedule.control(2).W(2).val = inj_rate(2);
% schedule.control(2).W(3).val = 0;
% 
% % 3nd control: 3 wells
% schedule.control(3).W(1).val = inj_rate(1);
% schedule.control(3).W(2).val = inj_rate(2);
% schedule.control(4).W(3).val = inj_rate(3);
% 
% % 4th control: all wells shut
% schedule.control(4).W(1).val = 0;
% schedule.control(4).W(2).val = 0;
% schedule.control(4).W(3).val = 0;
% 
% 
% % Specifying length of simulation timesteps
% schedule.step.val = [repmat(year, 30, 1); ...
%                      repmat(10*year, 50, 1)];
% 
% % Specifying which control to use for each timestep.
% % The first 100 timesteps will use control 1, the last 100
% % timesteps will use control 2.
%           
% schedule.step.control = [ones(1, 1); ...
%                          ones(5, 1)* 2;
%                          ones(8, 1)* 3; ...
%                          ones(66, 1) * 4];
                     


%% Model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);

%% Simulate
[wellSol, states] = simulateScheduleAD(initState, model, schedule);

%% Plot results
% % Plotting CO2 saturation for timestep 100 (100 years after start)
% clf;
% 
% % Plotting CO2 staturation for timestep 200 (1100 years after start)
% [h, h_max] = upscaledSat2height(states{end}.s(:,2), states{end}.sGmax, Gt, ...
%                                 'pcWG', fluid.pcWG, ...
%                                 'rhoW', fluid.rhoW, ...
%                                 'rhoG', fluid.rhoG, ...
%                                 'p', states{end}.pressure);
%                             
%                             
% plotCellData(Gt.parent, height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid));
% colorbar; view(-63, 68);
% plotWell(Gt.parent, W, 'color', 'k');
% view(-80,20);
% set(gcf, 'position', [531 337 923 356]); axis tight;
% 
% 

%% Plot trapping inventory
% ta = trapAnalysis(Gt, false);
% reports = makeReports(Gt, {initState states{:}}, model.rock, model.fluid, ...
%                       schedule, [srw, src], ta, []);
% 
% h1 = figure; plot(1); ax = get(h1, 'currentaxes');
% plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');
% set(gcf, 'position', [0 0 1100, 740])
% set(gca, 'fontsize', 20);




% 
% plotCellData(Gt, rock2D.poro, 'edgealpha', 0.4); colorbar; view(-63, 68);
% plotWell(Gt.parent, W, 'color', 'k');
% axis tight; 

% injector_bhps = zeros(length(wellSol), length(wellSol{1}));
% 
% observation_sat = zeros(length(wellSol), 1);
% 
% obs_well_index = sub2ind([100,100],[25],[25]);
% 
% 
% for time_step = 1:length(wellSol)
%         
%     injector_bhps(time_step,:) = cell2mat({wellSol{time_step}.bhp})
%     
%     observation_sat(time_step,:) = cell2mat({states{time_step}.s(obs_well_index,2)})
%     
% 
% end
% 
% time_axis=cell2mat({schedule.step.val})./60/60/24;
% time_axis=cumsum(time_axis);
% 
% hold on
% xlim([0 0.2*10e5])
% plot(time_axis,injector_bhps(:,1))
% plot(time_axis,injector_bhps(:,2))
% plot(time_axis,injector_bhps(:,3))
% hold off

% plotting saturations

% s = states{end}.s(:, 2);
% s = states{end}.s(:, [1,2]);
% 
% s = states{end}.s(:, 2);
% clf
% plotCellData(Gt, s)
% axis tight; view(v)
% 
% plotCellData(Gt, states{end}.s(:, 2), 'edgealpha', 0.4); colorbar; view(-63, 68);
% plotWell(Gt.parent, W, 'color', 'k');
% axis tight; 
% 



sat_map_MATLAB = zeros(size(states,1),80,80,8);
sat_map_all_layers_reconstructed = zeros(size(states,1),80*80*8);    
sat_map_all_first_layer = zeros(size(states,1),80*80*1);

pressure_map_MATLAB = zeros(size(states,1),80,80,8);
pressure_map_first_layer = zeros(size(states,1),80*80*8);

% generating a new folder for each simulation:
SIM_DATA = 'sim_data/'
SIM_DATA_PARAMS = 'sim_data/parameters/'
SAVE_PATH_SIM = [SIM_DATA 'sim_' num2str(i)];
if not(isfolder(SAVE_PATH_SIM))
    mkdir(SAVE_PATH_SIM)
end

%% Process Results

% 
injector_bhps = zeros(length(wellSol), length(wellSol{1}));
observation_sat = zeros(length(wellSol), 1);

obs_well_index = sub2ind([NL,NW],[W_COORDS(1, end)],[W_COORDS(2, end)]);


for time_step = 1:length(wellSol)

    injector_bhps(time_step,:) = cell2mat({wellSol{time_step}.bhp});

    observation_sat(time_step,:) = cell2mat({states{time_step}.s(obs_well_index,2)});

end



%% Extracting Data

% % generating a new folder for each simulation:
% SAVE_PATH_SIM = [SIM_DATA 'sim_' num2str(i) '/'];
% if not(isfolder(SAVE_PATH_SIM))
%     mkdir(SAVE_PATH_SIM)
% end




sat_map_MATLAB = zeros(size(states,1),NL,NW,ND);
sat_map_all_layers_reconstructed = zeros(size(states,1),NL*NW*ND);
sat_map_all_first_layer = zeros(size(states,1),NL*NW*1);

pressure_map_MATLAB = zeros(size(states,1),NL,NW,1);
pressure_map_first_layer = zeros(size(states,1),NL*NW*1);


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


% search_idx = [];
% schedule_idxs = unique(schedule.step.control);
% for iii=1:size(schedule_idxs)
%     
%     current_idx = schedule_idxs(iii);
%     search_idx = [search_idx current_idx];
%     
%     is_member = ismember(schedule.step.control, search_idx);
%     indexies = find(is_member); 
% end
% 



















