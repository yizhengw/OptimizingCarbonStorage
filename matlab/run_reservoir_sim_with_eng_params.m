
function run_reservoir_sim(p, PriorModelParameters, i, SIM_DATA)

% Input: Porosity map
% (80x80x8)
% Output: Bottom Hole pressure for all 3 injection wells
% (time_steps x number of injectors)


% injector well coordinates

 
x_pos_1 = PriorModelParameters.('x_injector_1')(i);
y_pos_1 = PriorModelParameters.('y_injector_1')(i);

x_pos_2 = PriorModelParameters.('x_injector_2')(i);
y_pos_2 = PriorModelParameters.('y_injector_2')(i);

x_pos_3 = PriorModelParameters.('x_injector_3')(i);
y_pos_3 = PriorModelParameters.('y_injector_3')(i);

x_obs = PriorModelParameters.('x_observation_1')(i);
y_obs = PriorModelParameters.('y_observation_1')(i);


% well injection rates
rate_1= 0.5;
rate_2= 0.5;
rate_3= 0.5;


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

% c_rock  = 4.35e-5 / barsa;
% srw     = 0.27; % residual water
% src     = 0.20; % residual CO2
% pe      = 5 * kilo * Pascal; % capillary entry pressure
% muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

c_rock  = PriorModelParameters.('c_rock')(i) / barsa;
srw     = PriorModelParameters.('srw')(i); % residual water
src     = PriorModelParameters.('src')(i); % residual CO2
pe      = PriorModelParameters.('pe')(i) * kilo * Pascal; % capillary entry pressure
muw     = PriorModelParameters.('muw')(i) * Pascal * second; % brine viscosity


% make geo model
% takes porosity map as input - structur and perm are computed and
% returned
[G, Gt, rock, rock2D, bcIxVE] = makeSlopingAquifer_custom_80_80_8(p)


% Specify well information
wc_global1 = false(G.cartDims);
wc_global1(x_pos_1, y_pos_1, 1:8) = true;
wc1 = find(wc_global1(G.cells.indexMap));


inj_rate_1 = rate_1 * mega * 1e3 / year / co2_rho;
inj_rate_2 = rate_2 * mega * 1e3 / year / co2_rho;
inj_rate_3 = rate_3 * mega * 1e3 / year / co2_rho;

W = addWell([], G, rock, wc1, ...
            'type', 'rate', ...  % inject at constant rate
            'val', inj_rate_1, ... % volumetric injection rate
            'comp_i', [0 1]);    % inject CO2, not water


wc_global2 = false(G.cartDims);
wc_global2(x_pos_2, y_pos_2, 1:8) = true;
wc2 = find(wc_global2(G.cells.indexMap));
W = addWell(W, G, rock, wc2, ...
    'type', 'rate', ...  % inject at constant rate
    'val', inj_rate_2, ... % volumetric injection rate
    'comp_i', [0 1]);    % inject CO2, not water

wc_global3 = false(G.cartDims);
wc_global3(x_pos_3, y_pos_3, 1:8) = true;
wc3 = find(wc_global3(G.cells.indexMap));
W = addWell(W, G, rock, wc3, ...
    'type', 'rate', ...  % inject at constant rate
    'val', inj_rate_3, ... % volumetric injection rate
    'comp_i', [0 1]);    % inject CO2, not water

        
%% Grid and rock
% Make top surface grid used for VE simulation
[Gt, G, transMult] = topSurfaceGrid(G);

% Shift G 100 meters down, and plot both grids for comparison
GG = G;
GG.nodes.coords(:,3) = GG.nodes.coords(:,3) + 100;
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

%% SCHEDULE

% Setting up two copies of the well and boundary specifications. 
% Modifying the well in the second copy to have a zero flow rate.
schedule.control    = struct('W', W2D, 'bc', bc2D);
schedule.control(2) = struct('W', W2D, 'bc', bc2D);
schedule.control(3) = struct('W', W2D, 'bc', bc2D);
schedule.control(4) = struct('W', W2D, 'bc', bc2D);

% 1st control: only 1 well
schedule.control(1).W(2).val = inj_rate_1;
schedule.control(1).W(2).val = 0;
schedule.control(1).W(3).val = 0;

% 2nd control: 2 wells
schedule.control(2).W(2).val = inj_rate_1;
schedule.control(2).W(2).val = inj_rate_2;
schedule.control(2).W(3).val = 0;

% 3nd control: 3 wells
schedule.control(3).W(1).val = inj_rate_1;
schedule.control(3).W(2).val = inj_rate_2;
schedule.control(4).W(3).val = inj_rate_3;

% 4th control: all wells shut
schedule.control(4).W(1).val = 0;
schedule.control(4).W(2).val = 0;
schedule.control(4).W(3).val = 0;


% Specifying length of simulation timesteps
schedule.step.val = [repmat(year, 30, 1); ...
                     repmat(10*year, 50, 1)];

% Specifying which control to use for each timestep.
% 1st well in year 1
% 1st + 2nd well after year 1 for 5 years
% 1+2+3 well for year 7-14
% after that shut in (no injection)
         
schedule.step.control = [ones(1, 1); ...
                         ones(5, 1)* 2; ...
                         ones(8, 1)* 3; ...
                         ones(66, 1) * 4];
                     


%% Model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);

%% Simulate
[wellSol, states] = simulateScheduleAD(initState, model, schedule);

%% Plot results
% Plotting CO2 saturation for timestep 100 (100 years after start)
clf;

% Plotting CO2 staturation for timestep 200 (1100 years after start)
% [h, h_max] = upscaledSat2height(states{end}.s(:,2), states{end}.sGmax, Gt, ...
%                                 'pcWG', fluid.pcWG, ...
%                                 'rhoW', fluid.rhoW, ...
%                                 'rhoG', fluid.rhoG, ...
%                                 'p', states{end}.pressure);
% plotCellData(Gt.parent, height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid));
% colorbar; view(-63, 68);
% plotWell(Gt.parent, W, 'color', 'k');
% view(-80,20);
% set(gcf, 'position', [531 337 923 356]); axis tight;
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

% getting the data for optimization
% pressure_history = [];
% for time_step = 1:length(reports)
%     temp_pressure = max(reports(time_step).sol.pressure);
%     pressure_history = [pressure_history; temp_pressure];
% end
% 
% max_pressure = max(pressure_history); 
% trapping_fractions = reports(last_time_step).masses;
% 
% plot(pressure_history)



% plotCellData(Gt, rock2D.poro, 'edgealpha', 0.4); colorbar; view(-63, 68);
% plotWell(Gt.parent, W, 'color', 'k');
% axis tight; 


injector_bhps = zeros(length(wellSol), length(wellSol{1}));
observation_sat = zeros(length(wellSol), 1);

obs_well_index = sub2ind([80,80],[x_obs],[y_obs]);


for time_step = 1:length(wellSol)
        
    injector_bhps(time_step,:) = cell2mat({wellSol{time_step}.bhp});
    
    observation_sat(time_step,:) = cell2mat({states{time_step}.s(obs_well_index,2)})

end


% %% Reconstructing saturation for the last time step
% 
% % Plotting CO2 staturation for timestep 200 (1100 years after start)
% [h, h_max] = upscaledSat2height(states{end}.s(:,2), states{end}.sGmax, Gt, ...
%                                 'pcWG', fluid.pcWG, ...
%                                 'rhoW', fluid.rhoW, ...
%                                 'rhoG', fluid.rhoG, ...
%                                 'p', states{end}.pressure);
%                             
% 
% last_sat_map = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
% 
% last_sat_map_reshaped=reshape(last_sat_map,80,80,8);
% 
% last_sat_map_1st_layer_end = last_sat_map_reshaped(:,:,1);
% 
% % Plotting CO2 staturation for timestep 14 (after injection period)
% [h, h_max] = upscaledSat2height(states{14}.s(:,2), states{14}.sGmax, Gt, ...
%                                 'pcWG', fluid.pcWG, ...
%                                 'rhoW', fluid.rhoW, ...
%                                 'rhoG', fluid.rhoG, ...
%                                 'p', states{end}.pressure);
%                             
% 
% last_sat_map = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
% 
% last_sat_map_reshaped=reshape(last_sat_map,80,80,8);
% 
% last_sat_map_1st_layer_mid = last_sat_map_reshaped(:,:,1);

%% Extracting Data


sat_map_MATLAB = zeros(size(states,1),80,80,8);
sat_map_all_layers_reconstructed = zeros(size(states,1),80*80*8);    
sat_map_all_first_layer = zeros(size(states,1),80*80*1);

pressure_map_MATLAB = zeros(size(states,1),80,80,1);
pressure_map_first_layer = zeros(size(states,1),80*80*1);

% generating a new folder for each simulation:

SAVE_PATH_SIM = [SIM_DATA 'sim_' num2str(i) '/'];
if not(isfolder(SAVE_PATH_SIM))
    mkdir(SAVE_PATH_SIM)
end

for k=1:size(states,1)
    
    % Saturations
    [h, h_max] = upscaledSat2height(states{k}.s(:,2), states{k}.sGmax, Gt, ...
        'pcWG', fluid.pcWG, ...
        'rhoW', fluid.rhoW, ...
        'rhoG', fluid.rhoG, ...
        'p', states{end}.pressure);
    
    
    sat_map = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
    
    sat_map_reshaped=reshape(sat_map,80,80,8);
    
    sat_map_MATLAB(k,:,:,:)= sat_map_reshaped;
    
    sat_map_all_layers_reconstructed(k,:)= sat_map;
    
    sat_map_all_first_layer(k,:)= reshape(sat_map_reshaped(:,:,1), 1,80*80);
    
    
    % Pressure
    
    pressure_map = states{k}.pressure;
    
    pressure_map_reshaped=reshape(pressure_map,80,80,1);
    
    pressure_map_MATLAB(k,:,:,:)= pressure_map_reshaped;
    pressure_map_first_layer(k,:)= pressure_map;
    
end

% Running the trap analysis to get the trapped masses of CO2
ta = trapAnalysis(Gt, false);
reports = makeReports(Gt, {initState states{:}}, model.rock, model.fluid, ...
                      schedule, [srw, src], ta, []);
                  
mass_fractions = cell2mat({reports.masses}.');

time_days = cell2mat({reports.t}.');


%% saving all variables

% Saturations
save([SAVE_PATH_SIM 'sat_map_MATLAB.mat'],'sat_map_MATLAB');
writeNPY(sat_map_all_layers_reconstructed, [SAVE_PATH_SIM 'sat_map_all_layers_reconstructed.npy']);
writeNPY(sat_map_all_first_layer, [SAVE_PATH_SIM 'sat_map_all_first_layer.npy']);

% Pressure
save([SAVE_PATH_SIM 'pressure_map_MATLAB.mat'],'pressure_map_MATLAB');
writeNPY(pressure_map_first_layer, [SAVE_PATH_SIM 'pressure_map_first_layer.npy']);

% Trapped Masses
writeNPY(mass_fractions, [SAVE_PATH_SIM 'mass_fractions.npy']);
    
% Time vector in days
writeNPY(time_days, [SAVE_PATH_SIM 'time_days.npy']);

% Bottom hole pressures of all 3 injector wells
% using well solutions NOT block pressure
writeNPY(injector_bhps, [SAVE_PATH_SIM 'injector_bhps.npy']);

% Sautrations versus time of the Observations well
% using the block saturations 
writeNPY(observation_sat, [SAVE_PATH_SIM 'observation_sat.npy']);
 


end

