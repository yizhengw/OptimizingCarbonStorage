% initialize_CCS_model

%%% Input Vars %%%
% GRID_ARRAY (l, w, d): Length, width, and depth of domain in grid cells

% Activate MRST Modules
mrstModule add ad-core
mrstModule add co2lab
mrstModule add ad-props
mrstModule add mimetic
mrstModule add libgeometry
mrstModule add deckformat
mrstModule add coarsegrid

% Compute Grid
NL = GRID_ARRAY(1);
NW = GRID_ARRAY(2);
ND = GRID_ARRAY(3);

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
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity
