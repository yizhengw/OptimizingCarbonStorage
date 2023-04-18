include("MRST_julia.jl")
using Distributions

using Revise
#using JLD
#load("/Users/markuszechner/Downloads/MapCollections.jld")["map"]
c_rock  = 4.35e-5
srw     = 0.27
src     = 0.20
pe      = 5
muw     = 8e-4

# Inj_1_x=10
# Inj_1_y=10
#
# Inj_2_x=20
# Inj_2_y=20
#
# Inj_3_x=30
# Inj_3_y=30
#
# Obs_1_x=40
# Obs_1_y=40

nl=80
nw=80
nd=8

# TODO get the ordering right - first well is obs well
W_COORDS=[10 20 30 40; 10 20 30 40]

total_time = 530.0;
injection_stop=30.0;

nstep_injection = 30.0;
nstep_shut_in = 20.0;
interval = 5.0;


poro = rand(Uniform(0.2,0.3),80,80,8)
perm = poro.^3.0*(1e-5).^2.0./(0.81*72.0*(1.0.-poro).^2.0)

MRST_dir = "/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab"

session = initializeMATLABSim(nl::Real, nw::Real, nd::Real,
                            MRST_dir::String,total_time::Real,
                            injection_stop::Real, nstep_injection::Real,
                            nstep_shut_in::Real, interval::Real)

# matlab_output = runMATLABSim(session, poro::Array{Float64, 3}, perm::Array{Float64,3},
#                      W_COORDS::Array{Int64, 2},c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real)

schedule_idx, injector_bhps, mass_fractions_plotting, time_days, pressure_map_first_layer, observation_sat =
                runMATLABSim_splitted_reward(session, poro::Array{Float64, 3}, perm::Array{Float64,3},
                W_COORDS::Array{Int64, 2},c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real)


reward = reward_function_splitted_reward(schedule_idx, injector_bhps, mass_fractions_plotting, time_days, pressure_map_first_layer, observation_sat)




schedule_idx_v = Int.(vec(schedule_idx))

schedule_index_unique = unique(schedule_idx_v)
search_idx = []

mass_fractions = mass_fractions_plotting[2:end,:] # take the added 0 out

reward=[]

for i in 1:length(schedule_index_unique)


    current_idx = schedule_index_unique[i]
    push!(search_idx, current_idx)

    println(search_idx)

    current_indices=indexin(schedule_idx_v, search_idx)


    injector_bhps_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(injector_bhps)[2])
    [injector_bhps_temp[:,i] = injector_bhps[:,i][schedule_idx_v.==current_indices] for i in 1:size(injector_bhps)[2]]


    mass_fractions_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(mass_fractions)[2])
    [mass_fractions_temp[:,i] = mass_fractions[:,i][schedule_idx_v.==current_indices] for i in 1:size(mass_fractions)[2]]


    pressure_map_first_layer_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(pressure_map_first_layer)[2])
    [pressure_map_first_layer_temp[:,i] = pressure_map_first_layer[:,i][schedule_idx_v.==current_indices] for i in 1:size(pressure_map_first_layer)[2]]

    observation_sat_temp = observation_sat[schedule_idx_v.==current_indices]


    #mass_fract_reward = mass_fractions[2:end,:]
    existed_vol = mass_fractions_temp[:,8][end]/mean(mass_fractions_temp[:,8])
    free_plume_vol = mass_fractions_temp[:,7][end]/mean(mass_fractions_temp[:,7])
    trapped_vol = sum(mass_fractions_temp[:,1:6],dims=2)[end]/mean(sum(mass_fractions_temp[:,1:6],dims=2))
    pressure = max(maximum(injector_bhps_temp),maximum(pressure_map_first_layer_temp))
    initial_pressure =  maximum(pressure_map_first_layer_temp[1,:]) #TODO check
    reward_temp =(pressure>initial_pressure*1.2)*(-1000) + existed_vol*(-1000) + free_plume_vol *(-5) + trapped_vol*(10)

    push!(reward, reward_temp)



end
