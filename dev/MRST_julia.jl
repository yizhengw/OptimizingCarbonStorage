using MATLAB
using Distributions
using GeoStats
using Plots

# function map_generator()
#
#     𝒢 = CartesianGrid(80, 80)
#     𝒫 = SimulationProblem(𝒢, :Z => Float64, 1)
#     S1 = FFTGS(:Z => (mean=0.2,variogram=SphericalVariogram( sill=0.001, range=20.0,
#                                             nugget=0.0001)))
#     sol = solve(𝒫, S1)
#
#     temp_poro=reshape(sol.reals.Z[1],80,80,1)
#     poro = repeat(temp_poro, outer=[1,1,8])
#
#     return poro
#
#
# end

# function initializeMRST_sim()
#
#     println("Initializing MATLAB session...")
#     session = MSession(0)
#
#     eval_string(session, "addpath('/Users/markuszechner/Documents/code-dev/CCS_dev/CCS-core/matlab')")
#     eval_string(session, "addpath('/Users/markuszechner/Documents/MATLAB/mrst-2020b/modules/co2lab')")
#
#     eval_string(session, "LoadCO2_modules")
#
#     eval_string(session, "initialize_CCS_model") #TODO: add grid dim variables
#
#     return session
#
# end
#
#
# function runMRST_sim(poro::Array{Float64,3}, perm::Array{Float64,3},
#     W_COORDS, c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real)
#
#     session = initializeMRST_sim() #TODO only call this once
#
#     put_variable(session, :WELL_COORD, W_COORDS)
#
#     paramters = Float64[c_rock srw src pe muw]
#     put_variable(session, :PARAMETERS, paramters)
#
#     put_variable(session, :PORO, poro)
#     put_variable(session, :PERM, perm)
#
#     eval_string(session, "run_reservoir_simJULIA")
#
#
#     injector_bhps = jarray(get_mvariable(session, :injector_bhps))
#     mass_fractions = jarray(get_mvariable(session, :mass_fractions))
#     time_days = jarray(get_mvariable(session, :time_days))
#     pressure_map_first_layer = jarray(get_mvariable(session, :pressure_map_first_layer))
#     observation_sat = jarray(get_mvariable(session, :observation_sat))
#
#     return [injector_bhps,mass_fractions, time_days, pressure_map_first_layer, observation_sat]
#
# end
#
function reward_function(matlab_output::Array{Array{Float64,2},1})
                    # % legend names
                    # names = {'Dissolved'           , 'Structural residual' , ...
                    #          'Residual'            , 'Residual in plume'  , ...
                    #          'Structural subscale' , 'Structural plume'   , ...
                    #          'Free plume'          , 'Exited'};
    mass_fract_reward = matlab_output[2][2:end,:]
    existed_vol = mass_fract_reward[:,8][end]/mean(mass_fract_reward[:,8])
    free_plume_vol = mass_fract_reward[:,7][end]/mean(mass_fract_reward[:,7])
    trapped_vol = sum(mass_fract_reward[:,1:6],dims=2)[end]/mean(sum(mass_fract_reward[:,1:6],dims=2))
    pressure = max(maximum(matlab_output[1]),maximum(matlab_output[4]))
    initial_pressure =  maximum(matlab_output[4][1,:])
    r =(pressure>initial_pressure*1.2)*(-1000) + existed_vol*(-1000) + free_plume_vol *(-5) + trapped_vol*(10)

    return r

end

# NEW FUNCTIONS

function initializeMATLABSim(nl::Real, nw::Real, nd::Real,
                            MRST_dir::String,TOTAL_TIME::Real,
                            INJECTION_STOP::Real, NSTEP_INJECTION::Real,
                            NSTEP_SHUT_IN::Real, INTERVAL::Real)

                    # initializeMATLABSim(nl::Real, nw::Real, nd::Real, l::Real, w::Real,
                    #                     d::Real, interval::Real, total_time::Real, MRST_dir::String)
    println("Initializing MATLAB session...")
    session = MSession(0)
    eval_string(session, "addpath('$MRST_dir')")
    #eval_string(session, "addpath('$FILE_DIR')")
    eval_string(session, "addpath('matlab')")
    # eval_string(session, "startup")
    eval_string(session, "clear")

    # grid variables
    grid_array = Float64[nl nw nd]
    put_variable(session, :GRID_ARRAY, grid_array)

    # schedule variables
    put_variable(session, :TOTAL_TIME, total_time) # in years
    put_variable(session, :INJECTION_STOP, injection_stop) # in years from beginning
    put_variable(session, :NSTEP_INJECTION, nstep_injection) # number of MRST simulation steps/numerical
    put_variable(session, :NSTEP_SHUT_IN, nstep_shut_in)# number of MRST simulation steps
    put_variable(session, :INTERVAL, interval) # time between wells

    # domain_array = Float64[l w d]
    # put_variable(session, :DOMAIN_ARRAY, domain_array)

    # schedule variables
    # put_variable(session, :INTERVAL, Float64[interval])
    # put_variable(session, :TOTAL_TIME,  Float64[total_time])

    eval_string(session, "LoadCO2_modules")
    eval_string(session, "initialize_CCS_model")
    println("MATLAB Session Opened")
    return session
end

function runMATLABSim(session, porosity::Array{Float64, 3}, perm::Array{Float64,3},
     w_coords::Array{Int64, 2},c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real)

    # c_rock  = 4.35e-5
    # srw     = 0.27
    # src     = 0.20
    # pe      = 5.0
    # muw     = 8e-4
    paramters = Float64[c_rock srw src pe muw]
    put_variable(session, :PARAMETERS, paramters)

    put_variable(session, :POROSITY, porosity)
    put_variable(session, :PERMEABILITY, perm)


    put_variable(session, :W_COORDS, w_coords)

    eval_string(session, "run_reservoir_simJULIA_splitted") # Entry Point
    # vOs = jarray(get_mvariable(session, :vOs))
    # times = jarray(get_mvariable(session, :times))
    # # return vOs, times
    # #####
    # qOs = jarray(get_mvariable(session, :qOs))
    # #####
    # return vOs, qOs, times
    injector_bhps = jarray(get_mvariable(session, :injector_bhps))
    mass_fractions = jarray(get_mvariable(session, :mass_fractions))
    time_days = jarray(get_mvariable(session, :time_days))
    pressure_map_first_layer = jarray(get_mvariable(session, :pressure_map_first_layer))
    observation_sat = jarray(get_mvariable(session, :observation_sat))

    return [injector_bhps,mass_fractions, time_days, pressure_map_first_layer, observation_sat]
end

function runMATLABSim_splitted_reward(session, porosity::Array{Float64, 3}, perm::Array{Float64,3},
     w_coords::Array{Int64, 2},c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real)

    # c_rock  = 4.35e-5
    # srw     = 0.27
    # src     = 0.20
    # pe      = 5.0
    # muw     = 8e-4
    paramters = Float64[c_rock srw src pe muw]
    put_variable(session, :PARAMETERS, paramters)

    put_variable(session, :POROSITY, porosity)
    put_variable(session, :PERMEABILITY, perm)


    put_variable(session, :W_COORDS, w_coords)

    eval_string(session, "run_reservoir_simJULIA_splitted") # Entry Point
    # vOs = jarray(get_mvariable(session, :vOs))
    # times = jarray(get_mvariable(session, :times))
    # # return vOs, times
    # #####
    # qOs = jarray(get_mvariable(session, :qOs))
    # #####
    # return vOs, qOs, times
    injector_bhps = jarray(get_mvariable(session, :injector_bhps))
    mass_fractions = jarray(get_mvariable(session, :mass_fractions))
    time_days = jarray(get_mvariable(session, :time_days))
    pressure_map_first_layer = jarray(get_mvariable(session, :pressure_map_first_layer))
    observation_sat = jarray(get_mvariable(session, :observation_sat))
    schedule_idx = jarray(get_mvariable(session, :schedule_idx))

    return schedule_idx, injector_bhps, mass_fractions, time_days, pressure_map_first_layer, observation_sat
end


function reward_function_splitted_reward(schedule_idx, injector_bhps, mass_fractions, time_days, pressure_map_first_layer, observation_sat)
            # % legend names
            # names = {'Dissolved'           , 'Structural residual' , ...
            #          'Residual'            , 'Residual in plume'  , ...
            #          'Structural subscale' , 'Structural plume'   , ...
            #          'Free plume'          , 'Exited'};


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

    return reward


end
