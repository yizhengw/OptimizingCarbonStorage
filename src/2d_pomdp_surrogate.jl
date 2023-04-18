struct SurrogatePOMDP2D <: CCSPOMDP
    spec::POMDPSpecification
    max_n_wells::Int64 # From spec, for convenience
    max_placement_t::Float64 # From spec, for convenience
    initial_data::RockObservations # Initial rock observations
    delta::Int64 # Minimum distance between wells
    injection_rate::Float64 # Fixed well injection rate
    rng::AbstractRNG
    function SurrogatePOMDP2D(s::POMDPSpecification; rng::AbstractRNG=Random.GLOBAL_RNG)
        initialize_surrogate()
        new(s, s.max_n_wells, s.max_placement_t, s.initial_data, s.delta, s.injection_rate, rng)
    end
end




# function runMATLABSim(session, porosity::Array{Float64, 3}, w_coords::Array{Int64, 2})
#     put_variable(session, :POROSITY, porosity)
#     put_variable(session, :W_COORDS_J, w_coords)
#     eval_string(session, "run_simulation_julia_semi_general") # Entry Point
#     trapped_fractions = jarray(get_mvariable(session, :trapped_fractions))
#     free_fractions = jarray(get_mvariable(session, :free_fractions))
#     excited_fractions = jarray(get_mvariable(session, :excited_fractions))
#     observation_sat = jarray(get_mvariable(session, :observation_sat))
#     observation_poro = jarray(get_mvariable(session, :observed_poro))
#     return trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro
# end

function initial_matlab()

py"""
import matlab.engine
import numpy as np
def initializeMATLABSim():
    print("Initializing MATLAB session...")
    eng = matlab.engine.start_matlab()
    eng.addpath('matlab')
    eng.eval("clear", nargout=0)
    eng.addpath('/home/ccs/mrst-2022b/')
    eng.addpath('/home/ccs/mrst-2022b/solver_ccs/')
    eng.startup(nargout=0)
    eng.startuplocal_custom(nargout=0)
    print("MATLAB Session Opened")
    return eng


def runMATLABSim(session, porosity, w_coords):
    session.workspace['POROSITY'] = matlab.double(porosity.tolist())
    session.workspace['W_COORDS_J'] = matlab.double(w_coords.tolist())
    session.eval("[a,b,c,d,e] = run_simulation_julia_semi_general(POROSITY, W_COORDS_J);", nargout=0)
    trapped_fractions = session.workspace['a']
    trapped_fractions = np.squeeze(np.array(trapped_fractions))
    free_fractions = session.workspace['b']
    free_fractions = np.squeeze(np.array(free_fractions))
    excited_fractions = session.workspace['c']
    excited_fractions = np.squeeze(np.array(excited_fractions))
    observation_sat = session.workspace['d']
    observation_sat = np.squeeze(np.array(observation_sat))
    observation_poro = session.workspace['e']
    return trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro
"""
end

function dnorm_inj_sg(x)
    return x .* 30
end 


function reward_equation(exited_vol, free_vol, trapped_vol)
    return -1000.0 * exited_vol - 1.0 * free_vol + 10.0 * trapped_vol
end

function calcuate_reward(m::SurrogatePOMDP2D, s::CCSState, surrogate_output::SurrogateModelOutput, normalize::Bool = true)
    reward_mass = normalize ? dnorm_inj_sg(surrogate_output.gs_inject_mass) : surrogate_output.gs_inject_mass;
    exited_volume = reward_mass[3,:]
    free_plume_vol = reward_mass[2,:]
    trapped_vol = reward_mass[1,:]
    if isnothing(surrogate_output.gs_post_satmap)
        start_t = isnothing(s.well_coords) ? 10 : ((size(s.well_coords,2) + 1) * 10 )
        reward =  (start_t - 10 > 0) ? reward_equation((exited_volume[start_t] - exited_volume[start_t - 10]), (free_plume_vol[start_t] - free_plume_vol[start_t - 10]), (trapped_vol[start_t] - trapped_vol[start_t - 10])) : reward_equation(exited_volume[start_t], free_plume_vol[start_t], trapped_vol[start_t])
    else
        reward_mass_post = normalize ? dnorm_inj_sg(surrogate_output.gs_post_mass) : surrogate_output.gs_post_mass;

        exited_volume_post = reward_mass_post[3,:]
        exited_vol_lastwell =  exited_volume_post[end] - exited_volume[20]
        free_volume_post = reward_mass_post[2,:]
        free_vol_lastwell = free_volume_post[end] - free_plume_vol[20]
        trapped_vol_post = reward_mass_post[1,:]
        trapped_vol_lastwell = trapped_vol_post[end] - trapped_vol[20]
        reward = reward_equation(exited_vol_lastwell, free_vol_lastwell, trapped_vol_lastwell)
    end
    return reward
end

function load_conformal_quantiles()
py"""
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
import pickle

def interval_pred(model_,data_, Q):
    qr_lo_pred = model_.predict(np.array(data_).reshape(-1,1),0.075)
    qr_up_pred = model_.predict(np.array(data_).reshape(-1,1),0.925)
    return (max(0,np.minimum(qr_lo_pred, qr_up_pred)[0] - Q) , max(0,np.maximum(qr_lo_pred, qr_up_pred)[0] + Q))

model_col = []
Q_col = []
# save_dir_prefix = "/home/ccs/POMCPOW_quantile2/"
save_dir_prefix = "/home/ccs/sur_output_quantile/quantile_train_8000/"
for idx, ele in enumerate(["trap9","trap19","traplast","free9","free19","freelast","exit9","exit19","exitlast"]):
    model_filename = save_dir_prefix + 'model_'+ele+'.sav'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    model_col.append(loaded_model)
    Q_filename = save_dir_prefix + 'Q_'+ele+'.npy'
    Q = np.load(Q_filename)[0]
    Q_col.append(Q)
"""
end



# function calcuate_reward(m::SurrogatePOMDP2D, s::CCSState, surrogate_output::SurrogateModelOutput, normalize::Bool = true)
#     reward_mass = normalize ? dnorm_inj_sg(surrogate_output.gs_inject_mass) : surrogate_output.gs_inject_mass;
#     exited_volume = reward_mass[3,:]
#     free_plume_vol = reward_mass[2,:]
#     trapped_vol = reward_mass[1,:]
#     if isnothing(surrogate_output.gs_post_satmap)
#         start_t = isnothing(s.well_coords) ? 10 : ((size(s.well_coords,2) + 1) * 10)

#         exited_pass_python = exited_volume[10]
#         free_pass_python = free_plume_vol[10]
#         trap_pass_python = trapped_vol[10]
#         py"""
#         upper_exit = interval_pred(model_col[6],$exited_pass_python,Q_col[6])[1]
#         upper_free = interval_pred(model_col[3],$free_pass_python,Q_col[3])[1]
#         lower_trap = interval_pred(model_col[0],$trap_pass_python,Q_col[0])[0]
#         """
#         if start_t - 10 > 0
#             exited_pass_python_19 = exited_volume[20]
#             free_pass_python_19 = free_plume_vol[20]
#             trap_pass_python_19 = trapped_vol[20]
#             py"""
#             upper_exit_19 = interval_pred(model_col[7],$exited_pass_python_19,Q_col[7])[1]
#             upper_free_19 = interval_pred(model_col[4],$free_pass_python_19,Q_col[4])[1]
#             lower_trap_19 = interval_pred(model_col[1],$trap_pass_python_19,Q_col[1])[0]
#             """
#             #delta_exit = py"upper_exit_19" - py"upper_exit"
#             delta_exit = py"upper_exit_19" - exited_volume[10]
#             delta_free = py"upper_free_19" - py"upper_free"
#             delta_trap = py"lower_trap_19" - py"lower_trap"
#             #reward = reward_equation(delta_exit, delta_free, delta_trap)

#             println("exited should be ",exited_volume[20] - exited_volume[10]," but CF uses ",delta_exit)
#             reward = reward_equation(delta_exit,(free_plume_vol[20] - free_plume_vol[10]), (trapped_vol[20] - trapped_vol[10]))
#         else
#             #reward = reward_equation(py"upper_exit", py"upper_free", py"lower_trap")
#             println("exited should be ",exited_volume[10]," but CF uses ",py"upper_exit")
#             reward = reward_equation(py"upper_exit", free_plume_vol[10], trapped_vol[10])

#         end

#     else
#         reward_mass_post = normalize ? dnorm_inj_sg(surrogate_output.gs_post_mass) : surrogate_output.gs_post_mass;

#         exited_volume_post = reward_mass_post[3,:]
#         free_volume_post = reward_mass_post[2,:]
#         trapped_vol_post = reward_mass_post[1,:]

#         exited_pass_python_19 = exited_volume[20]
#         free_pass_python_19 = free_plume_vol[20]
#         trap_pass_python_19 = trapped_vol[20]

#         exited_pass_python_last = exited_volume_post[end]
#         free_pass_python_last = free_volume_post[end]
#         trap_pass_python_last = trapped_vol_post[end]

#         py"""
#         upper_exit_19 = interval_pred(model_col[7],$exited_pass_python_19,Q_col[7])[1]
#         upper_free_19 = interval_pred(model_col[4],$free_pass_python_19,Q_col[4])[1]
#         lower_trap_19 = interval_pred(model_col[1],$trap_pass_python_19,Q_col[1])[0]

#         upper_exit_last = interval_pred(model_col[8],$exited_pass_python_last,Q_col[8])[1]
#         upper_free_last = interval_pred(model_col[5],$free_pass_python_last,Q_col[5])[1]
#         lower_trap_last = interval_pred(model_col[2],$trap_pass_python_last,Q_col[2])[0]

#         """
#         # exited_vol_lastwell =  py"upper_exit_last" - py"upper_exit_19"
#         exited_vol_lastwell =  py"upper_exit_last" - exited_volume[20]
#         # free_vol_lastwell = py"upper_free_last" - py"upper_free_19"
#         # trapped_vol_lastwell = py"lower_trap_last" - py"lower_trap_19"
#         free_vol_lastwell = free_volume_post[end] - free_plume_vol[20]
#         trapped_vol_lastwell = trapped_vol_post[end] - trapped_vol[20]
#         println("exited should be ",exited_volume_post[end] - exited_volume[20]," but CF uses ",exited_vol_lastwell)
#         reward = reward_equation(exited_vol_lastwell, free_vol_lastwell, trapped_vol_lastwell)
#     end
#     return reward
# end


function calcuate_reward_matlab(s::CCSState, matlab_trap, matlab_free, matlab_exit)
    exited_volume = matlab_exit
    free_plume_vol = matlab_free
    trapped_vol = matlab_trap

    if isnothing(s.well_coords) || size(s.well_coords)[2] < 2

        start_t = isnothing(s.well_coords) ? 9 : ((size(s.well_coords,2) + 1) * 10 -1)

        reward =  (start_t - 9 > 0) ? reward_equation((exited_volume[start_t] - exited_volume[start_t - 10]), (free_plume_vol[start_t] - free_plume_vol[start_t - 10]), (trapped_vol[start_t] - trapped_vol[start_t - 10])) : reward_equation(exited_volume[start_t], free_plume_vol[start_t], trapped_vol[start_t])

    else
        exited_vol_lastwell =  exited_volume[end] - exited_volume[19]
        free_vol_lastwell = free_plume_vol[end] - free_plume_vol[19]
        trapped_vol_lastwell = trapped_vol[end] - trapped_vol[19]
        reward = reward_equation(exited_vol_lastwell, free_vol_lastwell, trapped_vol_lastwell)

    end
    return reward
end


function run_surrogate(s::CCSState, a::CartesianIndex, sp::CCSState)
    initialize_surrogate_input(s)
    coords_p = sp.well_coords


    # #############################
    # save_dir_prefix = "/home/ccs/POMCPOW_quantile/"
    # trap_diff = deserialize(string(save_dir_prefix,"trap_dif.dat"));
    # free_diff = deserialize(string(save_dir_prefix,"free_dif.dat"));
    # exit_diff = deserialize(string(save_dir_prefix,"exit_dif.dat"));

    # trap_diff = trap_diff ./ 30
    # free_diff = free_diff ./ 30
    # exit_diff = exit_diff ./ 30

    # trap_diff = [quantile(row, 0.975) for row in eachrow(trap_diff)];
    # free_diff = [quantile(row, 0.025) for row in eachrow(free_diff)];
    # exit_diff = [quantile(row, 0.025) for row in eachrow(exit_diff)];
    # #############################
    

    # gs_inject_satmap, gs_inject_mass, p_inject = run_inj_surrogate(coords_p)
    gs_inject_satmap, gs_inject_mass, p_inject = isnothing(coords_p) ? (nothing, nothing, nothing) : run_inj_surrogate(coords_p)


    gs_post_satmap, gs_post_mass, p_post = isnothing(coords_p) || size(coords_p,2) < 3 ? (nothing, nothing, nothing) : run_post_surrogate()


    # ##############################
    # if isnothing(gs_inject_mass)
    #     gs_inject_mass = gs_inject_mass

    # else
    #     reward_mass = gs_inject_mass;

    #     exited_volume = reward_mass[3,:]
    #     free_plume_vol = reward_mass[2,:]
    #     trapped_vol = reward_mass[1,:]
    #     exited_volume = exited_volume - exit_diff[1:size(exited_volume)[1]];
    #     exited_volume = [Float32(x) for x in exited_volume];
    #     free_plume_vol = free_plume_vol - free_diff[1:size(free_plume_vol)[1]];
    #     free_plume_vol = [Float32(x) for x in free_plume_vol];
    #     trapped_vol = trapped_vol - trap_diff[1:size(trapped_vol)[1]];
    #     trapped_vol = [Float32(x) for x in trapped_vol];

    #     gs_inject_mass[3,:] = exited_volume;
    #     # gs_inject_mass[2,:] = free_plume_vol;
    #     # gs_inject_mass[1,:] = trapped_vol;
    # end

    # if isnothing(gs_post_mass)
    #     gs_post_mass = gs_post_mass

    # else
    #     reward_mass_post = gs_post_mass;
    #     exited_volume_post = reward_mass_post[3,:];
    #     exited_volume_post = exited_volume_post - exit_diff[31:end];
    #     exited_volume_post = [Float32(x) for x in exited_volume_post];
    #     free_volume_post = reward_mass_post[2,:];
    #     free_volume_post = free_volume_post - free_diff[31:end];
    #     free_volume_post = [Float32(x) for x in free_volume_post];
    #     trapped_vol_post = reward_mass_post[1,:];
    #     trapped_vol_post = trapped_vol_post - trap_diff[31:end];
    #     trapped_vol_post = [Float32(x) for x in trapped_vol_post];

    #     gs_post_mass[3,:] = exited_volume_post;
    #     # gs_post_mass[2,:] = free_volume_post;
    #     # gs_post_mass[1,:] = trapped_vol_post;
    # end

    # ############################


    isnothing(gs_inject_satmap) ? nothing : gs_inject_satmap[gs_inject_satmap .<= 0.0] .= 0.0
    isnothing(gs_inject_mass) ? nothing : gs_inject_mass[gs_inject_mass .<= 0.0] .= 0.0
    isnothing(gs_post_satmap) ? nothing : gs_post_satmap[gs_post_satmap .<= 0.0] .= 0.0
    isnothing(gs_post_mass) ? nothing : gs_post_mass[gs_post_mass .<= 0.0] .= 0.0

    return SurrogateModelOutput(gs_inject_satmap, gs_inject_mass, p_inject, gs_post_satmap, gs_post_mass, p_post)
end

# run_surrogate(s::CCSState, a::CartesianIndex, sp::CCSState) = run_surrogate(m, s, a, sp)


function POMDPs.observation(m::SurrogatePOMDP2D, s::CCSState, a::CartesianIndex, surrogate_output::Union{Nothing,SurrogateModelOutput}=nothing)
    o_porosity = s.porosity[a]
    if isnothing(s.obs_well_coord)
        return Deterministic(CCSObservation(Float64[], Float32[], o_porosity))
    else
        obs_well_x = s.obs_well_coord[1]
        obs_well_y = s.obs_well_coord[2]

        sat_hist = isnothing(s.well_coords) || size(s.well_coords,2) == 0 ? Float32[] : surrogate_output.gs_inject_satmap[1,obs_well_x,obs_well_y,end,1:(size(s.well_coords,2) * 10 - 1)]
        obs = CCSObservation(Float64[],sat_hist, o_porosity) 
        return Deterministic(obs)
    end
end






function POMDPs.gen(m::SurrogatePOMDP2D, s::CCSState, a::CartesianIndex, rng::AbstractRNG=Random.GLOBAL_RNG)
    if s.obs_well_coord == nothing
        coords_p = deepcopy(s.well_coords)
        obs_well_p = [a[1]; a[2]]
        sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
        o = rand(rng, observation(m, s, a))
        r = 0.0
    else
        if s.well_coords == nothing
            coords_p = hcat([a[1]; a[2]])
            obs_well_p = s.obs_well_coord
            sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
            surrogate_output = run_surrogate(s, a, sp)
            o = rand(rng, observation(m, s, a, surrogate_output))
            r = calcuate_reward(m, s, surrogate_output)
        else
            coords_p = hcat(deepcopy(s.well_coords), [a[1]; a[2]])
            obs_well_p = s.obs_well_coord
            sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
            surrogate_output = run_surrogate(s, a, sp)
            o = rand(rng, observation(m, s, a, surrogate_output))
            r = calcuate_reward(m, s, surrogate_output)
        end
    end
    return (sp=sp, o=o, r=r)
end


function random_vector()
    return [rand(1:80), rand(1:80)]
end

function run_matlab(s::CCSState, a::CartesianIndex, sp::CCSState)
    initial_matlab()

    coords_p = sp.well_coords
    porosity = sp.porosity

    # provide fake inject wells the first well is obs and the rest are injectors

    w_coords = hcat(sp.obs_well_coord, sp.well_coords);
    while size(w_coords)[2] != 4
        new_vector = random_vector()
        while any(new_vector .== w_coords)
            new_vector = random_vector()
        end
        w_coords = hcat(w_coords, new_vector)
    end
    if isnothing(coords_p)
        trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro = (nothing, nothing, nothing, nothing, nothing)
    else
        py"""
        session = initializeMATLABSim()
        a,b,c,d,e = runMATLABSim(session, $porosity, $w_coords)
        session.quit()
        """
        trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro = py"a", py"b", py"c", py"d", py"e"

    end
    return trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro
end


function gen_matlab(s::CCSState, a::CartesianIndex)
    if s.obs_well_coord == nothing
        coords_p = deepcopy(s.well_coords)
        obs_well_p = [a[1]; a[2]]
        sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
        o = CCSObservation(Float64[], Float32[], s.porosity[a])
        r = 0.0
    else
        if s.well_coords == nothing
            coords_p = hcat([a[1]; a[2]])
            obs_well_p = s.obs_well_coord
            sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)

            trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro = run_matlab(s, a, sp)
            
            o = CCSObservation(Float64[], Float32[], s.porosity[a])
            r = calcuate_reward_matlab(s, trapped_fractions, free_fractions, excited_fractions)
        else
            coords_p = hcat(deepcopy(s.well_coords), [a[1]; a[2]])
            obs_well_p = s.obs_well_coord
            sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
            trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro = run_matlab(s, a, sp)
            o = CCSObservation(Float64[], Float32.(vec(observation_sat))[1:(size(s.well_coords,2) * 10 - 1)], s.porosity[a])
            r = calcuate_reward_matlab(s, trapped_fractions, free_fractions, excited_fractions)
        end
    end
    return (sp=sp, o=o, r=r)
end


# function POMDPs.gen(m::SurrogatePOMDP2D, s::CCSState, a::CartesianIndex, rng::AbstractRNG=Random.GLOBAL_RNG)
#     if s.well_coords == nothing
#         coords_p = hcat([a[1]; a[2]])
#         obs_well_p = s.obs_well_coord
#         sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
#         surrogate_output = run_surrogate(s, a, sp)
#         o = rand(rng, observation(m, s, a, surrogate_output))
#         r = calcuate_reward(m, s, surrogate_output)
#     else
#         coords_p = hcat(deepcopy(s.well_coords), [a[1]; a[2]])
#         obs_well_p = s.obs_well_coord
#         sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p, s.srw, s.src)
#         surrogate_output = run_surrogate(s, a, sp)
#         o = rand(rng, observation(m, s, a, surrogate_output))
#         r = calcuate_reward(m, s, surrogate_output)
#     end
#     # println("gen checkcheck1: ", s.srw)
#     # println("gen checkcheck2: ", s.src)
#     return (sp=sp, o=o, r=r)
# end

function surrogate_inj_reward(sg_output, dp_output, t) #TODO
    return 0.0
end

function POMDPs.actions(m::SurrogatePOMDP2D)
    spacing = m.spec.grid_spacing + 1
    idxs = CartesianIndices(m.spec.grid_dim)[1:spacing:end, 1:spacing:end]
    reshape(collect(idxs), prod(size(idxs)))
end

function POMDPs.actions(m::SurrogatePOMDP2D, s::CCSState)
    action_set = Set(POMDPs.actions(m))  #order set
    for i=1:size(s.well_coords)[2]
        coord = s.well_coords[:, i]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

function POMDPs.actions(m::SurrogatePOMDP2D, b::ReservoirBelief)
    action_set = Set(POMDPs.actions(m))
    n_initial = length(m.initial_data)
    n_obs = size(b.rock_belief.data.coordinates)[2] - n_initial
    for i=1:n_obs
        coord = b.rock_belief.data.coordinates[:, i + n_initial]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

POMDPs.discount(::SurrogatePOMDP2D) = 0.99


# comment out this the other one is in pomdp_base
function POMDPs.isterminal(m::SurrogatePOMDP2D, s::CCSState)
    if isnothing(s.well_coords)  
        return false
    else
        n_initial = 0
        n_wells = size(s.well_coords)[2] - n_initial
        return n_wells >= m.max_n_wells 
    end
end

############## the following one is the final used function
function POMDPModelTools.obs_weight(m::SurrogatePOMDP2D, s::CCSState,
                                    a::CartesianIndex, sp::CCSState, o::CCSObservation)
    w_porosity = pdf(truncated(Normal(sp.porosity[a], 0.1*abs(sp.porosity[a])),0,1),o.porosity)
    if isnothing(sp.well_coords)
        return w_porosity
    else
        surrogate_output = run_surrogate(s, a, sp)
        # porosity_weight = P(s.porosity | o)
        obs_well_x = sp.obs_well_coord[1]
        obs_well_y = sp.obs_well_coord[2]
        
        sat_currentstate = size(sp.well_coords,2) == 1 ? Float32[] : surrogate_output.gs_inject_satmap[1,obs_well_x,obs_well_y,end,1:((size(sp.well_coords,2) - 1) * 10 - 1)]
        if sum(sat_currentstate .==0) >=1
            w_sat = 1
        else
            mean_ = sat_currentstate
            vari_ = Matrix(one(eltype(mean_))I, size(mean_,1), size(mean_,1)) .* [0.1*ele for ele in mean_]
            w_sat = pdf(MvNormal(mean_,vari_ ),o.saturation_history)
        end
        return w_porosity*w_sat
    end
end





# function POMDPModelTools.obs_weight(m::SurrogatePOMDP2D, s::CCSState,
#                                     a::CartesianIndex, sp::CCSState, o::CCSObservation)
#     w_porosity = pdf(truncated(Normal(sp.porosity[a], 0.1*sp.porosity[a]),0,1),o.porosity)
#     return w_porosity
# end



# function POMDPModelTools.obs_weight(m::SurrogatePOMDP2D, s::CCSState,
#                                     a::CartesianIndex, sp::CCSState, o::CCSObservation)
#     if isnothing(sp.well_coords)
#         return exp(-(sp.porosity[a] - o.porosity))
#     else
#         surrogate_output = run_surrogate(s, a, sp)
#         # TODO: Compare surrogate output (MSE)
#         # porosity_weight = P(s.porosity | o)
#         # porosity_weight * exp(-MSE_pressure_histories)
#         # NOTE: Start with MSEs
#         obs_well_x = sp.obs_well_coord[1]
#         obs_well_y = sp.obs_well_coord[2]
#         o_currentstate = sp.porosity[a]
#         sat_currentstate = size(sp.well_coords,2) == 1 ? Float32[] : surrogate_output.gs_inject_satmap[1,obs_well_x,obs_well_y,end,1:((size(sp.well_coords,2) - 1) * 5 - 1),1]
#         w_porosity = exp(-(o_currentstate - o.porosity))
#         w_sat = exp(-sum(sat_currentstate .- o.saturation_history))
#         return w_porosity*w_sat
#     end
# end

# function POMDPModelTools.obs_weight(m::SurrogatePOMDP2D, s::CCSState,
#                                     a::CartesianIndex, sp::CCSState, o::CCSObservation)
#     if isnothing(sp.well_coords)
#         return exp(-(sp.porosity[a] - o.porosity))
#     else
#         # TODO: Compare surrogate output (MSE)
#         # porosity_weight = P(s.porosity | o)
#         # porosity_weight * exp(-MSE_pressure_histories)
#         # NOTE: Start with MSEsobs_well_x = sp.obs_well_coord[1]
        
#         o_currentstate = sp.porosity[a]
    
#         w_porosity = exp(-(o_currentstate - o.porosity))
        
#         return w_porosity
#     end
# end
