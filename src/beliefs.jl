mutable struct ReservoirBelief #TODO Add well locations
    rock_belief::GSLIBDistribution
    pressure_history::Vector{Float64}
    saturation_history::Vector{Float64}
    well_coords::Union{Nothing, Matrix{Int64}}
    obs_well_coords::Union{Nothing, Vector{Int64}}
    engineer_parameters_belief::EngineeringParameterBelief
end

Distributions.support(d::ReservoirBelief) = [rand(d)]

struct ReservoirBeliefUpdater <: POMDPs.Updater
    spec::POMDPSpecification
    num_MC
    maxIter
    ReservoirBeliefUpdater(spec::POMDPSpecification;  num_MC = 300, maxIter=1) = new(spec, num_MC, maxIter)
end

function obs_model(s::CCSState, a::CartesianIndex, surrogate_output::Union{Nothing,SurrogateModelOutput}=nothing)
    o_porosity = s.porosity[a]
    if isnothing(s.obs_well_coord)
        return Deterministic(CCSObservation(Float64[], Float32[], o_porosity))
    else
        obs_well_x = s.obs_well_coord[1]
        obs_well_y = s.obs_well_coord[2]

        sat_hist = isnothing(s.well_coords) || size(s.well_coords,2) == 0 ? Float32[] : surrogate_output.gs_inject_satmap[1,obs_well_x,obs_well_y,end,1:(size(s.well_coords,2) * 10 - 1),1]

        obs = CCSObservation(Float64[],sat_hist, o_porosity) 
        return Deterministic(obs)
    end
end

function run_esmda(para_samples, obs_samples, real_obs, alpha, measurement_error)
    #input arguments dimensions:
    #para_samples: num_MC * num_PCs
    #obs_samples:  num_MC * time_step
    #real_obs: a vector (timestep,:)
     # computing C_d, the covariance that includes the measurement error
    measurement_error = measurement_error
    std_data = Diagonal(measurement_error) # C_d = 0.1% error. time_step * time_step

    m = transpose(para_samples) # num_PCs * num_MC 
    # println("m size should be num_PCs by num_MC: ", size(m))
    count = 0 
    num_ensem = size(m)[2]
    num_para = size(m)[1]
    length_data = size(real_obs)[1]  # length of the data variable, in other words: timesteps
    perb_data = zeros(length_data, num_ensem) 
    upd_para = zeros(num_para, num_ensem)
    # for p in 1:maxIter
    for i in 1:length_data
        perb_data[i,:] = rand(Normal(real_obs[i], alpha * (std_data[i,i])), num_ensem)
    end
    dPrior = transpose(obs_samples) # time_step * num_MC
    # println("dPrior dimension should be time_step by num_MCs: ", size(dPrior))
    prior_para = transpose(para_samples) # num_PCs * num_MC 
    # println("prior_para dimension should be num_PCs by num_MCs: ", size(prior_para))
    diff_para = transpose(transpose(prior_para) .- squeeze(mean(prior_para, dims = 2))')  #  num_PC * num_MC 
    diff_data = transpose(transpose(dPrior) .- squeeze(mean(dPrior, dims = 2))')  #timestep * numMC
    # println("diff_para dimension should be num_PC by num_MC: ", size(diff_para))
    # println("diff_data dimension should be time_step by num_MC: ", size(diff_data))
    CMD = (diff_para * transpose(diff_data)) ./  (num_ensem - 1)  #num_PC * num_MC     numMC *timestep 
    CDD = (diff_data * transpose(diff_data)) ./ (num_ensem - 1) # dim * dim
    # println("CMD dimension should be num_PC by time_step: ", size(CMD))
    # println("CDD dimension should be time_step by time_step: ", size(CDD))
    # println("CDD dimension should be time_step by time_step: ", CDD)
    # println("std: ", std_data)
    # println("inv: ", CDD .+ alpha .* std_data)
    try
        K = CMD * inv(CDD .+ alpha .* std_data)
        for j in 1:num_ensem   
            rj = perb_data[:,j] .- dPrior[:,j]
            upd_para[:,j] = prior_para[:,j] .+ (K *rj)
        end
        # println("upd_para dimension should be num_PCs by num_MCs: ", size(upd_para))
        prior_para = upd_para
    # println("esmda return should be num_PCs by num_MCs: ", size(prior_para))
    catch
        println("no esmda updates -- just return prior pc scores!")
    end
    return prior_para 
end


function squeeze( A :: AbstractArray )
     keepdims = Tuple(i for i in size(A) if i != 1);
     return reshape( A, keepdims );
end

function POMDPs.update(up::ReservoirBeliefUpdater, b::ReservoirBelief,
                    action::CartesianIndex, obs::CCSObservation)

    act_array = reshape(Float64[action[1] action[2]], 2, 1)
    if b.obs_well_coords == nothing
        obs_well_coords = [action[1], action[2]]
        well_coords = b.well_coords
    else
        obs_well_coords = b.obs_well_coords
        well_coords = hcat(b.well_coords, [action[1], action[2]])
    end
    bp = ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, well_coords, obs_well_coords, b.engineer_parameters_belief)
    bp.rock_belief.data.coordinates = [b.rock_belief.data.coordinates act_array] # This is specific to our problem formulation where action:=location
    bp.rock_belief.data.porosity = [bp.rock_belief.data.porosity; obs.porosity]
    append!(bp.pressure_history, obs.pressure_history)
    append!(bp.saturation_history, obs.saturation_history)



    println("sat hist:", obs.saturation_history)
    if size(obs.saturation_history)[1] != 0 && (sum(obs.saturation_history==0)/size(obs.saturation_history)[1]) <0.4
        reconstruct_res = nothing
        sat_obs =  obs.saturation_history
        measurement_error = sat_obs .* 0.001
        for i in 1:up.maxIter
            map_samples = nothing
            sat_samples = nothing
            entire_sat_samples = nothing
            for j in 1:up.num_MC
                # if (i*j) % 100 == 0 
                #     println("In iteration ", i*j, " out of ", up.maxIter*up.num_MC)
                # end
                if isnothing(reconstruct_res)
                    bp.rock_belief.esmda_update = nothing
                    s_current = rand(b)
                    s_next = rand(bp)
                    sp = CCSState(s_next.porosity, s_next.permeability, s_current.well_coords, s_current.obs_well_coord, s_current.srw, s_current.src)
                    porosity = sp.porosity
                    permeability = sp.permeability
                else
                    s_current = rand(b)
                    vals = reconstruct_res[:,j];
                    poro_2D = reshape(vals, (80,80,1));
                    porosity = repeat(poro_2D, outer=(1, 1, 8));
                    permeability = porosity.^3.0*(1e-5).^2.0./(0.81*72.0*(1.0.-porosity).^2.0);
                    sp = CCSState(porosity, permeability, s_current.well_coords, s_current.obs_well_coord, s_current.srw, s_current.src)
                end

                map_samples = isnothing(map_samples) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(map_samples,[reshape(porosity[:,:,1],(1,:))]);
                surrogate_output = run_surrogate(sp, action, sp);
                obs_well_x = sp.obs_well_coord[1]
                obs_well_y = sp.obs_well_coord[2]
                sat_hist = isnothing(sp.well_coords) || size(sp.well_coords,2) == 0 ? Float32[] : surrogate_output.gs_inject_satmap[1,obs_well_x,obs_well_y,end,1:(size(sp.well_coords,2) * 10 - 1)]

                sat_samples = isnothing(sat_samples) ? vcat([reshape(sat_hist,(1,:))]) : vcat(sat_samples,[reshape(sat_hist,(1,:))])

                # entire_sat_samples = isnothing(entire_sat_samples) ? vcat([reshape(surrogate_output.gs_inject_satmap[1,:,:,end,1:(size(sp.well_coords,2) * 10 - 1)],tuple(1, size(surrogate_output.gs_inject_satmap[1,:,:,end,1:(size(sp.well_coords,2) * 10 - 1)])...))]) : vcat(entire_sat_samples,[reshape(surrogate_output.gs_inject_satmap[1,:,:,end,1:(size(sp.well_coords,2) * 10 - 1)],(1,size(surrogate_output.gs_inject_satmap[1,:,:,end,1:(size(sp.well_coords,2) * 10 - 1)])...))])

            end
            map_samples_new = reshape(reduce(vcat, map_samples), up.num_MC, :);
            sat_samples_new = reshape(reduce(vcat, sat_samples), up.num_MC, :);
            # entire_sat_samples_new = reshape(reduce(vcat, entire_sat_samples), up.num_MC, :);
            M = MultivariateStats.fit(PCA, convert(Matrix,map_samples_new'));
            pc_scores = MultivariateStats.transform(M, convert(Matrix,map_samples_new'))
            res_es = run_esmda(pc_scores', sat_samples_new, sat_obs, 1, measurement_error)
            reconstruct_res = MultivariateStats.reconstruct(M,res_es)
        end
        bp.rock_belief.esmda_update = reconstruct_res
    end


    # if size(obs.saturation_history)[1] != 0
    #     obs_smaples = nothing
    #     state_samples = Vector{CCSState}()
    #     diff_o_obs = Vector{Float64}()
    #     for i in 1:up.abc_samples
    #         println(string("ABC in loop ", i, "/", up.abc_samples))
    #         s_current = rand(b)
    #         s_next = rand(bp)
    #         s_next = CCSState(s_next.porosity, s_next.permeability, s_current.well_coords, s_current.obs_well_coord, s_current.srw, s_current.src)
    #         surrogate_output_next_temp = run_surrogate(s_next, action, s_next)
    #         o_next_temp = rand(obs_model(s_next, action, surrogate_output_next_temp))
    #         #push!(obs_smaples, o_next_temp.saturation_history)
    #         obs_smaples = isnothing(obs_smaples) ? hcat(o_next_temp.saturation_history) : hcat(obs_smaples,o_next_temp.saturation_history)
    #         obs_true_obs_samples = sqrt(sum((obs.saturation_history - o_next_temp.saturation_history) .^ 2))
    #         push!(state_samples, s_next)
    #         push!(diff_o_obs,Float64(obs_true_obs_samples))
    #     end
    #     dis_o_samples = pairwise(Euclidean(), obs_smaples, dims = 2)
    #     pair_dis_upper_tri = dis_o_samples[triu!(trues(size(dis_o_samples)), 1)]
    #     threshold_abc = quantile(pair_dis_upper_tri, up.abc_threshold)
    #     posterior_state = state_samples[diff_o_obs .< threshold_abc]
    #     println("Here in the posterior state: ", size(posterior_state))
    #     posterior_eng = reduce(hcat, [[ele.srw ; ele.src ] for ele in posterior_state])
    #     # bp = @set bp.engineer_parameters_belief = EngineeringParameterBelief(posterior_eng[1,:], posterior_eng[2,:])
    #     bp.engineer_parameters_belief = EngineeringParameterBelief(posterior_eng[1,:], posterior_eng[2,:]) 
    #     return bp
    # else
    #     return bp
    # end
    return bp

end



# function POMDPs.update(up::ReservoirBeliefUpdater, b::ReservoirBelief,
#                     action::CartesianIndex, obs::CCSObservation)

#     act_array = reshape(Float64[action[1] action[2]], 2, 1)
#     obs_well_coords = b.obs_well_coords
#     well_coords = hcat(b.well_coords, [action[1], action[2]])

#     bp = ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, well_coords, obs_well_coords, b.engineer_parameters_belief)
#     bp.rock_belief.data.coordinates = [b.rock_belief.data.coordinates act_array] # This is specific to our problem formulation where action:=location
#     bp.rock_belief.data.porosity = [bp.rock_belief.data.porosity; obs.porosity]
#     append!(bp.pressure_history, obs.pressure_history)
#     append!(bp.saturation_history, obs.saturation_history)



#     return bp
# end

function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, s::CCSState)
    rock_belief = GSLIBDistribution(up.spec)
    init_well_coords = rock_belief.data.coordinates
    eng_belief = EngineeringParameterBelief(up.spec)
    ReservoirBelief(rock_belief, Float64[], Float64[], init_well_coords, nothing, eng_belief)
end

function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, b::ReservoirBelief)
    ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, b.well_coords, b.obs_well_coords, b.engineer_parameters_belief)
end



function Base.rand(rng::AbstractRNG, b::ReservoirBelief)
    rock_state = rand(rng, b.rock_belief)
    eng_state = rand(rng, b.engineer_parameters_belief)
    CCSState(rock_state[1], rock_state[2], b.well_coords, b.obs_well_coords,eng_state[1],eng_state[2])
end

Base.rand(b::ReservoirBelief) = Base.rand(Random.GLOBAL_RNG, b)

function calculate_obs_weight(o::Matrix{Float64}, o_true::Matrix{Float64};
                            l::Float64=1.0)
    d = mean((o - o_true).^2)
    return exp(-d/(2*l^2))
end

function Plots.plot(b::ReservoirBelief) # TODO add well plots
    poro_mean, poro_var = solve_gp(b.rock_belief)
    # println(size(poro_mean))
    # STOP
    fig1 = heatmap(poro_mean[:,:,1], title="Porosity Mean", fill=true, clims=(0.0, 0.38))
    fig2 = heatmap(poro_var[:,:,1], title="Porosity Variance")
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
