
abstract type CCSPOMDP <: POMDP{CCSState, CartesianIndex, CCSObservation} end

@with_kw struct POMDPSpecification
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (80, 80, 1) #  dim x dim grid size
    max_n_wells::Int64 = 3 # Maximum number of wells
    time_interval::Float64 = 1.0 # Minimum time between wells (in years)
    max_placement_t::Float64 = 10.0 # Maximum time to drill wells
    simulation_t::Float64 = 30.0 # Simulation time horizon
    reward_weights::Vector{Float64} = [1.0, 1.0, 1.0, 1.0]
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 2 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 1 # Number of cells in between each cell in which wells can be placed
    injection_rate::Float64 = 10.0 # Fixed injection rate in # TODO: Units and default values
    poro_mean::Float64 = 0.2 # Porosity variogram mean # TODO Not used
    Az::Float64 = 1300.0 # KC vertical coefficients # TODO Not used
    Axy::Float64 = 1000.0 # KC horizontal coefficients # TODO Not used
    poro_variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    poro_nugget::Tuple = (1, 0)
    srw_bounds::Vector{Float64} = [0.25,0.35]
    src_bounds::Vector{Float64} = [0.15,0.25]
    num_samples_eng_para::Int64 = 5000
    # abc_samples::Int64 = 500
    # abc_threshold::Float64 = 0.2
end

function GSLIBDistribution(s::POMDPSpecification)
    return GSLIBDistribution(grid_dims=s.grid_dim, n=s.grid_dim,
            data=s.initial_data, variogram=s.poro_variogram, nugget=s.poro_nugget)
end




struct EngineeringParameterBelief
    srw_belief::Vector{Float64} # use kernel density to resample new data based on this 
    src_belief::Vector{Float64} # use kernel density to resample new data based on this 
end

function EngineeringParameterBelief(s::POMDPSpecification)
    return EngineeringParameterBelief(rand(Distributions.Uniform(s.srw_bounds[1],s.srw_bounds[2]),s.num_samples_eng_para), rand(Distributions.Uniform(s.src_bounds[1],s.src_bounds[2]),s.num_samples_eng_para))

end

function Base.rand(rng::AbstractRNG, eb::EngineeringParameterBelief)
    ## need to add boundary to new samples
    h_srw = KernelDensity.default_bandwidth(eb.srw_belief)
    new_srw = rand(Normal(rand(eb.srw_belief) , h_srw))

    h_src = KernelDensity.default_bandwidth(eb.src_belief)
    new_src = rand(Normal(rand(eb.src_belief) , h_src))

    return new_srw, new_src
end

Base.rand(eb::EngineeringParameterBelief) = Base.rand(Random.GLOBAL_RNG,eb)

"""
    sample_coords(dims::Tuple{Int, Int}, n::Int)
Sample coordinates from a Cartesian grid of dimensions given by dims and return
them in an array
"""
function sample_coords(dims::Tuple{Int, Int, Int}, n::Int)
    idxs = CartesianIndices(dims)
    samples = sample(idxs, n)
    sample_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(samples)
        sample_array[1, i] = sample[1]
        sample_array[2, i] = sample[2]
    end
    return (samples, sample_array)
end

function sample_initial(spec::POMDPSpecification, n::Integer)
    coords, coords_array = sample_coords(spec.grid_dim, n)
    dist = GSLIBDistribution(spec)
    state_tuple = rand(dist)
    porosity = state_tuple[1][coords]
    return RockObservations(porosity, coords_array)
end

function sample_initial(spec::POMDPSpecification, coords::Array)
    n = length(coords)
    coords_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(coords)
        coords_array[1, i] = sample[1]
        coords_array[2, i] = sample[2]
    end
    dist = GSLIBDistribution(spec)
    state_tuple = rand(dist)
    porosity = state_tuple[1][coords]
    return RockObservations(porosity, coords_array)
end

function initialize_data!(spec::POMDPSpecification, n::Integer)
    new_rock_obs = sample_initial(spec, n)
    append!(spec.initial_data.porosity, new_rock_obs.porosity)
    spec.initial_data.coordinates = hcat(spec.initial_data.coordinates, new_rock_obs.coordinates)
    return spec
end

function initialize_data!(spec::POMDPSpecification, coords::Array)
    new_rock_obs = sample_initial(spec, coords)
    append!(spec.initial_data.porosity, new_rock_obs.porosity)
    spec.initial_data.coordinates = hcat(spec.initial_data.coordinates, new_rock_obs.coordinates)
    return spec
end

### DO NOT OVERWRITE ###
POMDPs.discount(::CCSPOMDP) = 0.95
POMDPs.isterminal(m::CCSPOMDP, s::CCSState) = size(s.t)[2] >= m.max_n_wells ||
                    s.t >= m.max_placement_t

struct CCSInitStateDist
    reservoir_distribution::GSLIBDistribution
    eng_distribution::EngineeringParameterBelief
end

function POMDPs.initialstate(m::CCSPOMDP)
    reservoir_dist = GSLIBDistribution(m.spec)
    eng_dist = EngineeringParameterBelief(m.spec)
    CCSInitStateDist(reservoir_dist, eng_dist)
end

function Base.rand(d::CCSInitStateDist)
    porosity, permeability, _ = rand(d.reservoir_distribution) # Maps
    srw, src = rand(d.eng_distribution) # Engineering Params
    CCSState(porosity, permeability, nothing, nothing, srw, src);
end

# function gen_reward(s::CCSState, a::CartesianIndex, sp::CCSState)
#     return 0.0
# end # TODO


## Implement for POMDP types
function POMDPs.gen(m::CCSPOMDP, s, a, rng)
    throw("POMDPs.gen function not implemented for $(typeof(m)) type")
    return (sp=sp, o=o, r=r)
end

# function POMDPs.actions(m::CCSPOMDP)
#     throw("POMDPs.actions function not implemented for $(typeof(m)) type")
#     return actions
# end

# function POMDPs.actions(m::CCSPOMDP, s::CCSState)
#     throw("POMDPs.actions function not implemented for $(typeof(m)) type")
#     return actions
# end

# function POMDPs.actions(m::CCSPOMDP, b)
#     throw("POMDPs.actions function not implemented for $(typeof(m)) type")
#     return actions
# end

# For POMCPOW
# function POMDPModelTools.obs_weight(p::CCSPOMDP, s, a, sp, o)
#     throw("POMDPModelTools.obs_weight function not implemented for $(typeof(m)) type")
#     return weight
# end
