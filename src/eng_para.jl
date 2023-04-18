struct EngineeringParameterBelief
    srw_belief::Vector{Float64} # use kernel density to resample new data based on this 
    src_belief::Vector{Float64} # use kernel density to resample new data based on this 
end

function EngineeringParameterBelief(s::POMDPSpecification)
    return EngineeringParameterBelief(srw_belief=rand(Uniform(s.srw_bounds[1],s.srw_bounds[2]),s.num_samples_eng_para), src_belief=rand(Uniform(s.src_bounds[1],s.src_bounds[2]),s.num_samples_eng_para))

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