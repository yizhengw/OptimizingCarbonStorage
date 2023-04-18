module CCS

using POMDPs
using POMDPModelTools
using Random
# using GeoStats
using CSV
using Plots
# using MATLAB
using PyCall
using DataStructures
using DataFrames
using Parameters
using StatsBase

using Serialization 
using KernelDensity 
using Distributions 
using Distances
using LinearAlgebra
using Statistics
using MultivariateStats



export
        CCSState,
        CCSObservation,
        SurrogateModelOutput,
        RockObservations
include("common.jl")

# export
#         sample_coords,
#         sample_data,
#         ReservoirDistribution,
#         ReservoirDistributionUpdater
# include("geostats.jl")

export
        GSLIBDistribution
include("gslib.jl")

# export
#         initializeMATLABSim,
#         runMATLABSim
# include("simulator.jl")

export
        RockObservations,
        CCSState,
        CCSObservation,
        CCSPOMDP,
        POMDPSpecification,
        EngineeringParameterBelief,
        initialize_data!
include("pomdp_base.jl")



export
        ReservoirBelief,
        ReservoirBeliefUpdater,
        solve_gp
include("beliefs.jl")

# export
#         SimPOMDP2D,
#         close_sess
# include("2d_pomdp_sim.jl")

include("2d_surrogate.jl")

export
        SurrogatePOMDP2D,
        run_surrogate,
        gen_matlab,
        run_matlab,
        load_conformal_quantiles,
        initialize_surrogate_input,
        run_inj_surrogate,
        initialize_surrogate
include("2d_pomdp_surrogate.jl")

export
        TestPOMDP2D
include("2d_pomdp_test.jl")

end


