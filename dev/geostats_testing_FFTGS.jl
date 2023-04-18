# Used for parameterizing, sampling, and estimating from distribution

using GeoStats
using Parameters
using RandomNumbers
using Distances
using Random
using Plots

@with_kw mutable struct RockObservations
    porosity::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end


struct CCSState #TOO: Add "engineering parameters", add boolean for whether each well is injecting or not
    porosity::Array{Float64}  # 3D array of porosity values for each grid-cell
    perm_xy::Array{Float64} # 3D array of permeability values for each grid-cell
    perm_z::Array{Float64} # 3D array of permeability values for each grid-cell
    # pressure::Array{Float64} # 3D array of pressure values for each grid-cell
    # saturation::Array{Float64} # 3D array of C02 saturation values for each grid-cell
    # parameter::Float64
    well_coords::Matrix{Int64} # 2D grid cell location of each well
end

@with_kw struct ReservoirDistribution # Only a distribution over the rock properties right now
    grid_dims::Tuple{Int64, Int64} = (80, 80)
    data::RockObservations = RockObservations()
    domain::CartesianGrid{2, Int64} = CartesianGrid{Int64}(80, 80)
    radius::Float64 = 10.0
    poro_mean::Float64 = 0.2
    Az::Float64 = 1300.0
    Axy::Float64 = 1000.0
    poro_variogram::Variogram = SphericalVariogram( sill=0.001, range=20.0,
                                            nugget=0.0001)
    neighborhood::NormBall{Float64, Euclidean} = NormBall(10.0)
end

function Base.rand(rng::AbstractRNG, d::ReservoirDistribution)
    # data_dict = OrderedDict(:porosity=>d.data.porosity)
    # geodata = PointSet(data_dict, d.data.coordinates)
    if isempty(d.data.coordinates) # Unconditional simulation
        problem = SimulationProblem(d.domain, (:porosity => Float64), 1)
    else # Conditional simulation
        # coordinates = convert(Array{Int64, 2}, d.data.coordinates)
        table = DataFrame(porosity=d.data.porosity)
        domain = PointSet(d.data.coordinates)
        # geodata = georef((porosity=d.data.porosity), domain)
        geodata = georef(table, domain)
        problem = SimulationProblem(geodata, d.domain, (:porosity), 1)
    end

    solver = FFTGS(:Z => (mean=d.poro_mean,variogram=d.poro_variogram))

    solution = GeoStats.solve(problem, solver)
    porosity = solution[:porosity][1] #clamp.(solution[:porosity][1], 0.01, 0.38)
    temp_poro=reshape(porosity,80,80,1)
    poro = repeat(temp_poro, outer=[1,1,8])
    # porosity = reshape(porosity, d.grid_dims)

    B = (porosity.^3)./(1.0 .- porosity).^2
    Kz = d.Az.*B
    Kxy = d.Axy.*B
    state = CCSState(porosity, Kxy, Kz, d.data.coordinates)
    return state
end

Base.rand(d::ReservoirDistribution) = Base.rand(Random.GLOBAL_RNG, d)
# dd = ReservoirDistribution()
# a=Base.rand(dd)






d = ReservoirDistribution()

if isempty(d.data.coordinates) # Unconditional simulation
    problem = SimulationProblem(d.domain, (:porosity => Float64), 1)
else # Conditional simulation
    # coordinates = convert(Array{Int64, 2}, d.data.coordinates)
    table = DataFrame(porosity=d.data.porosity)
    domain = PointSet(d.data.coordinates)
    # geodata = georef((porosity=d.data.porosity), domain)
    geodata = georef(table, domain)
    problem = SimulationProblem(geodata, d.domain, (:porosity), 1)
end
solver = SGS(
                    :porosity => ( mean=d.poro_mean,
                                variogram=d.poro_variogram,
                                neighborhood=NormBall(d.radius),
                                maxneighbors=10
                                   )
                     )
solution = GeoStats.solve(problem, solver)
porosity = solution[:porosity][1]
#porosity = clamp.(solution[:porosity][1], 0.01, 0.38)
porosity = reshape(porosity, d.grid_dims)
B = (porosity.^3)./(1.0 .- porosity).^2
Kz = d.Az.*B
Kxy = d.Axy.*B
state = CCSState(porosity, Kxy, Kz, d.data.coordinates)


fig = heatmap(state.porosity, fill=true)
display(fig)
