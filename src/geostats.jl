# Used for parameterizing, sampling, and estimating from distribution

@with_kw struct ReservoirDistribution # Only a distribution over the rock properties right now
    grid_dims::Tuple{Int64, Int64, Int64} = (80, 80, 1)
    data::RockObservations = RockObservations()
    domain::CartesianGrid{2, Int64} = CartesianGrid{Int64}(80, 80)
    poro_mean::Float64 = 0.2
    Az::Float64 = 1300.0
    Axy::Float64 = 1000.0
    poro_variogram::Variogram = SphericalVariogram(sill=0.001, range=20.0,
                                            nugget=0.0001)
end

function Base.rand(rng::AbstractRNG, d::ReservoirDistribution)
    if isempty(d.data.coordinates) # Unconditional simulation
        problem = SimulationProblem(d.domain, (:porosity => Float64), 1)
    else # Conditional simulation
        table = DataFrame(porosity=d.data.porosity)
        domain = PointSet(d.data.coordinates)
        geodata = georef(table, domain)
        problem = SimulationProblem(geodata, d.domain, (:porosity), 1)
    end
    solver = FFTGS(
                        :porosity => ( mean=d.poro_mean,
                                    variogram=d.poro_variogram
                                       )
                         )
    solution = GeoStats.solve(problem, solver)
    poro_1D = solution[:porosity][1]
    poro_1D = clamp.(poro_1D, 0.1, 0.4)
    # porosity = clamp.(solution[:porosity][1], 0.05, 0.38)
    poro_2D = reshape(poro_1D, d.grid_dims)
    porosity = repeat(poro_2D, outer=(1, 1, 8))
    Kz=Kxy = porosity.^3.0*(1e-5).^2.0./(0.81*72.0*(1.0.-porosity).^2.0)
    # B = (porosity.^3)./(1.0 .- porosity).^2
    # Kz = d.Az.*B
    # Kxy = d.Axy.*B
    rock_state = (porosity, Kxy, Kz)
    return rock_state
end

Base.rand(d::ReservoirDistribution) = Base.rand(Random.GLOBAL_RNG, d)

function solve_gp(d::ReservoirDistribution)
    table = DataFrame(porosity=d.data.porosity)
    domain = PointSet(d.data.coordinates)
    geodata = georef(table, domain)
    problem = EstimationProblem(geodata, d.domain, :porosity)
    solver = Kriging(
                        :porosity => ( mean=d.poro_mean,
                                    variogram=d.poro_variogram
                                       )
                         )
    solution = GeoStats.solve(problem, solver)
    porosity_mean = reshape(solution[:porosity], d.grid_dims)
    porosity_var = reshape(solution[:porosity_variance], d.grid_dims)
    return (porosity_mean, porosity_var)
end
