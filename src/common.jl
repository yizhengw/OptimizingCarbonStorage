struct CCSState # TODO: Add "engineering parameters", add boolean for whether each well is injecting or not
    porosity::Array{Float64}  # 3D array of porosity values for each grid-cell
    permeability::Array{Float64} # 3D array of permeability values for each grid-cell
    # pressure::Array{Float64} # 3D array of pressure values for each grid-cell
    # saturation::Array{Float64} # 3D array of C02 saturation values for each grid-cell
    # parameter::Float64
    well_coords::Union{Nothing, Matrix{Int64}} # 2D grid cell location of each injection well
    obs_well_coord::Union{Nothing, Vector{Int64}}

    srw::Float64
    src::Float64
end

struct SurrogateModelOutput
    gs_inject_satmap::Union{Nothing, Array{Float32, 5}}
    gs_inject_mass::Union{Nothing, Matrix{Float32}}
    p_inject::Union{Nothing, Array}
    gs_post_satmap::Union{Nothing,  Array{Float32, 5}}
    gs_post_mass::Union{Nothing,  Matrix{Float32}}
    p_post::Union{Nothing,  Array}
end

@with_kw mutable struct RockObservations
    porosity::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

function Base.length(obs::RockObservations)
    return length(obs.porosity)
end

struct CCSObservation
    pressure_history::Vector{Float64} # Pressure observed at observation well since last step
    saturation_history::Vector{Float32} # CO2 saturation observed at observation well since last step
    porosity::Float64
end
