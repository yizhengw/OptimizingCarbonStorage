
struct SimPOMDP2D <: CCSPOMDP
    spec::POMDPSpecification
    max_n_wells::Int64 # From spec, for convenience
    max_placement_t::Float64 # From spec, for convenience
    initial_data::RockObservations # Initial rock observations
    delta::Int64 # Minimum distance between wells
    injection_rate::Float64 # Fixed well injection rate
    matlab_sess::MSession # Bound Matlab session call (auto-generated)
    rng::AbstractRNG
    function SimPOMDP2D(s::POMDPSpecification, mrst_dir::String, print_matlab::Bool=false; rng::AbstractRNG=Random.GLOBAL_RNG)
        matlab_sess = initializeMATLABSim(s.grid_dim[1], s.grid_dim[2], 8, #  8 = number vertical cells
        # s.reservoir_dims[1], s.reservoir_dims[2], s.reservoir_dims[3],
        # 365, s.simulation_t,
        mrst_dir, print_matlab)
        new(s, s.max_n_wells, s.max_placement_t, s.initial_data, s.delta,
        s.injection_rate, matlab_sess, rng)
    end
end

close_sess(m::SimPOMDP2D) = MATLAB.close(m.matlab_sess)

function POMDPs.gen(m::SimPOMDP2D, s::CCSState, a::CartesianIndex, rng::AbstractRNG)
    poro_array = s.porosity
    perm_array = s.permeability
    o_porosity = poro_array[a]
    n_initial = length(m.initial_data)
    if s.obs_well_coord == nothing
        println("No injection wells present. Skipping simulation...")
        o = CCSObservation(Float64[], Float64[], o_porosity)
        r = 0.0
        coords_p = s.well_coords
        obs_well_p = [a[1]; a[2]]
    else
        coords_p = hcat(s.well_coords, [a[1]; a[2]])
        inj_well_coords = hcat(coords_p[:, n_initial+1:end], s.obs_well_coord)
        r_t = size(inj_well_coords)[2] - 1
        println("Starting MATLAB simulation...")
        # println("Well Locations: $(s.c)")
        t1 = time_ns()
        schedule_idx, injector_bhps, mass_fractions, time_days,
        pressure_map_first_layer, observation_sat = runMATLABSim(m.matlab_sess,
                                        poro_array, perm_array, inj_well_coords)
        t2 = time_ns()
        dt = (t2 - t1)/1.0e9
        println("Elapsed: $dt (s)")
        println("MATLAB simulation complete.")
        o = CCSObservation(Float64[], Float64[], o_porosity)
        r = MRST_reward(schedule_idx, injector_bhps, mass_fractions, time_days,
                                    pressure_map_first_layer, observation_sat)
        if r_t == m.max_n_wells
            r = r[end]
        else
            r = r[r_t]
        end
        obs_well_p = s.obs_well_coord
    end
    sp = CCSState(s.porosity, s.permeability, coords_p, obs_well_p)
    return (sp=sp, o=o, r=r)
end

function POMDPs.actions(m::SimPOMDP2D)
    spacing = m.spec.grid_spacing + 1
    idxs = CartesianIndices(m.spec.grid_dim)[1:spacing:end, 1:spacing:end]
    reshape(collect(idxs), prod(size(idxs)))
end

function POMDPs.actions(m::SimPOMDP2D, s::CCSState)
    action_set = Set(POMDPs.actions(m))
    for i=1:size(s.well_coords)[2]
        coord = s.well_coords[:, i]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

function POMDPs.actions(m::SimPOMDP2D, b::ReservoirBelief)
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

POMDPs.discount(::SimPOMDP2D) = 0.99

function POMDPs.isterminal(m::SimPOMDP2D, s::CCSState)
    n_initial = length(m.initial_data)
    n_wells = size(s.well_coords)[2] - n_initial
    return n_wells >= m.max_n_wells
end

function POMDPModelTools.obs_weight(p::SimPOMDP2D, s::CCSState,
                            a::CartesianIndex, sp::CCSState, o::CCSObservation)
    o_true = sp.porosity[a]
    if o_true == o
        return 1.0
    else
        return 0.0
    end
end
