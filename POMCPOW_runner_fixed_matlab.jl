
using Pkg
Pkg.activate()
Pkg.develop(path=".")
using Revise
using CCS
using Random
# 

try using POMDPs catch;Pkg.add("POMDPs"); using POMDPs end
try using POMDPSimulators catch;Pkg.add("POMDPSimulators"); using POMDPSimulators end
try using POMCPOW catch;Pkg.add("POMCPOW"); using POMCPOW end
try using POMDPPolicies catch;Pkg.add("POMDPPolicies"); using POMDPPolicies end
try using Random catch;Pkg.add("Random"); using Random end
try using BeliefUpdaters catch;Pkg.add("BeliefUpdaters"); using BeliefUpdaters end
try using Serialization catch;Pkg.add("Serialization"); using Serialization end
try using Infiltrator catch;Pkg.add("Infiltrator"); using Infiltrator end
try using KernelDensity catch;Pkg.add("KernelDensity"); using KernelDensity end
try using Distributions catch;Pkg.add("Distributions"); using Distributions end
try using Distances catch;Pkg.add("Distances"); using Distances end
try using LinearAlgebra catch;Pkg.add("LinearAlgebra"); using LinearAlgebra end
try using Statistics catch;Pkg.add("Statistics"); using Statistics end
try using PyCall catch;Pkg.add("PyCall"); using PyCall end
# try using MATLAB catch;Pkg.add("MATLAB"); using MATLAB end

using Serialization

# function initializeMATLABSim()
#     println("Initializing MATLAB session...")
#     session = MSession(0)
#     eval_string(session, "addpath('matlab')")
#     eval_string(session, "clear")
#     eval_string(session, "addpath('/home/ccs/mrst-2022b/')")
#     eval_string(session, "addpath('/home/ccs/mrst-2022b/solver_ccs/')")
#     eval_string(session, "startup")
#     eval_string(session, "startuplocal_custom")
#     println("MATLAB Session Opened")
#     return session
# end



widening_pairs = [2.0 5.0;0.3 0.7;2.0 5.0;0.3 0.7];
all_pairs = reduce(hcat, collect(x) for x in Iterators.product(eachrow(widening_pairs)...));

for countt in 1:1
    for (idxx, ele) in enumerate(eachcol(all_pairs))
        if idxx != 10
            # idxx = 15
            # countt = 1
            # ele = all_pairs[:,15]
            # load_conformal_quantiles()
            DIMS = (80, 80, 1)
            N_INITIAL = 0
            MAX_WELLS = 3
            # MRST_DIR = "/home/jmern91/mrst-2021a"
            # MRST_DIR = "/Users/markuszechner/Documents/MATLAB/mrst-2020b"

            spec = POMDPSpecification(grid_dim=DIMS, max_n_wells=MAX_WELLS)
            Random.seed!(1234+countt-1)
            gslib_dist = GSLIBDistribution(spec)
            porosity, permeability, _ = rand(gslib_dist);
            init_rock_obs = CCS.sample_initial(spec, N_INITIAL)
            init_well_coords = init_rock_obs.coordinates
            eng_para_dist = EngineeringParameterBelief(spec)
            srw, src = rand(eng_para_dist)
            srw = 0.27
            src = 0.20
            s0 = CCSState(porosity, permeability, nothing, nothing, srw, src);
            GC.gc()
            m = SurrogatePOMDP2D(spec)
            up = ReservoirBeliefUpdater(spec)
            b0 = initialize_belief(up, s0)
            solver = POMCPOWSolver(criterion=MaxUCB(20.0),tree_queries = 100, k_action = ele[1],alpha_action = ele[2], k_observation = ele[3], alpha_observation = ele[4])
            planner = POMDPs.solve(solver, m)
            # planner = RandomPolicy(m)
            V = 0.0
            t = 0
            println("Entering Simulation...")
            t_start = time_ns()
            @show s0.well_coords

            save_dir_prefix = string("/home/ccs/POMCPOW_new_model_march_seed",countt,"index",idxx,"/")
            if !isdir(save_dir_prefix)
                mkdir(save_dir_prefix)
            end


            Serialization.serialize(string(save_dir_prefix,"b0.dat"),b0)




            while !isterminal(m, s0)
                # global b0
                # global s0
                # global t
                # global V
                # session = initializeMATLABSim()
                a = action(planner, deepcopy(b0))
                sp, o, r = gen_matlab(s0, a)
                #sp, o, r = gen(m, s0, a)
                # close(session)
                bp = update(up, b0, a, o)

                if isnothing(sp.well_coords)

                    Serialization.serialize(string(save_dir_prefix,"bp",t,".dat"),bp)
                    Serialization.serialize(string(save_dir_prefix,"r",t,".dat"),r)


                else
                    surrogate_output = run_surrogate(s0, a, sp)
                    # session = initializeMATLABSim()
                    trapped_fractions,free_fractions,excited_fractions,observation_sat,observation_poro = run_matlab(s0, a, sp)
                    # close(session)

                    Serialization.serialize(string(save_dir_prefix,"bp",t,".dat"),bp)
                    Serialization.serialize(string(save_dir_prefix,"a",t,".dat"),a)
                    Serialization.serialize(string(save_dir_prefix,"sp",t,".dat"),sp) #porosity maop
                    Serialization.serialize(string(save_dir_prefix,"sat",t,".dat"),surrogate_output.gs_inject_satmap)
                    Serialization.serialize(string(save_dir_prefix,"matlab_trapp",t,".dat"),trapped_fractions)
                    Serialization.serialize(string(save_dir_prefix,"matlab_free",t,".dat"),free_fractions)
                    Serialization.serialize(string(save_dir_prefix,"matlab_exited",t,".dat"),excited_fractions)
                    Serialization.serialize(string(save_dir_prefix,"matlab_obs_poro",t,".dat"),observation_poro)
                    Serialization.serialize(string(save_dir_prefix,"mass",t,".dat"),surrogate_output.gs_inject_mass) 
                    Serialization.serialize(string(save_dir_prefix,"pos_sat",t,".dat"),surrogate_output.gs_post_satmap)
                    Serialization.serialize(string(save_dir_prefix,"post_mass",t,".dat"),surrogate_output.gs_post_mass) 
                    Serialization.serialize(string(save_dir_prefix,"obs_well",t,".dat"),sp.obs_well_coord)
                    Serialization.serialize(string(save_dir_prefix,"r",t,".dat"),r)
                end
                s0, b0 = sp, bp
                t +=1
                tp = clamp(t-2, 0, Inf)
                V += POMDPs.discount(m)^tp*r
                GC.gc()
            end


            t_end = time_ns()
            dt = (t_end - t_start)/1.0e9
            println("Discounted Return: $V")
            println("Total Runtime: $dt seconds")
            Serialization.serialize(string(save_dir_prefix,"discountedreward.dat"),V)
        end 
                
    end
end



