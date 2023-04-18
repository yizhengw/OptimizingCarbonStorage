using Pkg
Pkg.activate()
Pkg.develop(path=".")
using POMDPs
using CCS
using Random
using POMDPModelTools
using Plots
using MCTS
using Revise
using Serialization
using Statistics
using MultivariateStats

save_dir_prefix = "/home/ccs/Random/"
t = 0
belief = deserialize(string(save_dir_prefix,"b",t,".dat"));
# state = deserialize(string(save_dir_prefix,"sp2.dat"));
# real_poro = state.porosity
ensemble = nothing
for i in 1:100
    global ensemble
    porosity = rand(belief).porosity 
    ensemble = isnothing(ensemble) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble,[reshape(porosity[:,:,1],(1,:))]);
end

ensemble = reshape(reduce(vcat, ensemble), 100, :);
b = heatmap(var(reshape(ensemble,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
display(b)
savefig(b, "/home/ccs/POMCPOW/varrr_diff4.png")



###################
save_dir_prefix = "/home/ccs/Random/"
t = 3
mass = deserialize(string(save_dir_prefix,"mass",t,".dat")); 
cate_mass = vcat(sum(mass[1:6,:],dims=1),reshape(mass[7,:],(1,:)), reshape(mass[8,:],(1,:)));
cate_mass_cumsum = vcat(cumsum(sum(mass[1:6,:],dims=1),dims=2),cumsum(reshape(mass[7,:],(1,:)),dims =2), cumsum(reshape(mass[8,:],(1,:)),dims=2));


mass_post = deserialize(string(save_dir_prefix,"mass_post",t,".dat")); 
cate_mass_post = vcat(sum(mass_post[1:6,:],dims=1),reshape(mass_post[7,:],(1,:)), reshape(mass_post[8,:],(1,:)));

function cumsum_steps(a_list,year_per_step)
    return cumsum(a_list .* year_per_step)
end
cate_mass_post_cumsum = hcat(cumsum_steps(sum(mass_post[1:6,:],dims=1)[1,:], 500/20),cumsum_steps(mass_post[7,:], 500/20), cumsum_steps(mass_post[8,:], 500/20));

plot(1:50, hcat(cate_mass_cumsum,  cate_mass_cumsum[1:3,end] .+ cate_mass_post_cumsum')', label = ["trapped" "free" "exited"],ylim=(0, 1200),legend=:topleft)



plot(1:size(mass,2), (cate_mass)', label = ["trapped" "free" "exited"], legend=:topleft)
plot(1:size(mass,2), (cate_mass_cumsum)', label = ["trapped" "free" "exited"], legend=:topleft)

######################

spec = POMDPSpecification(grid_dim=(80, 80, 1), max_n_wells=3);
pomdp = SurrogatePOMDP2D(spec);

## Get ground truth realization
Random.seed!(1234)
#s = rand(initialstate(pomdp))

## Sample initial state and initialize the belief
gt_up = ReservoirBeliefUpdater(spec, num_MC=80)
#b = initialize_belief(gt_up, s)

## Plot the ground truth porosity
# heatmap(s.porosity[:,:,1])

## Construct the Belief-state mdp
pl_up = ReservoirBeliefUpdater(spec, num_MC=20)
gbmdp = GenerativeBeliefMDP(pomdp, pl_up)
bs0 = rand(initialstate(gbmdp))

## Construct the DPW solver
solver = DPWSolver(n_iterations=10, exploration_constant=20.0)
planner = solve(solver, gbmdp)

# action(planner, bs0)
s = rand(bs0)


save_dir_prefix = "/home/ccs/src/ESMDA_MCTS_test_80/"
## Stepthrough
t = 0
Serialization.serialize(string(save_dir_prefix,"bs0",t,".dat"),bs0)
Serialization.serialize(string(save_dir_prefix,"s",t,".dat"),s)

V = 0.0
println("Entering Simulation...")
t_start = time_ns()
a_list = [CartesianIndex(20 ,30), CartesianIndex(25 ,35), CartesianIndex(18 ,26), CartesianIndex(35 ,40)]
while !isterminal(pomdp, s)
    global bs0
    global s
    global t
    global V

    # a = action(planner, deepcopy(bs0))
    a = a_list[t+1]

    sp, o, r = gen(pomdp, s, a)
    surrogate_output = run_surrogate(s, a, sp)
    bp = update(gt_up, bs0, a, o)

    # save some stuff
    s, bs0 = sp, bp
    t += 1
    tp = clamp(t-2, 0, Inf)
    V += POMDPs.discount(pomdp)^tp*r
    println("a = $a")
    println("r = $r")
    println("t = $t")
    Serialization.serialize(string(save_dir_prefix,"bs0",t,".dat"),bs0)
    Serialization.serialize(string(save_dir_prefix,"a",t,".dat"),a)
    Serialization.serialize(string(save_dir_prefix,"s",t,".dat"),s)
    Serialization.serialize(string(save_dir_prefix,"tree",t,".dat"),planner.tree)
    Serialization.serialize(string(save_dir_prefix,"sat",t,".dat"),surrogate_output.gs_inject_satmap) 
end
t_end = time_ns()
dt = (t_end - t_start)/1.0e9
println("Discounted Return: $V")
println("Total Runtime: $dt seconds")



######PCA score viz


save_dir_prefix = "/home/ccs/src/ESMDA_MCTS_test_80/"
t = 0
belief = deserialize(string(save_dir_prefix,"bs0",t,".dat"));
ensemble = nothing
for i in 1:300
    global ensemble
    porosity = rand(belief).porosity
    ensemble = isnothing(ensemble) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble,[reshape(porosity[:,:,1],(1,:))]);
end
ensemble = reshape(reduce(vcat, ensemble), 300, :);
# b = heatmap(var(reshape(ensemble,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
# display(b)
M = MultivariateStats.fit(PCA, convert(Matrix,ensemble'));
pc_scores_prior = MultivariateStats.transform(M, convert(Matrix,ensemble'));
scatter(pc_scores_prior[:,1], pc_scores_prior[:,2], label = "prior")



t = 4
belief = deserialize(string(save_dir_prefix,"bs0",t,".dat"));
ensemble = nothing
for i in 1:300
    global ensemble
    porosity = rand(belief).porosity
    ensemble = isnothing(ensemble) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble,[reshape(porosity[:,:,1],(1,:))]);
end
ensemble = reshape(reduce(vcat, ensemble), 300, :);
# b = heatmap(var(reshape(ensemble,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
# display(b)
M = MultivariateStats.fit(PCA, convert(Matrix,ensemble'));
pc_scores_pos = MultivariateStats.transform(M, convert(Matrix,ensemble'));

scatter!(pc_scores_pos[:,1], pc_scores_pos[:,2], label = "posterior")


######## plot time 
plot([20,40,80,100,150,200,250],[7.05, 13.72, 27.19, 33.91, 50.78, 67.54, 87.19] .* 3 ./60, marker=(:d), label = "n_iterations = 10", legend=:topleft, xaxis = ("# MC in ESMDA"), yaxis = ("time (hours)"))
plot!([20,40,80,100,150,200,250],[7.05, 13.72, 27.19, 33.91, 50.78, 67.54, 87.19] .* 15 ./60, marker=(:d), label = "n_iterations = 50")
plot!([20,40,80,100,150,200,250],[7.05, 13.72, 27.19, 33.91, 50.78, 67.54, 87.19] .* 30 ./60, marker=(:d), label = "n_iterations = 100")



##########
## Construct the POMDP Specification and POMDP
spec = POMDPSpecification(grid_dim=(80, 80, 1), max_n_wells=3);
pomdp = SurrogatePOMDP2D(spec);

## Get ground truth realization
Random.seed!(1234)
#s = rand(initialstate(pomdp))

## Sample initial state and initialize the belief
gt_up = ReservoirBeliefUpdater(spec, num_MC=200)
#b = initialize_belief(gt_up, s)

## Plot the ground truth porosity
# heatmap(s.porosity[:,:,1])

## Construct the Belief-state mdp
pl_up = ReservoirBeliefUpdater(spec, num_MC=20)
gbmdp = GenerativeBeliefMDP(pomdp, pl_up)
global bs0 = rand(initialstate(gbmdp))
s = rand(bs0)

# ensemble2 = nothing
# for i in 1:100
#     global ensemble2
#     porosity = rand(bs0).porosity
#     ensemble2 = isnothing(ensemble2) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble2,[reshape(porosity[:,:,1],(1,:))]);
# end

# ensemble2 = reshape(reduce(vcat, ensemble2), 100, :);
# b = heatmap(var(reshape(ensemble2,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
# display(b)



## Construct the DPW solver
solver = DPWSolver(n_iterations=10, exploration_constant=20.0)
planner = solve(solver, gbmdp)
a = action(planner, deepcopy(bs0))

save_dir_prefix = "/home/ccs/src/ESMDA_MCTS/"
t = 0 
Serialization.serialize(string(save_dir_prefix,"tree",t,".dat"),planner.tree)

# spec = POMDPSpecification(grid_dim=(80, 80, 1), max_n_wells=3);
# pomdp = SurrogatePOMDP2D(spec);

# ## Get ground truth realization
# Random.seed!(1234)
# times = []
# for ele in [20,40]
#     global times
#     t_start = time_ns()
#     pl_up = ReservoirBeliefUpdater(spec, num_MC=ele)
#     gbmdp = GenerativeBeliefMDP(pomdp, pl_up)
#     solver = DPWSolver(n_iterations=100, exploration_constant=20.0)
#     planner = solve(solver, gbmdp)
#     bs0 = rand(initialstate(gbmdp))
#     a = action(planner, deepcopy(bs0))
#     t_end = time_ns()
#     dt = (t_end - t_start)/1.0e9
#     push!(times, dt)
# end



ensemble3 = nothing
for i in 1:100
    global ensemble3
    porosity = rand(bs0).porosity
    ensemble3 = isnothing(ensemble3) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble3,[reshape(porosity[:,:,1],(1,:))]);
end

ensemble3 = reshape(reduce(vcat, ensemble3), 100, :);
b = heatmap(var(reshape(ensemble3,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
display(b)

sp, o, r = gen(pomdp, s, a)
bp = POMDPs.update(gt_up, bs0, a, o)
ensemble3 = nothing
for i in 1:100
    global ensemble3
    porosity = rand(bp).porosity
    ensemble3 = isnothing(ensemble3) ? vcat([reshape(porosity[:,:,1],(1,:))]) : vcat(ensemble3,[reshape(porosity[:,:,1],(1,:))]);
end

ensemble3 = reshape(reduce(vcat, ensemble3), 100, :);
b = heatmap(var(reshape(ensemble3,(100,80,80)), dims =1 )[1,:,:], clims=(0,0.007))
display(b)