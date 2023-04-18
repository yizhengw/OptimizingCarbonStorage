using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Statistics
using Plots

using CCS

DIMS = (20, 20, 1)
COORDS = [CartesianIndex((1, 1, 1)), CartesianIndex((1, 10, 1)), CartesianIndex((5, 1, 1))]
N = 10

spec = POMDPSpecification(grid_dim=DIMS, max_n_wells=3)
initialize_data!(spec, COORDS)
spec.initial_data.porosity[:] .= 0.3
m = TestPOMDP2D(spec)
ds0 = POMDPs.initialstate_distribution(m)

S = Array{Float64}[]
for i in 1:N
    s = rand(ds0)
    push!(S, s.porosity[:,:,1])
end

mean_field = mean(S)
std_field = std(S)

fig = heatmap(mean_field, title="Mean Porosity Field", fill=true, clims=(0.1, 0.6))
savefig(fig, "./sgsim_mean.png")
display(fig)

fig = heatmap(std_field, title="StdDev Porosity Field", fill=true) #, clims=(0.1, 0.6))
savefig(fig, "./sgsim_stdev.png")
display(fig)
