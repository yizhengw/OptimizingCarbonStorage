using Revise

using POMDPs
using POMDPPolicies
using CCS

CCS.py"""
from ccs.gas_saturation_injection_period import *
"""
# CCS.initialize_surrogate()

DIMS = (80, 80, 1)
N_INITIAL = 0
MAX_WELLS = 3

spec = POMDPSpecification(grid_dim=DIMS, max_n_wells=MAX_WELLS)
initialize_data!(spec, N_INITIAL)
m = SurrogatePOMDP2D(spec)
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

a = CartesianIndex(40, 40)
result = POMDPs.gen(m, s0, a, m.rng)

a = CartesianIndex(30, 30)
result = POMDPs.gen(m, result.sp, a, m.rng)
