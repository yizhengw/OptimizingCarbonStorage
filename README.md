# Optimizing Carbon Storage Operations for Long-Term Safety
Repository defining the problem for the carbon capture and sequestration optimal planning project. Problem is defined as a POMDP using the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.
 

## POMDP Definition
- State Space: Locations of wells + map of the aquifer porosity
- Action Space: Place a well at any unoccupied location in 3D grid (80 × 80 × 8)
- Observation Space: Observe porosity and CO2 saturation at observation well
- Observation Model: Internal to the surrogate simulation  
- Transition Model: Internal to the surrogate simulation 
- Reward Function: $−1000.0 ×m_{exited}−10.0 ×m_{free}+10.0 ×m_{trapped}$
- Intial State Distribution `ReservoirDistribution`: Heirarchical Gaussian process over rock properties. Sampled with approximate sequential Gaussian simulation (SGS). 


## Reservoir Simulator MRST

You can get the MRST simulator here (we are using version 2021a):

https://www.sintef.no/projectweb/mrst/download/

Running the simulation: `simulation_master.m` - instructions and comments are in the files itself.

We used a mac to run the simulations. All instructions in the files are based on macOS - if you use windows you might need to change the path to windows syntex.


# Saving MRST files for python

We are using the following package for saving the MRST results in the npy format

https://github.com/kwikteam/npy-matlab

don't forget to add the path into the matlab file:

e.g.: addpath('/Users/markuszechner/Documents/MATLAB/npy-matlab-master/npy-matlab/')
