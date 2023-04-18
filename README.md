# CCS-core
Repository defining the problem for the carbon capture and sequestration optimal planning project. Problem is defined as a POMDP using the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

## Setup
TODO

## Example Usage
TODO

## POMDP Definition
- State Space `NamedTuple{(:porosity, :Kx, :Kxy, :facies, :coords), Tuple{Array{Float64}, Array{Float64}, Array{Float64}, Array{Int64}, Array{Int64}}`: Completely defines the rock properties of the reservoir with the porosity, vertical permeability, horizontal permeability, and facies-type arrays. Also defines locations of existing well coordinates. TODO: Add fluid state. 
- Action Space `Array{Int64}`: List of all the grid coordinates at which a well may be placed. Excludes all coodrinates within specified "keepout" distance from a previously placed well. TODO: Add shut-off actions (?)
- Observation Space `Tuple{Float64, Int64}`: The porosity and the facies type at the drilled well location, observed upon well placement.  TODO: Add fluid observations
- Observation Model: Noise free observations of porosity and facies type at well grid location. 
- Transition Model: Wells placed determinsitically at specified location. Rock properties constant. Fluid properties updated by flow simulation (?). 
- Reward Function: TODO
- Intial State Distribution `ReservoirDistribution`: Heirarchical Gaussian process over rock properties. Sampled with approximate sequential Gaussian simulation (SGS). Initial fluid properties are fixed TODO: Change (?). Built around the [GeoStats.jl](https://github.com/JuliaEarth/GeoStats.jl) framework.

### Included Belief Models
TODO

## Folder Structure
- `ccs`: Python source code
- `src`: Julia source code
- `matlab`: Matlab source code
- `scripts`: Scripts executing source code instances, e.g. for experiments

## Development Use
Please push the finalized functions/files into this repository. If you have an ad-hoc or one-off script, please push to `CCS-scratch`.

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
