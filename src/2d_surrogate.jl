function initialize_surrogate(model_path::String="/home/ccs/src/CCS-core/ccs/models/")

    
    sg_inj_path = string("/home/ccs/src/CCS-core/ccs/models/", "ensumble_sg_inj.pt")
    sg_post_inj_path = string(model_path, "ensumble_post_sg.pt")
    dp_inj_path = string(model_path, "dp_inj_model_renamed.pt")
    dp_post_inj_path = string(model_path, "dp_post_inj_model_renamed.pt")


    # Include current directory on python path (only do it once)
    if !any(map(p->occursin("CCS-core/ccs/models/", p), CCS.pyimport("sys")."path"))
        pushfirst!(PyVector(pyimport("sys")["path"]), abspath(joinpath(@__DIR__, "..")))
        pushfirst!(PyVector(pyimport("sys")["path"]), abspath(joinpath(@__DIR__, "..", model_path)))
        pushfirst!(PyVector(pyimport("sys")["path"]), abspath(joinpath(@__DIR__, "..", "/home/ccs/src/CCS-core/ccs/")))
    end

    py"""
    # import torch.fft
    import numpy as np
    import torch
    import sys
    from sg_inj import *
    from post_sg_inj import *
    from surrogate_util import *

    if 'sg_inj_model' in locals():
        del sg_inj_model
    if 'dp_inj_model' in locals():
        del dp_inj_model
    if 'sg_post_inj_model' in locals():
        del sg_post_inj_model
    if 'dp_post_inj_model' in locals():
        del dp_post_inj_model

    # sg_inj_model = torch.load($sg_inj_path)
    # dp_inj_model = None # torch.load($dp_inj_path)
    # # sg_post_inj_model = torch.load($sg_post_inj_path)
    # sg_post_inj_model = None
    # #sg_post_inj_model = None
    # dp_post_inj_model = None # torch.load($dp_post_inj_path)
    """
    # return (py"sg_inj_model", py"dp_inj_model", py"sg_post_inj_model", py"dp_post_inj_model")
end

function initialize_surrogate_input(s::CCSState)
    # TODO remove if these become integrated in state eventually
    c_rock  = 4.35e-5
    srw     = 0.27
    src     = 0.20
    pe      = 5.0
    muw     = 8e-4
    # srw = s.srw
    # src = s.src
    k_map = s.porosity
    py"""
    x_inj = initialize_input($k_map)
    x_post_inj = initialize_post_input($k_map)
    """
    return (py"x_inj", py"x_post_inj")
    
end



function run_inj_surrogate(coords::Matrix{Int64})
    sg_inj_path = string("/home/ccs/src/CCS-core/ccs/models/", "ensumble_sg_inj.pt")
    py"""    
    import numpy as np
    import torch
    import sys
    from sg_inj import *
    from surrogate_util import *


    if 'sg_post_inj_model' in locals():
        del sg_post_inj_model
    if 'sg_inj_model' in locals():
        pass
    else:
        sg_inj_model = torch.load($sg_inj_path)


    add_well(x_inj, $coords)
    x = x_inj.astype(np.float32)
    
    x = torch.from_numpy(x)

    device = torch.device('cuda:0')
    # xx = x.to(device)
    sg_inj_model = sg_inj_model.to(device)

    with torch.no_grad():
        sg_output = sg_inj_model(x.to(device))
        # dp_output = dp_inj_model(xx) # TODO: Uncomment and use.
    # sg_output = sg_output.numpy()
    """

    # sg_output = py"sg_output"
    # dp_output = py"dp_output" # TODO: Uncomment and us.

    #Array{Float32, 6} size: 1 80 80 8 30 1
    sg_output_satmap = py"sg_output[0].cpu().detach().numpy()"

    #Matrix{Float32}  size: 8 30
    sg_output_mass = py"sg_output[1].cpu().detach().numpy()"  

    dp_output = nothing
    return (sg_output_satmap, sg_output_mass, dp_output)
end






function run_post_surrogate()
    sg_post_inj_path = string("/home/ccs/src/CCS-core/ccs/models/", "ensumble_post_sg.pt")
    py"""
    import numpy as np
    import torch
    import sys
    from post_sg_inj import *
    # Use predicted sg to configure post injection input

    if 'sg_inj_model' in locals():
        del sg_inj_model
    if 'sg_post_inj_model' in locals():
        pass
    else:
        sg_post_inj_model = torch.load($sg_post_inj_path)
 
    sg_pred, mass_pred = sg_output
    pred_plot = sg_pred.cpu().detach().numpy()

    x_post_inj[...,1] = pred_plot[...,-20:].copy()
    x_post = x_post_inj.astype(np.float32)
    x_post = torch.from_numpy(x_post)



    #######################################################################
    # # grid_t = np.load(f'sim_1/time_days.npy')[31:,0]
    # grid_t = np.arange(31,51).astype(np.float64)   # check this
    # grid_t /= np.max(grid_t)
    # grid_t = grid_t[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

    # # x_inj = x.cpu().numpy()
    # x_post = x_inj[...,10:30,:].copy() # 1*80*80*8*30*11
    # x_post[...,1:2] = pred_plot[:,:,:,:,-20:,np.newaxis].copy() # 1*80*80*8*20*11
    # # print("looooooooook here: ", pred_plot.shape)
    # x_post[...,10] = grid_t.copy()

    # x_post = x_post.astype(np.float32)
    # x_post = torch.from_numpy(x_post) # <==== post inj input

    #######################################################################

    #######################################################################
    # Run model
    device = torch.device('cuda:0')
    xx_post = x_post.to(device)


    sg_post_inj_model = sg_post_inj_model.to(device)
    with torch.no_grad():
        sg_post_output = sg_post_inj_model(xx_post)
        # dp_post_output = dp_post_inj_model(xx_post) # TODO: Uncomment and use.
    """
    sg_post_output = py"sg_post_output"
    # dp_post_output = py"dp_post_output" # TODO: Uncomment and use.

    #Array{Float32, 6} size: 1 80 80 8 20 1
    sg_post_output_satmap = py"$sg_post_output[0].cpu().detach().numpy()"

    #Matrix{Float32} size 8 20
    sg_post_output_mass = py"$sg_post_output[1].cpu().detach().numpy()"


    dp_post_output = nothing
    return (sg_post_output_satmap, sg_post_output_mass, dp_post_output)
end

# pymodel = py"""
#     model = torch.load('models/sg_inj_model.pt')
#     """
# py"""
#     x = torch.zeros((1, 80, 80, 8, 30, 11)).to(torch.device('cuda:0'))
#     with torch.no_grad():
#         a, b = model(x)
#     """
# output = py"a"


# pymodel.forward(xinput)
