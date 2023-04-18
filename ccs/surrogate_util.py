import numpy as np

def initialize_input(k_map):
    n = 1
    idx = 0
    # input
    # INJ_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t

    K_MAP = np.zeros((n, 80, 80, 8))
    K_MAP[0,...] =  k_map
    # K_MAP[idx,:,:,:,:] = np.repeat(np.expand_dims(k_map, -1), 30, -1)

    grid_x = np.linspace(0, 1, 80)
    grid_x = grid_x[np.newaxis,:,np.newaxis,np.newaxis, np.newaxis]

    grid_y = np.linspace(0, 1, 80)
    grid_y = grid_y[np.newaxis,np.newaxis,:,np.newaxis, np.newaxis]

    grid_z = np.linspace(0, 0.1, 8)
    grid_z = grid_z[np.newaxis,np.newaxis,np.newaxis,:, np.newaxis]

    # dt = 31556952
    grid_t = np.linspace(0, 1, 30)

 
    # grid_t *= dt
    # grid_t = np.load(f'sim_1/time_days.npy')[1:31,0]
    grid_t /= np.max(grid_t)
    grid_t = grid_t[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

    x_inj = np.zeros((n, 80, 80, 8, 30, 6))
    x_inj[...,0] = K_MAP[...,None]
    # x[...,1] = INJ_MAP
    x_inj[...,2] = grid_x
    x_inj[...,3] = grid_y
    x_inj[...,4] = grid_z
    x_inj[...,5] = grid_t
    return x_inj


def add_well(x_array,coords): # Input np array, well number, coordinates
    if len(coords.shape) == 1:
        coords = coords.reshape(-1,1)

    for idx in range(coords.shape[1]):
        x = coords[:,idx][0] - 1
        y = coords[:,idx][1] - 1
        if idx == 0:
            x_array[0, x, y, :, : , 1] = 1
        else:
            dt = 10 # TODO
            start_t = dt*idx - 1
            x_array[0, x, y, :, start_t: , 1] = 1

    return x_array



def initialize_post_input(k_map):

    n = 1
    idx = 0
    K_MAP = k_map

    grid_x = np.linspace(0, 1, 80)
    grid_x = grid_x[np.newaxis,:,np.newaxis,np.newaxis, np.newaxis]

    grid_y = np.linspace(0, 1, 80)
    grid_y = grid_y[np.newaxis,np.newaxis,:,np.newaxis, np.newaxis]

    grid_z = np.linspace(0, 0.1, 8)
    grid_z = grid_z[np.newaxis,np.newaxis,np.newaxis,:, np.newaxis]

    # dt = 31556952
    grid_t = np.arange(0,20).astype(np.float64)
    # grid_t *= dt
    # grid_t = np.load(f'sim_1/time_days.npy')[1:31,0]
    grid_t /= np.max(grid_t)
    grid_t = grid_t[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

    x_post_inj = np.zeros((n, 80, 80, 8, 20, 6))
    x_post_inj[...,0] = K_MAP[None,...,None]
    x_post_inj[...,2] = grid_x
    x_post_inj[...,3] = grid_y
    x_post_inj[...,4] = grid_z
    x_post_inj[...,5] = grid_t
    return x_post_inj