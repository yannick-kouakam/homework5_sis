import numpy as np
import matlibplot as mp


def kalman_filter(self,transition,observation):
    """
    kamlan filter implemantation
    data is Guassian distribution of a linear function

    Parameter:
    transition: the transition matrix from the model
    observation : data from sensors
    """
    s_data = np.array(observation)
    t_data = np.array(transition)
    n_iter = s_data.shape[0]
    x_t = np.zeros(n_iter)
    z_t = np.zeros(n_iter)

    ## initialization
    x_t[0] =observation[0]

    for i in range(1,n_iter-1):
        K[i] = 0 #
        x_t[i] = 0 ##
    return (x_t,z_t)

def extended_kalman_filter(self,transition,observation):
    pass
