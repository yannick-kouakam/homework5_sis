import numpy as np
import matlibplot as mp
from utils.utils import GaussianDistribution

class KalmanFilter:
    """docstring for .
    """
    def __init__(self, initial_state,
                       state,
                       control,
                       observation,
                       state_noise,
                       obs_noise):

        self._predicted =initial_state
        self._updated =initial_state
        self._state =np.array(state)
        self._control = np.array(control)
        self._observation =np.array(observation)
        self._state_noise = np.array(state_noise)
        self._obs_noise = np.array(obs_noise)


        def predictionStep(self,control):

            control = np.array(control)

            prior_mean = self._state.dot(self._updated.mean) + self._control.dot(control)
            cov_by_state = self._state.(self._update.covariance).dot(self._state.transpose())

            prior_covariance = cov_by_state + self._state_noise.covariance

            self._predicted = GaussianDistribution(prior_mean,prior_covariance)









        def update_step(self,observations):
            pass





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
