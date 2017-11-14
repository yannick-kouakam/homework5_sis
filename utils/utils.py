#!/usr/bin/python
from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from scipy.stats import norm, multivariate_normal
import math
from random import random


def read_data(path):

    file = open(path,'r')
    lines = file.readlines()
    data = [float(x) for x in lines]
    index = 0
    result = []
    while index < len(data):
        temp = data[index:index+6]
        result.append(temp)
        index+=6
    return result

def calculate_position_fromsensordata(data):
    pass
def read(path):

    file = open(path,'r')
    lines = file.readlines()
    data = []

    for line in lines:
         line = line[0:len(line)-2]
         data.append([float(x) for x in line.split(',',5)])
    return data



def process_model(sensordata):

    x= sensordata[0][4]
    y = sensordata[0][5]
    theta = sensordata[0][3]
    current_pos = [x,y,theta]
    data_minus = sensordata[0]
    model =[]
    observation = []

    for data in sensordata:
        dsl = data[4] - data_minus[4]
        dsr = data[5] - data_minus[5]

        u = [((dsl+dsr)/2)*cos(current_pos[2]+((dsr-dsl)/(2*0.12))),
             ((dsl+dsr)/2)*sin(current_pos[2]+((dsr-dsl)/(2*0.12))),
             (dsr-dsl)/(0.12)]



        upd_pos = np.sum([current_pos,u],axis=0)
        x_obs = -0.5 +(1)*random()
        y_obs = -0.5 +(1)*random()
        theta_obs = -0.5 +(1)*random()
        new_obs = np.sum([upd_pos,[x_obs,y_obs,theta_obs]],axis=0)
        current_pos=upd_pos
        model.append(upd_pos)
        observation.append(new_obs)
        data_minus=data

    return (np.array(model),np.array(observation))







class GaussianDistribution(object):
    """docstring for gaussianDistribution."""

    def __init__(self, mean:np.array,covariance:np.array):

        self._mean = mean
        self._covariance = covariance

    @property
    def covariance(self):
        return self._covariance

    @property
    def mean(self):
        return self._mean



    def to_cov(X,n):
        """
        checks if X is a scalar in what case it returns a covariance
        matrix generated from it as the identity matrix mutiplied
        by X. the dimension will be n*n.
        If X is already a numpy array then it is returned unchanged.
        """

        try:
            X.shape
            if type(X)!= np.array:
                X=np.array(x)[0]
            return X
        except :
            cov = np.array(X)
            try:
                len(cov)
                return cov
            except :
                return np.eye(n) * X




    def gaussian(X,mean,var):
        """
        compute the normal distribution of x with the mean mean
         and the varirance var
        """

        return (np.exp((-0.5*(np.asarray(X)-mean)**2)/var)/ math.sqrt(2*math.pi*var))


    def mutivariate_gaussian(X,mean,cov):
        X= np.array(X,copy=False,ndmin=1).flatten()
        mean= np.array(mean,copy=False,ndmin=1).flatten()

        nx = len(mean)
        cov = to_cov(cov, nx)

        norm_coef = nx*math.log(2*math.pi)  + np.linalg.slogdet(cov)[1]

        error = X - mean

        if(sp.issparse(cov)):
            numerator= spln.spsolve(cov , error).T.dot(error)
        else:
            numerator  = np.linalg.solve(cov ,error).T.dot(error)

        return math.exp(-0.5*(norm_coef + numerator))





#
# data = read("../dataset.txt")
# X,X1 = process_model(data)
# x = [data[0] for data in X]
# y = [data[1] for data in X]
#
# x1 = [item[0] for item in X1]
# y1 = [item[1] for item in X1]
# plt.plot(x,y,label='position',c='b')
# plt.plot(x1,y1,label='observation',c='r')
# plt.legend()
# plt.show()
