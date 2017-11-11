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
import matplotlib.pyplot as plt
import numpy as np
import math


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



def process_model(sensordata):

    x= sensordata[0][1]
    y = sensodata[0][2]
    theta = sensodata[0][4]
    current_pos = [x,y,theta]
    data_minus = sensodata[0]
    result =[]

    for data in sensordata:
        dsl = data[1] - data_minus[1]
        dsr = data[2] - data_minus[2]
        u = [((dsl+dsr)/2)*cos(current_pos[2]+((dsr-dsl)/(2*0.12))),
             ((dsl+dsr)/2)*sin(current_pos[2]+((dsr-dsl)/(2*0.12))),
             (dsr-dsl)/(0.12)]
        upd_pos = current_pos+u
        current_pos=upd_pos
        result.append(upd_pos)
        data_minus=data

    return np.array(result)







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

print("begin")
xs = np.arange(15,30,0.05)
plt.plot(xs,gaussian(xs,23,0.05),label='$\sigma^2$=0.05',c='b')
plt.plot(xs, gaussian(xs, 23, 1), label='$\sigma^2$=1', ls=':', c='b')
plt.plot(xs, gaussian(xs, 23, 5), label='$\sigma^2$=5', ls='--', c='b')
plt.legend()
plt.show()
