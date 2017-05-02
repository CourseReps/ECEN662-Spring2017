# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:32:50 2017

@author: Kiran
"""
###################################################################
import numpy as np
import matplotlib.pyplot as plt
import math as m

###################################################################
# Kalman Filter Function
# Code Based on the Equations in Wikipedia
# http://en.wikipedia.org/wiki/Kalman_filter
def kalmanfilter(x,P,measurement,R,U,B,Q,F,H):
    """ 

    Prediction
    x[k+1] = F*x[k] + B*U    
    
    
    Parameters
    ----
    x : N-length numpy vector
        prior state
    P : N x N numpy matrix
        prior covariance
    measurement : m-length numpy vector
        measurement with process noise added
    R : m x m numpy matrix
        measurement covariance
    U : p x 1 numpy matrix
        deterministic system input
    B : n x p numpy matrix
        influence matrix for deterministic states
    Q : N x N numpy matrix
        process noise covariance matrix
    F : N x N numpy matrix
        state influence matrix
    H : m x N numpy matrix
        measurement influence matrix
    """

    # Predict
    
    # Predicted (a priori) state estimate
    x = F*x + B*U
    # Predicted (a priori) estimate covariance
    P = F*P*F.T + Q
    
    # Update
    
    # Innovation or measurement residual
    y = np.matrix(measurement).T - H * x
    # Innovation (or residual) covariance
    S = H * P * H.T + R
    # Optimal Kalman gain
    K = P * H.T * S.I
    # Updated (a posteriori) state estimate
    x = x + K*y
    # Updated (a posteriori) estimate covariance
    I = np.matrix(np.eye(F.shape[0]))
    P = (I - K*H)*P

    return x, P

###################################################################
#Initializing Variables for Projectile Trajectory
# discretization time
DT = 0.01
# final time
Tf = 14.4
# Initial Velocity
v = 100.0
# Angle
angle = 45.0
# initial vertical velocity
vy0 = v*m.sin(angle*m.pi/180.0)
#vy0 = 85.0
# initial horizontal velocity
vx0 = v*m.cos(angle*m.pi/180.0)
#vx0 = 62.5
# initial position
y0 = 0.0
x0 = 0.0
# gravity constant
g = 9.81
# prior covariance
P0 = np.matrix(np.eye(4))

N = int(Tf/DT) + 1
T = np.linspace(0.0,Tf,N)

# analytical solution
# y history
vyt = vy0 - g*T
yt = y0 + vy0*T - 0.5*g*T*T

# x history
vxt = np.ones(N)*vx0
xt = x0+vx0*T

###################################################################
#Adding Noise to the Variables for simulating Noisy Measurements of 
#sensors of Projectile
r=10;
R = np.diag([r,r,r,r]);

x_obs=[];
y_obs=[];
vxt_obs = [];
vyt_obs = [];

for k in range(N):
    wk = np.random.multivariate_normal(mean=np.zeros((4,)),cov=R);
    x_obs.append(xt[k]+wk[0])
    y_obs.append(yt[k]+wk[3])
    vxt_obs.append(vxt[k]+wk[1])
    vyt_obs.append(vyt[k]+wk[3])

###################################################################
#Declaring Kalman Filter Variables and applying Kalman Filter 
#for the Noisy Data
H = np.matrix('''1. 0. 0. 0.;
                 0. 1. 0. 0.;
                 0. 0. 1. 0.;
                 0. 0. 0. 1.''')

B = np.matrix('''0.;
                 0.;
                 0.;
                 1.''')

x = np.matrix('0. 0. 0. 0.').T 
x[1]=vx0;
x[3]=vy0;

P = P0 # initial uncertainty

#Q = np.matrix(np.eye(4)*m.pow(DT,1.0))
Q = np.matrix(np.diag([m.pow(DT,0.25),DT,DT,DT]))

result = []

i=0;
Pcov = np.zeros((len(T),4,4))
for meas in zip(x_obs,vxt_obs,y_obs,vyt_obs):
    #U = np.matrix('0. 0. 0. 0. ').T
    U = np.matrix('0.').T
    U[0] = -g*DT
    #U[2]=-0.5*g*DT*DT;
    #U[3]=-g*DT;
    F = np.matrix('''1. 0. 0. 0.;
                 0. 1. 0. 0.;
                 0. 0. 1. 0.;
                 0. 0. 0. 1.''')
    F[0,1] = DT
    F[2,3] = DT
    # add the current value of x to list
    result.append((x[:4]).tolist())
    # add current covariance
    Pcov[i,:,:] = P.copy()
    x, P = kalmanfilter(x, P, meas, R,U,B, Q, F, H)
    i=i+1;

kalman_x,kalman_vxt_obs,kalman_y,kalman_vyt_obs = zip(*result)

# print error statistics
print("Mean errors:")
print("%8s%8s%8s%8s" % ('x','y','vx','vy'))
print("%8.4f%8.4f%8.4f%8.4f" % (np.mean(np.array(kalman_x) - xt),np.mean(np.array(kalman_y) - yt),np.mean(np.array(kalman_vxt_obs) - vxt),np.mean(np.array(kalman_vyt_obs) - vyt) ))

###################################################################
'''
#Comparing The True, Measured, Kalman Filtered Data
# plot
plt.figure()
plt.plot(xt,yt,label='Truth')
plt.plot(x_obs,y_obs,'ro',label='Measured')
plt.plot(kalman_x, kalman_y,'g-',label='Filter')
plt.legend(loc='upper right')
plt.xlabel('X [t]')
plt.ylabel('Y [t]')
plt.title('Trajectory Comparison')
plt.grid()
plt.tight_layout()
plt.show()

# plot
plt.figure()
plt.plot(T,vyt,label='Truth')
plt.plot(T,vyt_obs,'ro',label='Measured')
plt.plot(T,kalman_vyt_obs,'g-',label='Filter')
plt.legend(loc='upper right')
plt.title('$V_y$ Comparison')
plt.grid()
plt.tight_layout()
plt.show()

# plot
plt.figure()
plt.plot(T,vxt,label='Truth')
plt.plot(T,vxt_obs,'ro',label='Measured')
plt.plot(T,kalman_vxt_obs,'g-',label='Filter')
plt.legend(loc='upper right')
plt.title('$V_x$ Comparison')
plt.grid()
plt.tight_layout()
plt.show()
'''

# 3 sigma plots
plt.figure(figsize=(8,5))

plt.subplot(221)
plt.plot(T,xt-np.array(kalman_x).flatten())
plt.plot(T, 3.0*np.sqrt(Pcov[:,0,0]),'r--')
plt.plot(T,-3.0*np.sqrt(Pcov[:,0,0]),'r--')
plt.title('X position error history')
plt.grid()

plt.subplot(222)
plt.plot(T,yt-np.array(kalman_y).flatten())
plt.plot(T, 3.0*np.sqrt(Pcov[:,2,2]),'r--')
plt.plot(T,-3.0*np.sqrt(Pcov[:,2,2]),'r--')
plt.title('Y position error history')
plt.grid()

plt.subplot(223)
plt.plot(T,vxt-np.array(kalman_vxt_obs).flatten())
plt.plot(T, 3.0*np.sqrt(Pcov[:,1,1]),'r--')
plt.plot(T,-3.0*np.sqrt(Pcov[:,1,1]),'r--')
plt.xlabel('time (sec)')
plt.title('X velocity error history')
plt.grid()

plt.subplot(224)
plt.plot(T,vyt-np.array(kalman_vyt_obs).flatten())
plt.plot(T, 3.0*np.sqrt(Pcov[:,3,3]),'r--')
plt.plot(T,-3.0*np.sqrt(Pcov[:,3,3]),'r--')
plt.xlabel('time (sec)')
plt.title('Y velocity error history')
plt.grid()

plt.tight_layout()

plt.show()

from scipy import linalg
# normalize the variables
Z = np.zeros((len(T),4))
for k in range(len(T)):
    C = linalg.sqrtm( np.linalg.inv(Pcov[k,:,:]) )
    #Xv = np.array([[kalman_x[k]],[kalman_y[k]],[kalman_vxt_obs[k]],[kalman_vyt_obs[k]]])
    Xv = np.array([kalman_x[k],kalman_y[k],kalman_vxt_obs[k],kalman_vyt_obs[k]])
    Xtrue = np.array([[xt[k]],[yt[k]],[vxt[k]],[vyt[k]]])
    Z[k,:] = (np.dot(C,Xv-Xtrue)).transpose()

print(np.mean(Z,axis=0))
print(np.std(Z,axis=0))
    
plt.figure(figsize=(8,5))

for k in range(4):
    plt.subplot(2,2,k+1)
    plt.hist(Z[:,k],20,normed=True)
    plt.title(r'$\mu = %8.4g, \sigma = %8.4g$' % (np.mean(Z[:,k]),np.std(Z[:,k])))
    plt.xlabel(r'$z_%d$' % (k+1))
    plt.grid()

plt.tight_layout()

plt.show()