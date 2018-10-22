
import numpy as np
import matplotlib.pyplot as plt

##### Parameter/setup ######

alpha=1
sigma=1
x0=5 # initial condition

tmax=10.0 #max time

np.random.seed(101) # for reproducability 

############################

plt.rcParams['figure.figsize'] = [6, 6]


def simulate(timestep):
    
    times=np.arange(0.0,tmax,timestep) # vector of times
    length=np.shape(times)[0] # Number of timesteps
    x=np.zeros(length) # vector to store x values
    x[0]=x0

    B=np.random.randn(length-1)*np.sqrt(timestep) # the gaussian noise. Builds an array of random numbers in distr]ibution, not necessary to do it this way.

    for i in range(0,length-1): # simulate
        x[i+1]=x[i]-alpha*x[i]*timestep+sigma*B[i] # Can just draw a random number here every time, remember to mutiply by np.sqrt(dt)
    
    return times, x

times_1, xs_1 = simulate(0.1)
times_2, xs_2 = simulate(0.01)

plt.figure(figsize = (20,8))
plt.plot(times_1,xs_1,'green',lw=2, label=r"Time steps $\Delta t = 0.1$")
plt.plot(times_2,xs_2,'blue',lw=2, label=r"Time steps $\Delta t = 0.01$")
plt.legend(fontsize=20)
plt.xlabel('$t$', fontsize = 24)
plt.ylabel('$X_t$', fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'Ornstein-Uhlenbeck process. $\alpha$ = {}, $\sigma$ = {}.'.format(alpha,sigma), fontsize = 26)
plt.savefig('ou_process_2.png')

plt.show()

