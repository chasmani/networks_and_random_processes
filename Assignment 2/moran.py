import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(1234)

def am_done(v):
    '''Return true if all elements of v the same. Else false.'''
    u=np.unique(v) # unique  elements of v
    if u.shape[0]>1:
        return False
    else:
        return True


L=40

#################

# X doesn't have a fixed dimension, depends on how long times are
# So just append to it as you go
X=np.zeros((1,L),dtype='int') # initialize (we will use np.append to append to this)
X[0,:]=np.arange(L)

time=0.0
times_list=[] # list of times for sample path

rate=L 
β=1.0/rate # mean wait time, uniformly picking 1/L individual at rate 1

while True: # loop until done 
    wait=np.random.exponential(β) # wait time
    times_list.append((time,time+wait))
    time+=wait
    
    # X is a matrix of types?
    old=X[-1,:] # index -1 is useful. In this case it gives the last row
    new=old.copy() # make a copy
    
    type_to_spread=np.random.choice(old) 

    # Works slightly different to maths - can target own position here, in maths only others
    idx=np.random.choice(np.arange(L))
    new[idx]=type_to_spread # infect    
    new=np.sort(new)
    
    if(am_done(new)):
        break    
        
    X=np.append(X,[new],axis=0) # Add to the 
    
X=np.append(X,[new],axis=0) # add this for completeness 
times_list.append((time,time+2*wait))

plt.figure()
pcm = sns.heatmap(X,cbar_kws={'label': r'Individual $j \in U[0,{}]$'.format(L)})
pcm.figure.axes[-1].yaxis.label.set_size(16)
plt.title('$X_t$ for fixed L={} individuals'.format(L), fontsize = 20)
plt.xlabel('Individuals, L', fontsize = 20)
plt.ylabel('Timestep ', fontsize = 20)

#plt.show()

start_times=[]
for i in range(len(times_list)):
    start_times.append(times_list[i][0])

def X_to_N(X):
    N=np.zeros_like(X)
    length=np.shape(N)[0]
    L=np.shape(N)[1]
    for row in range(length):
        row_list=list(X[row,:])
        for i in range(L):
            N[row,i]=row_list.count(i)   
    return N

N=X_to_N(X)


# Count up individuals at each time step, and match with how long each timestep was
# Gives a plot with continuous time
plt.figure()
for i in range(L):
    plt.plot(start_times,N[:,i],lw=2)
   
plt.xlabel('Time, $t$', fontsize = 20)
plt.ylabel('$N_t$', fontsize= 20)

plt.title("$N_t$ Number of individuals of each species \n out of total population size L = {}".format(L), fontsize = 16)
plt.savefig('moran.png')

plt.show()