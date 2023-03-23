#https://www.youtube.com/watch?v=8ZZDNd4eyVI&t=1090s
#https://docs.sciml.ai/DiffEqDocs/dev/tutorials/sde_example/

import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

## définition des symboles (pas utilisé)

t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function) #true value



the1 = the1(t)      #définition de the1 et the2 en fonction du temps

the2 = the2(t)           


the1_d = smp.diff(the1, t)      
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)   
the2_dd = smp.diff(the2_d, t)

## fin de la symbiologie

x1 = L1*smp.sin(the1)                   
y1 = -L1*smp.cos(the1)  
x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)



# Kinetic
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)   
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T_1 = T1+T2



# Potential
V1 = m1*g*y1                    
V2 = m2*g*y2
V_1 = V1 + V2



# Lagrangian
L_1 = T_1-V_1                   


LE1 = smp.diff(L_1, the1) - smp.diff(smp.diff(L_1, the1_d), t).simplify()   
LE2 = smp.diff(L_1, the2) - smp.diff(smp.diff(L_1, the2_d), t).simplify()



sols = smp.solve([LE1, LE2], (the1_dd, the2_dd),                    
                simplify=False, rational=False)


dz1dt_f_1 = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
dz2dt_f_1 = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])
dthe1dt_f_1 = smp.lambdify(the1_d, the1_d)
dthe2dt_f_1 = smp.lambdify(the2_d, the2_d)
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f_1(z1),
        dz1dt_f_1(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f_1(z2),
        dz2dt_f_1(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]


n_times = 1001
t = np.linspace(0, 200, n_times)
g = 9.81
m1=2
m2=1
L1 = 2
L2 = 1
y0=[1, -3, -1, 5]#y0 = conditions initlae de dSdT => the1, z1(=vitesse angulaire), the2 et z2
ans_1 = odeint(dSdt, y0, t=t, args=(g,m1,m2,L1,L2)) #"true value of hypothetical model"




#matrix of initial condition 

n_perturbation=10       #number of runs
init=np.vstack([y0,y0]) #creating a matrix full of y0 of the amount a runs wanted
for i in range (n_perturbation-2):
    init=np.vstack([init,y0])

        
#creating 4 stochastic perturbation for each parameters
np.random.seed(0)       #seed to have at each run the same numbers 
e1 = np.random.uniform(-0.01,0.01,n_perturbation)
e2 = np.random.uniform(-0.03,0.03,n_perturbation)
e3 = np.random.uniform(-0.05,0.05,n_perturbation)
e4 = np.random.uniform(-0.001,0.001,n_perturbation)

#creating a matrix of stochastic pertubations
E=(np.vstack([e1,e2,e3,e4])).transpose()

cond_init= init+E #matrix from initial condition and stochastic noise

res_array=np.zeros(shape=(n_perturbation,n_times))

#finding answers at each timestep for each initial conditions
for i in range (n_perturbation):
    answ=odeint(dSdt, cond_init[i,:], t=t, args=(g,m1,m2,L1,L2))
    the2_all=answ.T[2]
    res_array[i,:]=the2_all
    plt.plot(t,the2_all)




the1 = ans_1.T[0]
the2 = ans_1.T[2]

#plot with n_pertubation runs
plt.plot(t, the2,"-r", label="hypothetic true value", linewidth=4)
plt.xlabel('time')
plt.ylabel('the2')
plt.legend()
plt.show()


#plot for the true value of the hypothetic value only
plt.plot(t, the2, "-b", label="hypothetic true value")
plt.xlabel('time')
plt.ylabel('the2')
plt.title('Chaotic behavior of the double pendulum')
plt.legend()
plt.show()


plt.plot(t, the2, "-b", label="hypothetic true value")
plt.plot(t,res_array[4,:],"-r", label="hypothetic small errors in initial conditions")
plt.xlabel('time')
plt.ylabel('the2')
plt.title('Chaotic behavior of the double pendulum')
plt.legend()
plt.show()

mean_all=np.zeros(shape=(1,n_times)) #creating array full of 0 shaped for each timesptep
max_all=np.zeros(shape=(1,n_times))
min_all=np.zeros(shape=(1,n_times))
for i in range (n_times):
    mean=np.mean(res_array[:,i])    #calculation the mean value of all runs for a timestep
    mean_all[:,i]=mean          #filling the array with mean value for each timestep
    maxi=np.max(res_array[:,i]) #calculation of the 95zh quantile for i timestep
    max_all[:,i]=maxi      #filling the array of 95th quantile for each timestep
    mini=np.min(res_array[:,i])
    min_all[:,i]=mini

plt.plot(t, the2, "-b", label="hypothetic true value")
plt.plot(t,np.transpose(mean_all),"-g", label="mean value of the ensemble")
plt.plot(t,np.transpose(max_all),"-r", label="upper limit of the ensemble")
plt.plot(t,np.transpose(min_all),"-r", label="lower limit of the ensemble")
plt.xlabel('time')
plt.ylabel('the2')
plt.title('Visualisation of ensemble model')
plt.legend()



