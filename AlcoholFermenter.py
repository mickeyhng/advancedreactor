"""


This code solves for concentration of biomass, substrate, and ethanol in batch 
fermentation of s. cerevisiae yeast and the batch time for brewing American 
and real beer.

@author: admin
"""
import numpy as np #For array               
import matplotlib.pyplot as plt  #For plotting 
from scipy.integrate import odeint #Solving set of diffEq

#%%Parameters used in model
mumax = 0.5 # maximum specific growth rate [per hour]
Ks = 0.001 # emperical constant for substrate inhibition 
pmax = 0.109 # maximum ethanol concentration [g C/g broth]
Cpfake=0.04/1.25*.78 #fake beer alcohol concentration [g C/g broth]
Cpreal=.06/1.25*.78 #real beer alcohol concentration [g C/g broth]
Ms = 0.008 #Maintenance coefficient for s. cerevisiae [per hour]

#%%Defining the set of differential equations
def brewmaster(C,t):
    X=C[0] #biomass concentration g C/g broth
    s=C[1] #substrate concentration g C/g broth
    p=C[2] #product concentration g C/g broth
    if s>1e-8: #when s is greater than 0, use the following set of diffeq
        dXdt=mumax*X*(s/(Ks+s))*(1-p/pmax)
        dsdt=-dXdt-Ms*X
        dpdt=dXdt/2+Ms*X
    else: #when s = 0, use following set of diffeq
        dXdt=-Ms*X
        dsdt=0
        dpdt=0
    dCdt=[dXdt,dsdt,dpdt]
    return dCdt

#%% Initial conditions [g C/g broth]
CX0=0.0005 #Initial biomass concentration
Cs0=0.24*0.4 #Initial substrate concentration
Cp0=0 #Initial product concentration

Co=[CX0,Cs0,Cp0] #Initial condition array

#%% Allocating time points
to=0 #initial time [hour]
tend=20 #ending time [hour]

h=0.001 #time step
n=int(tend/h+1)
tspan=np.linspace(to,tend,n) #Creating timespan

#%% Solving differential equations
C=odeint(brewmaster, Co, tspan) 

#To set Cp = 0 in case we get negative substrate concentration
for i in range (len(tspan)):
    if C[i,1]>0:
        C[i,1]=C[i,1]
    else:
        C[i,1]=0

#%% Post-processing
#To extract the time point for fake and real beer alcohol concentration
#Extracting the index when the product concentration array is just above 
#the required beer concentration        
for i in range (len(tspan)):
    if C[i,2]>Cpfake:
        fakeindex=i
        break

for i in range (len(tspan)):
    if C[i,2]>Cpreal:
        realindex=i
        break

#%%Plotting  
plt.figure(1)
plt.plot(tspan,C[:,0],label='Biomass')
plt.plot(tspan,C[:,1],label='Substrate')
plt.plot(tspan,C[:,2],label='Product')
plt.plot(tspan[fakeindex],Cpfake,'ro',label='4% ABV')
plt.plot(tspan[realindex],Cpreal,'bo',label='6% ABV')
plt.axis([0,20,0,0.1])
plt.legend(loc='center left')
plt.xlabel('Reaction time (hour)')
plt.ylabel('Dimensionless concentration (g carbon/g broth)')
plt.title('Batch reactor concentration for s. cerevisiae fermentation')

print('Theoretical fake beer batch time is %1.0f hours' %tspan[fakeindex])
print('Theoretical real beer batch time is %1.0f hours' %tspan[realindex])    

