"""
This code solves for the temperature and concentration profile of catalytic cracking of 
ortho-xylene.
To run the code, you manually changes Tin and Twall, calculate the Peclet's number.
Use particle Peclet number and correlation to find Dr and conductivity
"""

import numpy as np                # Matrix operation
import matplotlib.pyplot as plt   # For plotting
import math                       # For pi and exponential 

#%% Given information
Tin = 600                       # Inlet temperature, K
Twall = 652                      # Wall temperature, K
P = 2.2*101325                    # Inlet pressure, Pa
dp = 0.003                        # Pellet diameter, m
R = 0.05/2                        # Tube radius, m
L = 5                             # Tube length, m
eps = 0.6                         # Void fraction
u = 1                             # Superficial velocity, m/s
mu = 0.00003                      # Visocity, Pa*s
Cp = 0.237*4184                   # Heat capacity, J/kgK
Heat = 307*4184                 # Heat of reaction, J/mol
M = [0.10616,0.032,0.028]         # Molar mass of OX,O2,N2, kg/mol
Ctot = 101325/8.314/273.15         # Total concentration, mol/m^3
yin = np.divide([0.044/M[0],(Ctot-0.044/M[0])*0.21,(Ctot-0.044/M[0])*0.79],Ctot) # Inlet mol fraction
COin = (Ctot-0.044/M[0])*0.21      # Inlet oxygen concentration, mol/m^3
rho = sum(yin*P/8.314/Tin*M)      # Density, kg/m^3
Re = rho*dp*u/mu                  # Particle reynold number
Dr = 3.8E-4                       # Mass dispersion coefficient, m^2/s
kr = 0.4*4.184                    # Radial conductivity, J/m/s/K
Er = kr/rho/Cp                    # Thermal diffusivity, m^2/s
hr = 3*kr/dp/(Re)**0.25           # Wall heat transfer coefficient, J/m^2/s/K
delH =Heat/Cp/rho                 # Adiabatic temperature rise

#%% Initialization
II = 32                           # Number of radial increments
dr = R/II                         # radius increment, m
dzmax = dr**2*u/4/Er              # maximum length increment, m
jj = int(L/dzmax)+1               # Number of length increments
dz = L/jj                         # length increment, m
z = np.linspace(0,5,jj+1)         

Ca = np.zeros((II+1,jj+1))    
T = np.zeros((II+1,jj+1))      
Ca[:,0] = yin[0]*P/8.314/Tin      #Concentration initial condition
T[:,0] = Tin                      #Temperature initial condition

#%% Euler step
for j in range(1,jj+1):
# At centerline
  rate = math.exp(20.348-13636/T[0,j-1])*Ca[0,j-1]	# reaction rate at center
  dCadz = (4*Dr/u/dr**2)*(Ca[1,j-1]-Ca[0,j-1]) - eps/u*rate
  Ca[0,j] = Ca[0,j-1] + dCadz*dz  	# update Ca
  dTdz = (4*Er/u/dr**2)*(T[1,j-1]-T[0,j-1]) + eps*delH/u*rate
  T[0,j] = T[0,j-1] + dTdz*dz   	# update T
  
# Everywhere else
  for i in range(1,II):                   
    rate = math.exp(20.348-13636/T[i,j-1])*Ca[i,j-1] 
    dCadz = (Dr/u/dr**2)*((1/2/i+1)*Ca[i+1,j-1]-2*Ca[i,j-1]+(1-1/2/i)*Ca[i-1,j-1]) - eps/u*rate
    Ca[i,j] = Ca[i,j-1] + dCadz*dz		# update Ca
    dTdz = (Er/u/dr**2)*((1/2/i+1)*T[i+1,j-1]-2*T[i,j-1]+(1-1/2/i)*T[i-1,j-1]) + eps*delH/u*rate
    T[i,j] = T[i,j-1] + dTdz*dz   	# update T
    
#Wall boundary conditions
  Ca[-1,j] = (18*Ca[-2,j]-9*Ca[-3,j]+2*Ca[-4,j])/11	  #Concentration BC
  T[-1,j] = (6*dr*hr*Twall+18*kr*T[-2,j]-9*kr*T[-3,j]+2*kr*T[-4,j])/(6*dr*hr+11*kr) 
 	#Temperature BC
  
#%% Post-processing
plt.plot(z, T[0,:]) #Plotting centerline temperature
plt.title('Centerline Temperature Along Reactor Length')
plt.xlabel('Tube position, m')
plt.ylabel('Temperature, K')
plt.axis([0,5,300,max(T[0,:])+50])

#Computing conversion using mixing cup average
ri=np.linspace(0,R,II+1) #Discretizing radial space
Caout=2*dr*sum(Ca[:,-1]*ri)/R**2 #Mixing-cup average approximated as Riemann sum
COout=COin-3*(Ca[0,0]-Caout) #Oxygen concentration at outlet
Ratio=COout/Caout #For 5B
X=1-Caout/Ca[0,0] #Conversion for #4

Tmax = max(T[0,:]) #maximum centerline temperature, K
print('Maximum centerline temperature =', "%.1f" %Tmax,'K')
print('Conversion =', "%.3f" %X)