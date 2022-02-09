import math #For pi and exponential function
import matplotlib.pyplot as plt #For plotting
import numpy as np #For matrices and vectors 

Po = 11*101325 #inlet pressure [Pa]
T = 820+273.15 #operating temperature [K]
RJ = 8.314 #R(in J/mol·K),gas constant in [J/mol·K]
Rkcal = 1.99e-3 #gas constant [kcal/mol] 
nEo = 99 #initial molar flow rate of ethane [mol/s]
nYo = 1 #initial molar flow rate of ethylene [mol/s]
yWo = 0.4 #mole fraction of steam
d = 0.1 #reactor diameter [m]
Ac = math.pi*d**2/4 #cross-sectional area of the reactor [m^2]
L = 85 #length of the reactor [m]

#Defining initial molar flux [mol/m^2/s]
NEo = nEo/Ac #initial molar flux of ethane
NYo = nYo/Ac #initial molar flux of ethylene
NHo = 0 #initial molar flux of hydrogen
NPo = 0 ##initial molar flux of propane
NMo = 0 #initial molar flux of methane
NLo = 0 #initial molar flux of propylene
NAo = 0 #initial molar flux of acetylene
NBo = 0 ##initial molar flux of 1,3-butadiene
NWo = yWo/(1 - yWo)*(nEo + nYo)/Ac #initial molar flux of steam
No = np.array([NEo, NYo, NHo, NPo, NMo, NLo, NAo, NBo, NWo])

#Molar mass [g/mol]
ME = 30.069 #molar mass of ethane
MY = 28.0532 #molar mass of ethylene
MH = 2.01588 #molar mass of hydrogen
MP = 44.0956 #molar mass of propane
MM = 16.0425 #molar mass of methane
ML = 42.0797 #molar mass of propylene
MA = 26.0373 #molar mass of acetylene
MB = 54.0904 #molar mass of 1,3-butadiene
MW = 18.0153 #molar mass of steam
root_Ms = np.power([ME, MY, MH, MP, MM, ML, MA, MB, MW], 0.5) #For Wilke's model

##########################################################################
#Viscosities calculation using aT^2/(1+c/T+d/T^2) where a,b,c,d are coefficients
#obtained from DIPPR data base [Pa*s]

mufunc = lambda a, b, c, d: a*T**b/(1+c/T+d/T**2)
muE = mufunc(2.5906e-7, 0.67988, 98.902, 0) #viscosity of ethane
muY = mufunc(2.0789e-6, 0.4163, 352.7, 0) #viscosity of ethylene
muH = mufunc(1.797e-7, 0.685, .59, 140) #viscosity of hydrogen
muP = mufunc(4.9054e-8, 0.90125, 0, 0) #viscosity of propane
muM = mufunc(5.2546e-7, 0.59006, 105.67, 0) #viscosity of methane
muL = mufunc(7.3919e-7, 0.5423, 263.73, 0) #viscosity of propylene
muA = mufunc(1.2025e-6, 0.4952, 291.4, 0) #viscosity of acetylene
muB = mufunc(2.696e-7, 0.6715, 134.7, 0) #viscosity of 1,3-butadiene
muW = mufunc(1.7096e-8,1.1146, 0, 0) #viscosity of steam
mus = np.array([muE, muY, muH, muP, muM, muL, muA, muB, muW])
muo = np.sum(No*mus*root_Ms)/np.sum(No*root_Ms) #composition-dependent viscosity by Wilke definition

#Temperature-dependent rate constant calculated from 
#Arrhenius equation: k=ko*exp(-Ea/RT) where ko is preexponential factor and 
# Ea is activation energy. Ea given here are in [kcal]

Aheq = lambda ko, Ea: ko*math.exp(-1*Ea/Rkcal/T)
k1f = Aheq(4.652e13, 65.20) #[1/s]
k1r = Aheq(8.75e8/1000, 32.7) #[m^3/mol/s]
k2 = Aheq(3.85e11, 65.25) #[1/s]
k5f = Aheq(9.814e8, 36.92) #[1/s]
k5r = Aheq(5.87e4/1000, 7.043) #[m^3/mol/s]
k6 = Aheq(1.026e12/1000, 41.26) #[m^3/mol/s]
k8 = Aheq(7.083e13/1000, 60.43) #[m^3/mol/s]

#############################################################################
#Initialization

zo = 0 #initial length of the reactor [m]
zmax = L #final length of the reactor [m]

h = 0.01 #step size [m]

z = np.linspace(zo,zmax,int(zmax/h+1)) #evenly divide the space

#Vector initialization
N  = np.zeros([9,len(z)])
v  = np.zeros(len(z))
P  = np.zeros(len(z))
mu = np.zeros(len(z))


#Initial conditions
N[:,0] = No #initial flux of all species
P[0] = Po #Pressure at the entrance
v[0] = RJ*T/Po*np.sum(No) #initial velocity
mu[0] = muo

#Coefficients in RK4 method
m1 = np.zeros([9,len(z)])
m2 = np.zeros([9,len(z)])
m3 = np.zeros([9,len(z)])
m4 = np.zeros([9,len(z)])

#############################################################
#RK4 algorithmn
for i in range(len(z)-1):

    #Redefining the variable for easier coding, don't really need NW,NB,NP   
    NE = N[0,i]
    NY = N[1,i]
    NH = N[2,i]
    NP = N[3,i]
    NM = N[4,i]
    NL = N[5,i]
    NA = N[6,i]
    NB = N[7,i]
    NW = N[8,i]
    u  = v[i]
    
# m1 calculation   
#    Order of difeq: dNE/dz, dNY/dz, dNH/dz, dNP/dz, dNM/dz,
    #dNL/dz,dNA/dz,dNB/dz,dNW/dz
    m1[:,i] = np.multiply(
               [(k1r*NY*NH - k1f*NE*u - 2*k2*NE*u - k8*NY*NE)/u**2,
                (k1f*NE*u - k1r*NH*NY - k6*NA*NY - k8*NE*NY)/u**2,
                (k1f*NE*u - k1r*NY*NH)/u**2,                                    
                k2*NE/u,
                (k2*NE + k5f*NL*u - k5r*NM*NA + k8*NY*NE)/u**2,
                (k5r*NM*NA - k5f*NL*u + k8*NY*NE)/u**2,
                (k5f*NL*u - k5r*NM*NA - k6*NY*NA)/u**2,
                k6*NY*NA/u**2,
                0],h)   
 # m2 calculation   
 # Update the molar fluxes first, it is easier to write them out separately than 
 # writing them into your RK coefficient matrices 
 
    NE = NE + 0.5*m1[0,i]      
    NY = NY + 0.5*m1[1,i]
    NH = NH + 0.5*m1[2,i]
    NP = NP
    NM = NM + 0.5*m1[4,i]
    NL = NL + 0.5*m1[5,i]
    NA = NA + 0.5*m1[6,i] 
    NB = NB
    NW = NW
 
    #m2 coefficients update
    m2[:,i] = np.multiply(
               [(k1r*NY*NH -k1f*NE*u - 2*k2*NE*u - k8*NY*NE)/u**2,
                (k1f*NE*u - k1r*NH*NY - k6*NA*NY - k8*NE*NY)/u**2,
                (k1f*NE*u - k1r*NY*NH)/u**2,                                    
                k2*NE/u,
                (k2*NE + k5f*NL*u - k5r*NM*NA + k8*NY*NE)/u**2,
                (k5r*NM*NA - k5f*NL*u + k8*NY*NE)/u**2,
                (k5f*NL*u - k5r*NM*NA - k6*NY*NA)/u**2,
                k6*NY*NA/u**2,
                0],h)   
    
    #m3 coefficients calculation
    
    NE = NE + 0.5*m2[0,i]      
    NY = NY + 0.5*m2[1,i]
    NH = NH + 0.5*m2[2,i]
    NP = NP
    NM = NM + 0.5*m2[4,i]
    NL = NL + 0.5*m2[5,i]
    NA = NA + 0.5*m2[6,i] 
    NB = NB
    NW = NW
    
    m3[:,i] = np.multiply(
               [(k1r*NY*NH -k1f*NE*u - 2*k2*NE*u - k8*NY*NE)/u**2,
                (k1f*NE*u - k1r*NH*NY - k6*NA*NY - k8*NE*NY)/u**2,
                (k1f*NE*u - k1r*NY*NH)/u**2,                                    
                k2*NE/u,
                (k2*NE + k5f*NL*u - k5r*NM*NA + k8*NY*NE)/u**2,
                (k5r*NM*NA - k5f*NL*u + k8*NY*NE)/u**2,
                (k5f*NL*u - k5r*NM*NA - k6*NY*NA)/u**2,
                k6*NY*NA/u**2,
                0],h)         
    
    #m4 coefficients calculation
      
    NE = NE + m3[0,i]
    NY = NY + m3[1,i]
    NH = NH + m3[2,i]
    NP = NP
    NM = NM + m3[4,i]
    NL = NL + m3[5,i]
    NA = NA + m3[6,i] 
    NB = NB
    NW = NW
    
    m4[:,i] = np.multiply(
               [(k1r*NY*NH - k1f*NE*u - 2*k2*NE*u - k8*NY*NE)/u**2,
                (k1f*NE*u - k1r*NH*NY - k6*NA*NY - k8*NE*NY)/u**2,
                (k1f*NE*u - k1r*NY*NH)/u**2,                                    
                k2*NE/u,
                (k2*NE + k5f*NL*u - k5r*NM*NA + k8*NY*NE)/u**2,
                (k5r*NM*NA - k5f*NL*u + k8*NY*NE)/u**2,
                (k5f*NL*u - k5r*NM*NA - k6*NY*NA)/u**2,
                k6*NY*NA/u**2,
                0],h)  
    
    #Flux value for the next iteration
    N[:,i+1] = N[:,i] + (m1[:,i] + 2*m2[:,i] + 2*m3[:,i] + m4[:,i])/6

 ###########################################################
 #Euler step: update pressure, velocity and viscosity
#update on pressure by Euler's method 
    P[i+1] = P[i] + h*-312.6*u*mu[i]**0.25

#update on velocity using ideal gas law 
    v[i+1] = RJ*T/P[i+1]*np.sum(N[:,i+1])

#update on viscosity according to Wilke definition  
    mu[i+1] = np.sum(N[:,i+1]*mus*root_Ms)/np.sum(N[:,i+1]*root_Ms)

 ###########################################################
#Schmidt selectivity, derived based on the proportionality of rate of reaction
    #and flux and molar flow rate  
    S = (N[1,-1]-NYo)/(sum(N[:,-1])-N[0,-1]-N[8,-1]-NYo)
    
 ############################################################   
    
#Plotting
plt.figure(1)
plt.title('Mole fraction of ethane and ethylene')
plt.plot(z,np.divide(N[0,:],sum(N)),label='Ethylene mole fraction')
plt.plot(z,np.divide(N[1,:],sum(N)),label='Ethane mole fraction')
plt.xlabel('Distance (m)')
plt.ylabel('Mole Fraction')
plt.axis([0,85,0,0.6])
plt.legend()
plt.text(40,0.3,'Ethylene selectivity= %1.3f' %S, fontsize = 10)