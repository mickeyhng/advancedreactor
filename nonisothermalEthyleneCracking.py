"""
This code solves for non-isothermal cracking of ethane to ethylene using 
python's integrator, scipy.integrate.odeint. Official Python website 
recommends using the newer scipy.integrate.solve_ivp, which utilizes RK4.
The results of both methods are identical for the isothermal case. odeint was
ultimately because it requires less computing power.
"""
import math #For pi 
import matplotlib.pyplot as plt #For plotting
import numpy as np #For matrices and vectors operation
from scipy.integrate import odeint #For solving the system of ODEs

#%% Reactor parameters
Po = 11*101325 #inlet pressure [Pa]
To = 820+273.15 #operating temperature [K]
Tref = 298.15 #Reference temperature [K]
Twall = 890+273.15 #Wall temperature [K]
RJ = 8.314 #R(in J/mol·K),gas constant in [J/mol·K]
Rkcal = 1.99e-3 #gas constant [kcal/mol] 
nEo = 99 #initial molar flow rate of ethane [mol/s]
nYo = 1 #initial molar flow rate of ethylene [mol/s]
yWo = 0.4 #mole fraction of steam
d = 0.1 #reactor diameter [m]
Ac = math.pi*d**2/4 #cross-sectional area of the reactor [m^2]
L = 85 #length of the reactor [m]

#%%Defining initial molar flux [mol/m^2/s]
NEo = nEo/Ac #initial molar flux of ethane
NYo = nYo/Ac #initial molar flux of ethylene
NHo = 0 #initial molar flux of hydrogen
NPo = 0 ##initial molar flux of propane
NMo = 0 #initial molar flux of methane
NLo = 0 #initial molar flux of propylene
NAo = 0 #initial molar flux of acetylene
NBo = 0 ##initial molar flux of 1,3-butadiene
NWo = yWo/(1 - yWo)*(nEo + nYo)/Ac #initial molar flux of steam

#%% Setting up molar mass array [g/mol]
ME = 30.069 #molar mass of ethane
MY = 28.0532 #molar mass of ethylene
MH = 2.01588 #molar mass of hydrogen
MP = 44.0956 #molar mass of propane
MM = 16.0425 #molar mass of methane
ML = 42.0797 #molar mass of propylene
MA = 26.0373 #molar mass of acetylene
MB = 54.0904 #molar mass of 1,3-butadiene
MW = 18.0153 #molar mass of steam
Ms = np.array([ME, MY, MH, MP, MM, ML, MA, MB, MW])
root_Ms = Ms**0.5 #For Wilke's model

#%% Factor to reduce computation in ODEs 
rhovo=1/1000*(NEo*ME+NYo*MY+NWo*MW) #rho*v
Tfactor = 0.092*rhovo**0.8/d**1.2 #For temperature ODE

#%%Coefficients for correlations  to find mu, Cp, 
#thermal conductivity,and heat of reaction
#1. Ethane 2.Ethylene 3.Hydrogen 4.Propane 5.Methane 6.Propylene 
#7. Acetylene 8.Butadiene 9. Water


#Viscosities calculation using aT^2/(1+c/T+d/T^2) where a,b,c,d are coefficients
#obtained from DIPPR data base [Pa*s]
                      #A         B       C     D 
muCoeff=np.array([[2.5906e-7, 0.67988, 98.902, 0], 
                  [2.0789e-6, 0.4163,  352.7,  0],
                  [1.797e-7,  0.685,  .59,   140],
                  [4.9054e-8, 0.90125, 0,      0],
                  [5.2546e-7, 0.59006, 105.67, 0],
                  [7.3919e-7, 0.5423,  263.73, 0],
                  [1.2025e-6, 0.4952,  291.4,  0],
                  [2.696e-7,  0.6715,  134.7,  0],
                  [1.7096e-8, 1.1146,  0,      0]])

                        #A        B       C           D
kcondCoeff=np.array([[7.3869e-5,1.1689, 500.73,         0],
                     [8.6806e-6,1.4559, 299.72,    -29403],
                     [0.002653, 0.7452, 12,             0],
                     [-1.12,    0.10972,-9834.6,-7.5358e6],
                     [8.3983e-6,1.4268, -49.654,        0],
                     [4.49e-5,  1.2018, 421,            0],
                     [7.5782e-5,1.0327, -36.227,    31432],
                     [-20890,   0.9593, 9.382e10,       0],
                     [6.2041e-6,1.3973, 0,              0]])

             #Hformstd(J/mol)   A      B      C      D      E
Hrxndata = np.array([[-83820, 44.256,84.737,872.24,67.130,2430.4],
                     [52510,  33.380,94.790,1596,  55.100, 740.8], 
                     [0,      27.617,9.560, 2466,  3.760,  567.6],
                     [-104680,59.474,126.61,844.31,86.165,2482.7],
                     [-74520, 33.298,79.933,2086.9,41.602,991.96],
                     [19710,  43.852,150.60,1398.8,74.754,616.46],
                     [228200, 36.921,31.793,678.05,33.430,3036.6],
                     [109240, 50.950,170.50,1532.4,133.70, 685.6],
                     [-241814,33.363,26.790,2610.5,8.896,   1169]])

#%%Heat of reaction calculation
#Redefine the vector foe easier coding
A=Hrxndata[:,1]
B=Hrxndata[:,2]
C=Hrxndata[:,3]
D=Hrxndata[:,4]
E=Hrxndata[:,5]

#Heat of reaction at standard temperature calculation
Hrno1=Hrxndata[1,0]-Hrxndata[0,0]
Hrno2=Hrxndata[3,0]+Hrxndata[4,0]-2*Hrxndata[0,0]
Hrno3=Hrxndata[4,0]+Hrxndata[6,0]-Hrxndata[5,0]
Hrno4=Hrxndata[7,0]-Hrxndata[1,0]-Hrxndata[6,0]
Hrno5=Hrxndata[4,0]+Hrxndata[5,0]-Hrxndata[0,0]-Hrxndata[1,0]

#Calculate the change in enthalpy from reference to inlet temperature
intCp=A*(To-Tref)+B*C*(np.cosh(C/To)/np.sinh(C/To)
      -np.cosh(C/Tref)/np.sinh(C/Tref))-D*E*(np.tanh(E/To)-np.tanh(E/Tref))

#Heat of reaction at inlet temperature
Hrxn1=Hrno1-intCp[0]
Hrxn2=Hrno2-2*intCp[0]
Hrxn3=Hrno3-intCp[5]
Hrxn4=Hrno4-1*(intCp[6]+intCp[1])
Hrxn5=Hrno5-(intCp[0]+intCp[1])

#%%Arrhenius equation parameters in array form in descending order
                         #ko         Activation energy (kcal/mol)
AhEqParam = np.array([[4.652e13,     65.20],
                      [8.75e8/1000,   32.7],
                      [3.85e11,      65.25],
                      [9.814e8,      36.92],
                      [5.87e4/1000,  7.043],
                      [1.026e12/1000,41.26],
                      [7.083e13/1000,60.43]])

#%%Initialization

zo = 0 #initial length of the reactor [m]
zmax = L #final length of the reactor [m]

h = 0.001 #step size [m]

n=int(zmax/h+1)
zspan = np.linspace(zo,zmax,n) #evenly divide the space

#%% Defining the model
def model(N,z):
    
    #Redefining the variable for easier coding
    NE=N[0]
    NY=N[1]
    NH=N[2]
    NP=N[3]
    NM=N[4]
    NL=N[5]
    NA=N[6]
    NB=N[7]
    NW=N[8]
    P=N[9]
    T=N[10]
    
    Speciesflux=[NE,NY,NH,NP,NM,NL,NA,NB,NW] #Species flux array
    
    Ntot=sum(Speciesflux) #Total flux
    u=Ntot*RJ*T/P #Velocity using ideal gas law
    
    #Viscosity calculation using Wilke's model
    mus=muCoeff[:,0]*T**muCoeff[:,1]/(1+muCoeff[:,2]/T+muCoeff[:,3]/T**2)
    mus_wall=muCoeff[:,0]*Twall**muCoeff[:,1]/(1+muCoeff[:,2]/Twall+muCoeff[:,3]/Twall**2)
    mu=np.sum(Speciesflux*mus*root_Ms)/np.sum(Speciesflux*root_Ms)
    mu_wall=np.sum(Speciesflux*mus_wall*root_Ms)/np.sum(Speciesflux*root_Ms)
    
    #Thermal conductivity calculation using Wilke's model
    kconds=kcondCoeff[:,0]*T**kcondCoeff[:,1]/(1+kcondCoeff[:,2]/T+kcondCoeff[:,3]/T**2)
    kcond=np.sum(Speciesflux*kconds*root_Ms)/np.sum(Speciesflux*root_Ms)
    
    #Mixture heat capcity calculation
    Cp=A+B*(C/(T*np.sinh(C/T)))**2+D*(E/(T*np.cosh(E/T)))**2 #Cp of every
                                                           #species at temp T
    Cp_mass=Cp/(Ms/1000) #Converting Cp from molar to mass basis (J/kg/mol)
    Massflux=Speciesflux*Ms #Converting molar flux to mass flux
    Mtot=sum(Massflux)
    Cp_mix=np.sum(Cp_mass*Massflux/Mtot) #Mass average Cp
    
    #Updating rate constants
    k=AhEqParam[:,0]*np.exp(-1*AhEqParam[:,1]/Rkcal/T)
    k1f=k[0]
    k1r=k[1]
    k2=k[2]
    k5f=k[3]
    k5r=k[4] 
    k6=k[5]
    k8=k[6]
    
    #Calculate change of enthalpy from Tref to T
    DCp=A*(T-Tref)+B*C*(np.cosh(C/T)/np.sinh(C/T)
      -np.cosh(C/Tref)/np.sinh(C/Tref))-D*E*(np.tanh(E/T)-np.tanh(E/Tref))
    #Heat of reaction at T
    DHrxn1=Hrxn1+DCp[1]+DCp[2]
    DHrxn2=Hrxn2+DCp[3]+DCp[4]
    DHrxn3=Hrxn3+DCp[4]+DCp[6]
    DHrxn4=Hrxn4+DCp[7]
    DHrxn5=Hrxn5+DCp[4]+DCp[5]
    DHrxn=[DHrxn1,DHrxn2,DHrxn3,DHrxn4,DHrxn5]
    
    #Setting up a rate vector for heat of reaction 
    Rate=np.array([k1f*NE/u - k1r*NY*NH/u**2,
                   k2*NE/u,
                   k5f*NL/u - k5r*NA*NM/u**2,
                   k6*NA*NY/u**2,
                   k8*NY*NE/u**2])
    
    qrxn=-1*np.sum(DHrxn*Rate) #redefining heat of reaction for easier coding
    #Redefining overall heat transfer coefficient
    U=Tfactor*kcond**(2/3)*Cp_mix**(1/3)/mu_wall**0.14/mu**(0.326666666666)
    
    #Setting up system of ODEs
    dNEdz=(k1r*NY*NH - k1f*NE*u - 2*k2*NE*u - k8*NY*NE)/u**2
    dNYdz=(k1f*NE*u - k1r*NH*NY - k6*NA*NY - k8*NE*NY)/u**2
    dNHdz=(k1f*NE*u - k1r*NY*NH)/u**2             
    dNPdz=k2*NE/u
    dNMdz=(k2*NE*u + k5f*NL*u - k5r*NM*NA + k8*NY*NE)/u**2
    dNLdz=(k5r*NM*NA - k5f*NL*u + k8*NY*NE)/u**2
    dNAdz=(k5f*NL*u - k5r*NM*NA - k6*NY*NA)/u**2
    dNBdz=k6*NY*NA/u**2
    dNWdz=0
    dPdz=-313*mu**.25*u
    dTdz=1/rhovo/Cp_mix*(qrxn+U*(Twall-T))
    dNdz=[dNEdz,dNYdz,dNHdz,dNPdz,dNMdz,dNLdz,dNAdz,dNBdz,dNWdz,dPdz,dTdz]
    return dNdz

#%% Solving the system of ODEs
Ni=[NEo, NYo, NHo, NPo, NMo, NLo, NAo, NBo, NWo,Po,To] #Initial condition
N1=odeint(model,Ni,zspan) #Executing the program
#Total molar flux
Ntot=N1[:,0]+N1[:,1]+N1[:,2]+N1[:,3]+N1[:,4]+N1[:,5]+N1[:,6]+N1[:,7]+N1[:,8]
yE=np.divide(N1[:,0],Ntot) #Mole fraction of ethane
yY=np.divide(N1[:,1],Ntot) #Mole fraction of ethylene
nondimT=N1[:,10]/To        #T/To
nondimP=N1[:,9]/Po         #P/Po
#Vecloity calculation
v=(N1[:,0]+N1[:,1]+N1[:,2]+N1[:,3]+N1[:,4]+N1[:,5]+N1[:,6]+N1[:,7]+N1[:,8])*RJ*N1[:,10]/N1[:,9]
vo=(NEo+NYo+NWo)*RJ*To/Po #Inlet velocity (m/s)
nondimv=v/vo #v/vo
#%%Plotting
plt.figure(1)
plt.title('Mole fraction of ethane and ethylene along the reactor')
plt.plot(zspan,yE,label='Ethane mole fraction')
plt.plot(zspan,yY,label='Ethylene mole fraction')
plt.xlabel('Distance (m)')
plt.axis([0,85,0,0.6])
plt.legend()
plt.figure(2)
plt.title('Pressure, velocity, and temperature along the reactor')
plt.plot(zspan,nondimT,label='T/To')
plt.plot(zspan,nondimP,label='P/Po')
plt.plot(zspan,nondimv,label='v/vo')
plt.axis([0,85,0.4,2.7])
plt.legend()
plt.show()
#%%Conversion and Selectivity
XE=(NEo-N1[-1,0])/NEo #Ethane conversion
#Ethylene selectivity
S=(N1[-1,1]-NYo)/(N1[-1,1]+N1[-1,2]+N1[-1,3]+N1[-1,4]+N1[-1,5]+N1[-1,6]+N1[-1,7]-NYo) 
#Mass flow rate of ethylene at the outlet (kg/s)
MFE=N1[-1,1]*Ac*MY/1000 
#I think Prof. Okorafor said mass flow rate is more meaningful to my boss
#who might not be a ChemE
print('The conversion of ethane is %1.2f' %XE)
print('The selectivity of ethylene is %1.2f' %S)
print('The mass flow rate of ethylene at the outlet is %1.2f kg/s' %MFE)
