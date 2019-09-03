#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Monte Carlo simulations to price Binary and Lookback options

#load necessary libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest
import scipy as sp
import pandas as pd 
import pylab 
from statsmodels.graphics.gofplots import qqplot

np.random.seed(1)


# In[2]:


# We start by defining varios functions that wil be used to calculate
# the following options


#Binary Call Option ( all or nothing)
#Create functions to calculate options
def binarycall_option(T,So,K, sigma,r,N,M):
    """
    function to calculate call price using gbm
    T: maturity (years)
    So: spot (dollars)
    K: strike (dollars)
    sigma: volatility (real)
    r: interest rate (real)
    N: number of steps
    M: number of simulations
    """
    # STEP 1: input parameters
    S = sp.random.rand(N+1)  # N+1 random numbers from scipy
    sumpayoff = 0.0
    premium = 0.0   #define 
    dt = T / N   #time steps
    
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        sumpayoff += max(S[N]-K,0.0)

    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


#Binary Put Option= all or anothing
def binaryput_option(T,So,K,sigma,r,N,M):
    """
    function to calculate call price using gbm
    T: maturity (years)
    So: spot (dollars)
    K: strike (dollars)
    sigma: volatility (real)
    r: interest rate (real)
    N: number of steps
    M: number of simulations
    """
    # STEP 1: input parameters
    S = sp.random.rand(N+1)  # N+1 random numbers from scipy
    sumpayoff = 0.0
    premium = 0.0
    dt = T / N
    
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        sumpayoff += max(K-S[N],0.0)

    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


# In[3]:


#Fixed strike lookback

# Call
def lookbackcall_fixed(So,K,T,r,sigma,M,N):
    
    S = sp.random.rand(N+1)
    sumpayoff = 0.0
    premium = 0.0
    dt = T/N
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        Smax = max(S)
        sumpayoff += max(0, Smax-K)*np.exp(-r*T)

    
    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


# put
def lookbackput_fixed(So,K,T,r,sigma,M,N):
    
    S = sp.random.rand(N+1)
    sumpayoff = 0.0
    premium = 0.0
    dt = T/N
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        Smin = min(S)
        sumpayoff += max(0, K-Smin)*np.exp(-r*T)
        
    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


# In[4]:


#Floating strike lookback

# Call
def lookbackcall_floating(So,T,r,sigma,M,N):
    
    S = sp.random.rand(N+1)
    sumpayoff = 0.0
    premium = 0.0
    dt = T/N
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        Smin = min(S)
        sumpayoff += max(0, S[-1]-Smin)*np.exp(-r*T)
        
    
    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


# put
def lookbackput_floating(So,T,r,sigma,M,N):
    
    S = sp.random.rand(N+1)
    sumpayoff = 0.0
    premium = 0.0
    dt = T/N
    
    # STEP 2: MAIN SIMULATION LOOP
    for j in range(M):
   	
        S[0]=So

        # STEP 3: TIME INTEGRATION LOOP
        for i in range(N):
        
            epsilon =  sp.random.randn(1)
            S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon)
        
        # STEP 4: COMPUTE PAYOFF
        Smax = max(S)
        sumpayoff += max(0, Smax-S[-1])*np.exp(-r*T)
        
    
    # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
    premium =  math.exp(-r*T)*(sumpayoff / M) 
    
    
    return premium


# In[5]:


# Define Input Parameters
T = 1 # Maturity 
K = 100 # Strike 
So = 100 # Starting Price 
N = 252 # Time to Expiry 
sigma = 0.2 # volatility 
r = 0.05 # interest rate 


#Error Testing and martingdale property test
#try different simulations (paths) to examine associated errors
M1 = 100 # No. of simulations 1 
M2 = 1000 # No. of simulations 2
M3 = 2500 #No. of simulations 3
M4 = 5000 # No. of simulations 4 
M5 = 10000 # No. of simulations 5 

#Try different time steps to evaluate assiciated errors
No = 365  #daily returms including weekends
N1 = 252  #Annual trading days
N2 = 52 #Weekly trading
N3 = 12 # monthly

S = sp.random.rand(N+1)    #Generate N+1 random numbers
#calculate time stepd dt
dt = T/N
#Define payoff imputed variable
premium = 0
sumpayoff = 0
disc = math.exp(-r*T)


# In[6]:


# Define Paths and get the prices
# Step 2: Main loop
pfinal = []
dpfinal = []
epsilon = sp.random.randn(N,M2)
for j in range(M2):
    S[0] = So
    # Step 3: Time integration loop
    # get daily price returns
    price_paths = [So]
    
    for i in range(N):
        S[i+1] = S[i]*(1+r*dt+sigma*math.sqrt(dt)*epsilon[i][j])
        price_paths.append(S)
        
        #Discounted Prices
        S_disc = disc*S
        
    # Step 4: Compute Payoff
    sumpayoff += max(S[N]-K,0)
    
     #plot the paths
    plt.plot(S)
    plt.axhline(y=100.0, color='r',linestyle='-')
    plt.ylabel('Price')
    plt.xlabel('Day')
    plt.title('Option Pricing: Monte Carlo Simulations(1000s/252timesteps)')
    #append final prices
    pfinal.append(S[-1])
    #Get discounted prices
    dpfinal.append(S_disc)
    
#display plot
plt.show()

#print(S)
    
# Step 5: Compute Discounted Expected Payoff
premium = math.exp(-r*T)*(sumpayoff / M2)

# Step 6: Output Results
print('European Average Option Price: %.3f'% premium)


# In[7]:


#Calculate Option Prices with the functions
# 1. Binary Call
# Define Input Parameters
T = 1 # Maturity 
K = 100 # Strike 
So = 100 # Starting Price 
N = 52 # Time to Expiry 
sigma = 0.2 # volatility 
r = 0.05 # interest rate 
M = 1000


print('Binary Call Option Price: %.3f'%binarycall_option(T,So,K,sigma,r,N,M))
print('Binary Put Option Price: %.3f'%binaryput_option(T,So,K,sigma,r,N,M))
print('Lookback Call Fixed Option Price: %.3f'%lookbackcall_fixed(So,K,T,r,sigma,M,N))
print('Lookback Put Fixed Option Price: %.3f'%lookbackput_fixed(So,K,T,r,sigma,M,N))
print('Lookback Call Floating Option Price: %.3f'%lookbackcall_floating(T,So,r,sigma,M,N))
print('Lookback Put Floating Option Price: %.3f'% lookbackput_floating(T,So,r,sigma,M,N))


# In[8]:


#Plot histogram to evaluate normality 
#plot hist
plt.hist(pfinal, bins=50)
plt.ylabel('Price')
plt.xlabel('Frequency')
plt.title('Distribution of Terminal European Option Prices:(1000s/252ts)')
plt.show()

#convert data to pandas dataframe
pdata = pd.DataFrame(pfinal)
#print(pdata.head(5))

dpdata = pd.DataFrame(dpfinal)
#print(pdata.head(5))


# In[ ]:




