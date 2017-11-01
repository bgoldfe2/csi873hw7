# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:33:56 2017

@author: bgoldfeder2
"""

import numpy as np
import math
    
class GMM:
    
    def __init__(self, sigma,eps,mi):
        
        self.sigma = sigma
        self.eps = eps ## threshold to stop `epsilon`
        self.max_iters = mi
        self.sigma = sigma # sigma given as 1 in this hw.
        self.k = 2 # only two gaussian in this hw.
    
    
    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points       
        n = X.shape[0]
        print("n is ",n)
    
        #### Continue with steps one and two from text
        # Excpectations array for step one
        E = np.zeros((n,self.k))
        
        # choose the starting centroids/means         
        mu = np.asarray([-2.0,-4.0])
        print('mu is ',mu)
        
        # counter for the number of iterations
        num_iter = 0
        
        # difference in error for successive iterations
        past_err = [5,10]
       
        
        # Iterate till max_iters iterations        
        while num_iter < max_iters:
            num_iter += 1
            # Step one find expectation probability the x_i was generated
            # by the jth Normal distribution
            for j in range(self.k):
                for i in range(n):
                    #print ("muusss ",mu[0]," - ",mu[1])
                    #print ("X is ",X[i])
                    numer = math.exp((-1/(2*self.sigma**2))*((X[i]-mu[j])**2))
                    denom1 = math.exp((-1/(2*self.sigma**2))*((X[i]-mu[0])**2)) 
                    denom2 = math.exp((-1/(2*self.sigma**2))*((X[i]-mu[1])**2))
                    denom = denom1 + denom2
                    #print ("numer is ", numer, " denom1 is ",denom1," denom2 is ",denom2, "denom is ",denom)
                    E[i,j] = numer/denom
                    #print(E[i,j],end="")
                    
            # Step two derive a new max liklihood hypothesis m_1,m_2
            for j in range(self.k):
                numerator = 0.0
                denominator = 0.0
                for i in range(n):
                    numerator += E[i,j] * X[i]
                    denominator += E[i,j]
                mu[j]=numerator/denominator
                
                
            sum_min = [0.0,0.0]
            for j in range(self.k):
                    for i in range(n):
                        sum_min[j] += (X[i]-mu[j])**2
            print ("Error squared is ",sum_min)
            if ((sum_min[0]-past_err[0]) + (sum_min[1]-past_err[1])) < 2*eps:
                break
            # End of while iteration loop
            
        
        return mu,sum_min
    
    def plot_normals(self,mu):
       import matplotlib.pyplot as plt
       import matplotlib.mlab as mlab
       
        
       
       x1 = np.linspace(mu[0] - 3*self.sigma, mu[0] + 3*self.sigma, 100)
       plt.plot(x1,mlab.normpdf(x1, mu[0], self.sigma))
       
       x2 = np.linspace(mu[0] - 3*self.sigma, mu[1] + 3*self.sigma, 100)
       plt.plot(x2,mlab.normpdf(x2, mu[1], self.sigma))

       plt.show()
        
        
if __name__ == "__main__":

    # Parse commjand line options filename, epsilon, and maximum iterations    
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filepath", help="File path for data")
    parser.add_option("-e", "--eps", dest="epsilon", help="Epsilon to stop")    
    parser.add_option("-m", "--maxiters", dest="max_iters", help="Maximum no. of iteration")        
    parser.add_option("-s", "--sigma", dest="sigma", help="Sigma value")        

    options, args = parser.parse_args()
    
    if not options.filepath : raise('File not provided')
    
    if not options.epsilon :
        print("Used default eps = 0.0001" )
        eps = 0.0001
    else: eps = float(options.epsilon)
    
    if not options.max_iters :
        print("Used default maxiters = 1000" )
        max_iters = 200
    else: max_iters = int(options.maxiters)
    
    if not options.sigma :
        print("Used default sigma = 1" )
        sigma = 1
    else: sigma = int(options.sigma)
    
    X = np.genfromtxt(options.filepath, delimiter=' ')
    gmm = GMM(sigma,eps,max_iters)
    mu,err = gmm.fit_EM(X, max_iters)
    print ("mu is ",mu)
    gmm.plot_normals(mu)
    