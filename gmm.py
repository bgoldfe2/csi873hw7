# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:33:56 2017

@author: bgoldfeder2
"""

import numpy as np
import math
    
class GMM:
    
    def __init__(self, sigma =1,eps = 0.0001,mi = 1000):
        
        self.sigma = sigma
        self.eps = eps ## threshold to stop `epsilon`
        self.max_iters = mi
        self.sigma = 1 # sigma given as 1 in this hw.
        self.k = 2 # only two gaussian in this hw.
    
    
    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points       
        n = X.shape[0]
        print("n is ",n)
    
        #### Continue with steps one and two from text
        # Excpectations array for step one
        E = np.zeros((n,self.k))
        
        # choose the starting centroids/means         
        mu = np.divide(np.ones(2),2) 
        print('mu is ',mu)
        
 
        num_iter = 0
        # Iterate till max_iters iterations        
        while num_iter < max_iters:
            num_iter += 1
            # Step one find expectation probability the x_i was generated
            # by the jth Normal distribution
            for j in range(self.k):
                for i in range(n):
                    
                    E[i,j] = math.exp((-1/(2*sigma**2))*((X[i]-mu[j])**2)) \
                        / (math.exp((-1/(2*sigma**2))*((X[i]-mu[0])**2)) \
                        + math.exp((-1/(2*sigma**2))*((X[i]-mu[1])**2))) 
            
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
            # End of while iteration loop
            
        
        return mu,sum_min
    
    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
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
        max_iters = 1000
    else: max_iters = int(options.maxiters)
    
    if not options.sigma :
        print("Used default sigma = 1" )
        sigma = 1
    else: sigma = int(options.sigma)
    
    X = np.genfromtxt(options.filepath, delimiter=' ')
    gmm = GMM(sigma,eps,max_iters)
    mu,err = gmm.fit_EM(X, max_iters)
    print ("mu is ",mu,"error is ",err)
    #gmm.plot_log_likelihood()
    