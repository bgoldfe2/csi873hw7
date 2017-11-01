# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:33:56 2017

@author: bgoldfeder2
"""

import numpy as np
    
class GMM:
    
    def __init__(self, eps = 0.0001,mi = 1000):
        
        self.eps = eps ## threshold to stop `epsilon`
        self.max_iters = mi
        self.sigma = 1
    
    
    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points       
        n = X.shape
                
        # choose the starting centroids/means 
        print("k is ", self.k)
        init_mus = [2,4]
        mu = np.asarray(init_mus)
        
        #### Continue with steps one and two from text
        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []
        
        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 
                        
        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            
            
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        
        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)       
        
        return self.params
    
    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()
    
    def predict(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)
        
        
if __name__ == "__main__":

    # Parse commjand line options filename, epsilon, and maximum iterations    
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filepath", help="File path for data")
    parser.add_option("-e", "--eps", dest="epsilon", help="Epsilon to stop")    
    parser.add_option("-m", "--maxiters", dest="max_iters", help="Maximum no. of iteration")        
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
    
    X = np.genfromtxt(options.filepath, delimiter=' ')
    gmm = GMM(eps,max_iters)
    params = gmm.fit_EM(X, max_iters)
    print (params.log_likelihoods)
    gmm.plot_log_likelihood()
    print (gmm.predict(np.array([1, 2])))