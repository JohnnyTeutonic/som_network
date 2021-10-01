import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
class kohonen2():
    """example usage: 
    som = kohonen(5,5,np.random.random((20, 3)), n_iter=100)
    som.train()
    indices = som.predict(np.random.random((1000,1)))
    preds = np.zeros(indices.shape)
    for i, index in enumerate(indices):
        preds[i] = som.grid[0][index]
    preds = np.atleast_2d(preds)
    plt.imshow(som.grid, cmap="Spectral")
    plt.title('grid after 100 iterations')
    plt.imshow(preds.reshape(10,100), cmap='Spectral')"""
    def __init__(self, width, height, input_data, n_iter=100):
        self.width = width
        self.height = height
        self.n_nodes = self.width*self.height
        self.input_data = input_data
        self.n_iter = n_iter
        self.sigma_0 = max(self.height, self.width) / 2
        self.grid = np.random.random((self.width, self.height, self.input_data.shape[1]))
        #grid = np.random.randint(1,255,(30,10))/255
        #grid = grid/grid.sum(axis=0)
        self.sigma_0 = max(self.grid.shape[0], self.grid.shape[1])/2
        self.grid = self.grid.reshape(self.grid.shape[0]*self.grid.shape[1], self.input_data.shape[1])
        self.lambd = self.n_iter/np.log(self.sigma_0)

    def dist(self, x , y): 
        """euclidean distance metric"""
        return np.sqrt(np.sum((x-y)**2))
    
    def calc_neighbourhood_radius(self, sigma_0, t):
        """calculation of neighbourood radius using decaing exponential"""
        return sigma_0*np.exp(-(t/self.lambd))
    
    def learning_rate(self, t, initial_lr=0.1):
        """calculation of learning rate using decaing exponential"""
        return initial_lr*np.exp(-t/self.lambd)
    
    def train(self):
        for current_time in range(1, self.n_iter+1):
            cur_neighborhood_radius = self.calc_neighbourhood_radius(self.sigma_0, current_time)
            current_lr = self.learning_rate(current_time)
            #for _, _ in enumerate(self.input_data.T):
            current_input_vector = (self.input_data[np.random.choice(self.input_data.shape[0])-1])
            theta_vals = {}
            cur_influence = {}
            nodes_dist_cur = {}
            best_dist = np.inf
            for idx2, val in enumerate(self.grid): # find the smallest distance in the grid
                cur_dist = self.dist(val, current_input_vector)
                nodes_dist_cur[idx2] = cur_dist
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    BMU = idx2  # set the BMU to node with smallest distance
            result = np.zeros(self.height*self.width)
            for key, val in nodes_dist_cur.items():
                result[key] = val
            good_indices = np.argwhere(result<cur_neighborhood_radius) # find indices of nodes which are below threshold for neighborhood radius
            good_indices = good_indices[good_indices!=BMU]
            for val in good_indices:
                # function I chose to use for this calculation (with the euclidean distance) is the multivariate normal distribution 
                cur_influence[val] = self.dist(multivariate_normal.pdf(self.grid[BMU]), multivariate_normal.pdf(self.grid[val]))
                # get the theta vals
                theta_vals[val] = np.exp(-((cur_influence[val]**2)/(2*(cur_neighborhood_radius)**2)))
            # update the weights for the grid
            for key, val in theta_vals.items():
                self.grid[key] = self.grid[key] + current_lr*theta_vals[key]*(current_input_vector-self.grid[key])            
        #self.grid = self.grid.reshape(self.height, self.width, self.input_size)
        return self.grid
    
    def predict(self, samples):
        """use this function for predicting the label for a random color"""
        result = []
        for sample in samples.ravel():
            distances = cdist(self.grid[0][:,None], np.atleast_2d(sample), metric='euclidean')
            indices = np.argwhere(distances==distances.min())
            result.append(indices[0,0])
        return np.array(result)

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
class kohonen2():
    """example usage: 
    som = kohonen(5,5,np.random.random((20, 3)), n_iter=100)
    som.train()
    indices = som.predict(np.random.random((1000,1)))
    preds = np.zeros(indices.shape)
    for i, index in enumerate(indices):
        preds[i] = som.grid[0][index]
    preds = np.atleast_2d(preds)
    plt.imshow(som.grid, cmap="Spectral")
    plt.title('grid after 100 iterations')
    plt.imshow(preds.reshape(10,100), cmap='Spectral')"""
    def __init__(self, width, height, input_data, n_iter=100):
        self.width = width
        self.height = height
        self.n_nodes = self.width*self.height
        self.input_data = input_data
        self.n_iter = n_iter
        self.sigma_0 = max(self.height, self.width) / 2
        self.grid = np.random.random((self.width, self.height, self.input_data.shape[1]))
        #grid = np.random.randint(1,255,(30,10))/255
        #grid = grid/grid.sum(axis=0)
        self.sigma_0 = max(self.grid.shape[0], self.grid.shape[1])/2
        self.grid = self.grid.reshape(self.grid.shape[0]*self.grid.shape[1], self.input_data.shape[1])
        self.lambd = self.n_iter/np.log(self.sigma_0)

    def dist(self, x , y): 
        """euclidean distance metric"""
        return np.sqrt(np.sum((x-y)**2))
    
    def calc_neighbourhood_radius(self, sigma_0, t):
        """calculation of neighbourood radius using decaing exponential"""
        return sigma_0*np.exp(-(t/self.lambd))
    
    def learning_rate(self, t, initial_lr=0.1):
        """calculation of learning rate using decaing exponential"""
        return initial_lr*np.exp(-t/self.lambd)
    
    def train(self):
        for current_time in range(1, self.n_iter+1):
            cur_neighborhood_radius = self.calc_neighbourhood_radius(self.sigma_0, current_time)
            current_lr = self.learning_rate(current_time)
            #for _, _ in enumerate(self.input_data.T):
            current_input_vector = (self.input_data[np.random.choice(self.input_data.shape[0])-1])
            theta_vals = {}
            cur_influence = {}
            nodes_dist_cur = {}
            best_dist = np.inf
            for idx2, val in enumerate(self.grid): # find the smallest distance in the grid
                cur_dist = self.dist(val, current_input_vector)
                nodes_dist_cur[idx2] = cur_dist
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    BMU = idx2  # set the BMU to node with smallest distance
            result = np.zeros(self.height*self.width)
            for key, val in nodes_dist_cur.items():
                result[key] = val
            good_indices = np.argwhere(result<cur_neighborhood_radius) # find indices of nodes which are below threshold for neighborhood radius
            good_indices = good_indices[good_indices!=BMU]
            for val in good_indices:
                # function I chose to use for this calculation (with the euclidean distance) is the multivariate normal distribution 
                cur_influence[val] = self.dist(multivariate_normal.pdf(self.grid[BMU]), multivariate_normal.pdf(self.grid[val]))
                # get the theta vals
                theta_vals[val] = np.exp(-((cur_influence[val]**2)/(2*(cur_neighborhood_radius)**2)))
            # update the weights for the grid
            for key, val in theta_vals.items():
                self.grid[key] = self.grid[key] + current_lr*theta_vals[key]*(current_input_vector-self.grid[key])            
        #self.grid = self.grid.reshape(self.height, self.width, self.input_size)
        return self.grid
    
    def predict(self, samples):
        """use this function for predicting the label for a random color"""
        result = []
        for sample in samples.ravel():
            distances = cdist(self.grid[0][:,None], np.atleast_2d(sample), metric='euclidean')
            indices = np.argwhere(distances==distances.min())
            result.append(indices[0,0])
        return np.array(result)

