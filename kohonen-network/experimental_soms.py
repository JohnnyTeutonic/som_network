import numpy as np
class kohonen3():
    
    def __init__(self, width, height, input_size, input_data, n_iter=100):
        self.width = width
        self.height = height
        self.n_nodes = self.width*self.height
        self.input_size = input_size
        self.grid = np.random.random((self.width * self.height, self.input_size))
        self.input_data = input_data
        self.n_iter = n_iter
        self.sigma_0 = max(self.height, self.width) / 2
        self.lambd = self.n_iter/np.log(self.sigma_0)
        #self.result = np.zeros((self.width * self.height, self.width * self.height))

    def dist(self, x , y, z=None):   
        return np.sqrt(np.sum((x-y)**2))
    
    def calc_neighbourhood_radius(self, sigma_0, lambd, t):
        return sigma_0*np.exp(-(t/self.lambd))
    
    def learning_rate(self, t, initial_lr=0.1):
        return initial_lr*np.exp(-t/self.lambd)
    
    def train(self):
        for current_time in range(1, self.n_iter+1):
            cur_neighborhood_radius = self.calc_neighbourhood_radius(self.sigma_0, self.lambd, current_time)
            #print(cur_neighborhood_radius)
            current_lr = self.learning_rate(current_time)
            BMU_idxs  = {}
            BMU_nodes = {}
            #nodes_dist_all = {}
            for idx, val in enumerate(range(self.input_data.shape[1])):
                current_input_vector = (self.input_data[0][np.random.choice(self.input_data.shape)-1])
                self.theta_vals = {}
                self.cur_influence = {}
                self.nodes_dist_cur = {}
                BMU = np.inf
                best_node = 0
                for idx2, val in enumerate(self.grid.ravel()):
                    #for idx3, j in enumerate(range(len(self.grid[i]))):
                       # for idx2, val in enumerate(self.grid):
                    cur_dist = self.dist(current_input_vector, val)
                    #nodes_dist_all[idx, idx2] = cur_dist
                    self.nodes_dist_cur[idx2] = cur_dist
                    if cur_dist < BMU:
                        BMU = cur_dist
                        best_node = idx2
                result = np.zeros(self.height*self.width*self.input_size)
                for key, val in self.nodes_dist_cur.items():
                    result[key] = val
                print(result)
                good_indices = np.argwhere(result<cur_neighborhood_radius)
                good_indices = good_indices[good_indices!=best_node]
                print(good_indices, best_node)
                in_nodes = result[good_indices]
                for val in good_indices:
                    self.cur_influence[val] = self.dist(self.grid[best_node], self.grid[val])
                    self.theta_vals[val] = np.exp(-((self.cur_influence[val]**2)/(2*(cur_neighborhood_radius)**2)))
                for key, val in self.theta_vals.items():
                    self.grid[key] = self.grid[key] + current_lr*self.theta_vals[key]*(current_input_vector-self.grid[key])            
        #self.grid = self.grid.reshape(self.height, self.width, self.input_size)
        return self.grid
    
    import numpy as np
    
    
class kohonen2():
    
    def __init__(self, width, height, input_size, input_data, n_iter=100):
        self.width = width
        self.height = height
        self.n_nodes = self.width*self.height
        self.input_size = input_size
        self.grid = np.random.random((self.width * self.height, self.input_size))
        self.input_data = input_data
        self.n_iter = n_iter
        self.sigma_0 = max(self.height, self.width) / 2
        self.lambd = self.n_iter/np.log(self.sigma_0)
        #self.result = np.zeros((self.width * self.height, self.width * self.height))

    def dist(self, x , y, z=None):   
        return np.sqrt(np.sum((x-y)**2))
    
    def calc_neighbourhood_radius(self, sigma_0, lambd, t):
        return sigma_0*np.exp(-(t/self.lambd))
    
    def learning_rate(self, t, initial_lr=0.1):
        return initial_lr*np.exp(-t/self.lambd)
    
    def train(self):
        for current_time in range(1, self.n_iter+1):
            cur_neighborhood_radius = self.calc_neighbourhood_radius(self.sigma_0, self.lambd, current_time)
            print(cur_neighborhood_radius)
            current_lr = self.learning_rate(current_time)
            BMU_idxs  = {}
            BMU_nodes = {}
            nodes_dist_all = {}
            for idx, current_input_vector in enumerate(self.input_data):
                BMU_idxs[idx] = 0
                BMU_nodes[idx] = 0
                theta_vals = {}
                weights = {}
                cur_influence = {}
                nodes_dist_cur = {}
                BMU = np.inf
                best_node = 0
                for idx2, i in enumerate(range(len(self.grid))):
                    for j in range(len(self.grid[i])):
                        val = self.grid[i][j]

               # for idx2, val in enumerate(self.grid):
                        cur_dist = self.dist(val, current_input_vector)
                        nodes_dist_all[idx, idx2] = cur_dist
                        nodes_dist_cur[idx2] = cur_dist
                        if cur_dist < BMU:
                            BMU = cur_dist
                            best_node = idx2
                            BMU_idxs[idx] = cur_dist
                            BMU_nodes[idx] = best_node
                result = np.zeros(self.height*self.width)
                for key, val in nodes_dist_cur.items():
                    result[key] = val
                print(result)
                good_indices = np.argwhere(result<cur_neighborhood_radius)
                good_indices = good_indices[good_indices!=best_node]
                print(good_indices, best_node)
                in_nodes = result[good_indices]
                for val in good_indices:
                    cur_influence[val] = self.dist(self.grid[best_node], self.grid[val])
                    theta_vals[val] = np.exp(-((cur_influence[val]**2)/(2*(cur_neighborhood_radius)**2)))
                for key, val in theta_vals.items():
                    self.grid[key] = self.grid[key] + current_lr*theta_vals[key]*(current_input_vector-self.grid[key])            
        #self.grid = self.grid.reshape(self.height, self.width, self.input_size)
        return self.grid