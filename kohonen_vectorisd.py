import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


class kohonen3:
    """example usage:
    som = kohonen3(5,5,np.random.random((20, 3)), n_iter=100)
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
        self.n_nodes = self.width * self.height
        self.input_data = input_data
        self.n_iter = n_iter
        self.grid = np.random.random(
            (self.width, self.height, self.input_data.shape[0]))
        self.sigma_0 = max(self.grid.shape[0], self.grid.shape[1]) / 2
        self.grid = self.grid.reshape(
            self.grid.shape[0] * self.grid.shape[1],
            self.input_data.shape[0])
        self.lambd = self.n_iter / np.log(self.sigma_0)

    def calc_neighbourhood_radius(self, sigma_0, t):
        """calculation of neighbourood radius using decaing exponential"""
        return sigma_0 * np.exp(-(t / self.lambd))

    def learning_rate(self, t, initial_lr=0.1):
        """calculation of learning rate using decaing exponential"""
        return initial_lr * np.exp(-t / self.lambd)

    def calculate_distances(self, current_input_vector):
        """calculate distances of grid from current input vector"""
        distances = cdist(
            np.atleast_2d(current_input_vector),
            self.grid,
            metric='euclidean')
        return distances

    def get_min_distances(self, distances):
        """get the index of the minimum distance - the BMU itself"""
        return np.argmin(distances)

    def get_valid_neighbours(self, distances, BMU, neighborhood_radius):
        """after determining the BMU, find the indices of the nodes 
        that are within the current neighborhood radius"""
        ind = np.where(distances < neighborhood_radius)
        ind = ind[1][ind[1] != BMU]
        return ind

    def get_valid_influences(self, ind, BMU):
        """determine the influences of the nodes that
        are within the current neighbor radius"""
        cur_influences = cdist(
            self.grid[ind], np.atleast_2d(
                self.grid[BMU]), metric='euclidean')
        theta_vals = np.exp(-((cur_influences**2) /
                            (2 * (cur_influences)**2)))
        return theta_vals

    def update_weights(
            self,
            ind,
            theta_vals,
            current_input_vec,
            learning_rate):
        """update the weights of the indices in the grid that
        are within the current neighborhood radisu, according 
        to the current learning rate,
        and the current influence"""
        self.grid[ind] = self.grid[ind] + learning_rate * \
            theta_vals * (current_input_vec - self.grid[ind])

    def train(self):
        for current_time in range(1, self.n_iter + 1):
            cur_neighborhood_radius = self.calc_neighbourhood_radius(
                self.sigma_0, current_time)
            current_lr = self.learning_rate(current_time)
            current_input_vector = (
                self.input_data.T[np.random.choice(self.input_data.shape[1])])
            distances = self.calculate_distances(current_input_vector)
            # get the BMU
            BMU = self.get_min_distances(distances)
            # find the indices of the weights that are within the current
            # neighbourhood radius
            ind = self.get_valid_neighbours(
                distances, BMU, cur_neighborhood_radius)
            # get the influence of those weights
            theta_vals = self.get_valid_influences(ind, BMU)
            previous_state_grid = self.grid.copy()
            # update the weights for the grid in-place
            self.update_weights(
                ind,
                theta_vals,
                current_input_vector,
                current_lr)
            # early stopping - if previous state of grid is close to grid
            # (within a tolerance threshold), break out of loop as there
            # is no need to iterate any further
            if np.allclose(
                    previous_state_grid,
                    self.grid,
                    rtol=1e-02,
                    atol=1e-02):
                break
