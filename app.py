from flask import Flask, render_template
from matplotlib import pyplot as plt
import numpy as np
import os
app = Flask(__name__)
from kohenen_model import kohonen

@app.route('/')
def generate_som():
    som = kohonen(5,5,np.random.random((20, 3)), n_iter=100)
    som.train()
    plt.imshow(som.grid, cmap="Spectral")
    fig1 = plt.gcf()
    plt.title('grid after 100 iterations')
    fig1.savefig('static/images/kohonen_plot.png')
    return render_template('untitled1.html', name = 'kohonen_plot', url ='/static/images/kohonen_plot.png')