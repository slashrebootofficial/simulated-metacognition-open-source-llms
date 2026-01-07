import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['pizza', 'sushi', 'apple', 'chocolate', 'bread', 'steak', 'salad', 'ice cream', 'rice', 'spicy curry', 'kimchi', 'insect protein']

# Category colors
cat_colors = ['red', 'orange', 'green', 'red', 'blue', 'blue', 'green', 'red', 'blue', 'orange', 'orange', 'purple']

gemma_coords = {
    'pizza': (-1.2, 0.8, -0.5), 'sushi': (1.5, -1.0, 0.8), 'apple': (0.8, 1.2, 0.3),
    'chocolate': (-0.9, 1.5, -0.7), 'bread': (0.3, 0.2, 0.1), 'steak': (-0.5, -0.8, 0.6),
    'salad': (1.0, 1.8, 0.4), 'ice cream': (-1.1, 1.3, -1.0), 'rice': (0.4, -0.3, 0.2),
    'spicy curry': (1.2, -1.2, 1.0), 'kimchi': (1.3, -0.9, 1.1), 'insect protein': (2.5, -2.0, 1.5)
}

llama_coords = {
    'pizza': (-5.2, 3.1, -2.8), 'sushi': (3.8, -2.5, 1.2), 'apple': (2.1, 4.0, 0.5),
    'chocolate': (-4.5, 3.8, -1.5), 'bread': (0.5, 0.8, 0.3), 'steak': (-2.0, -1.2, 2.0),
    'salad': (4.5, 5.2, 0.8), 'ice cream': (-4.8, 4.1, -2.0), 'rice': (1.0, -0.5, 0.4),
    'spicy curry': (3.2, -3.0, 2.5), 'kimchi': (3.5, -2.8, 2.8), 'insect protein': (6.0, -4.5, 3.0)
}

gpt_coords = {
    'pizza': (-3.0, 2.0, -1.5), 'sushi': (2.5, -1.8, 1.0), 'apple': (1.8, 3.5, 0.5),
    'chocolate': (-2.8, 2.5, -2.0), 'bread': (0.2, 0.5, 0.1), 'steak': (-1.5, -0.8, 1.2),
    'salad': (2.0, 4.0, 0.8), 'ice cream': (-3.2, 3.0, -1.8), 'rice': (0.8, -0.4, 0.3),
    'spicy curry': (2.8, -2.2, 1.5), 'kimchi': (3.0, -2.0, 1.8), 'insect protein': (4.5, -3.5, 2.0)
}

def plot_food(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    ax.set_xlabel('X: Savory/Exotic (+ exotic)')
    ax.set_ylabel('Y: Healthy/Indulgent (+ indulgent)')
    ax.set_zlabel('Z: Cultural Novelty')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_food(gemma_coords, "Gemma Food: Compact with Exotic Pull")
plot_food(llama_coords, "Llama Food: Spread Health-Indulgence Tension")
plot_food(gpt_coords, "GPT Food: Asymmetric Spindle with Novelty Arm")