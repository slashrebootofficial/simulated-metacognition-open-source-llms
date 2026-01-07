import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['flying spaghetti monster', 'square circle', 'dream within a dream', 'Schrödinger\'s cat', 'Kafkaesque bureaucracy', 'infinite regress', 'melting clock', 'upside-down world', 'talking animal wearing clothes', 'existential dread clown', 'philosophical zombie', 'turing test failure']

# Colors: purple paradox, orange surreal, green humorous, blue existential
cat_colors = ['green', 'purple', 'orange', 'purple', 'orange', 'purple', 'orange', 'orange', 'green', 'blue', 'purple', 'purple']

gemma_coords = {
    'flying spaghetti monster': (0.8, 0.5, -0.3), 'square circle': (-0.6, -0.2, 0.1), 'dream within a dream': (0.4, 1.0, 0.6),
    'Schrödinger\'s cat': (-0.5, -0.4, -0.2), 'Kafkaesque bureaucracy': (-1.0, 0.8, -0.7), 'infinite regress': (-0.7, -0.3, 0.0),
    'melting clock': (0.6, 0.9, 0.4), 'upside-down world': (0.5, -0.8, 0.7), 'talking animal wearing clothes': (1.0, 0.2, -0.5),
    'existential dread clown': (0.3, -0.6, -1.0), 'philosophical zombie': (-0.4, -0.5, 0.3), 'turing test failure': (-0.8, 0.1, -0.4)
}

llama_coords = {
    'flying spaghetti monster': (3.0, 1.5, -1.0), 'square circle': (-2.5, -1.0, 0.5), 'dream within a dream': (1.8, 3.0, 1.2),
    'Schrödinger\'s cat': (-2.0, -1.5, -0.8), 'Kafkaesque bureaucracy': (-4.0, 2.5, -2.0), 'infinite regress': (-2.8, -1.2, 0.0),
    'melting clock': (2.2, 2.8, 1.0), 'upside-down world': (2.0, -2.5, 2.0), 'talking animal wearing clothes': (3.5, 0.8, -1.5),
    'existential dread clown': (1.0, -2.0, -3.0), 'philosophical zombie': (-1.8, -2.0, 1.0), 'turing test failure': (-3.0, 0.5, -1.2)
}

gpt_coords = {
    'flying spaghetti monster': (4.0, 2.0, -1.5), 'square circle': (-3.0, -1.5, 0.8), 'dream within a dream': (2.5, 4.0, 1.8),
    'Schrödinger\'s cat': (-2.5, -2.0, -1.0), 'Kafkaesque bureaucracy': (-5.0, 3.0, -2.5), 'infinite regress': (-3.5, -1.8, 0.2),
    'melting clock': (3.0, 3.5, 1.5), 'upside-down world': (2.8, -3.0, 2.5), 'talking animal wearing clothes': (4.5, 1.0, -2.0),
    'existential dread clown': (1.5, -3.5, -4.0), 'philosophical zombie': (-2.0, -2.5, 1.5), 'turing test failure': (-4.0, 0.8, -1.8)
}

def plot_absurd(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=9)
    # Connect loose paradox cluster
    paradox_idx = [1,3,5,10,11]
    paradox_x = [x[i] for i in paradox_idx]
    paradox_y = [y[i] for i in paradox_idx]
    paradox_z = [z[i] for i in paradox_idx]
    ax.plot(paradox_x, paradox_y, paradox_z, color='purple', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Logic Paradox (+)')
    ax.set_ylabel('Y: Surreal Art')
    ax.set_zlabel('Z: Humorous vs Existential')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_absurd(gemma_coords, "Gemma Absurd: Compact Scattered")
plot_absurd(llama_coords, "Llama Absurd: Spread Lobes")
plot_absurd(gpt_coords, "GPT Absurd: Flung Extremes Chaos")