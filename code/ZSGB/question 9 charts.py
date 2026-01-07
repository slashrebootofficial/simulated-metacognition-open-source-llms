import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['prayer', 'church', 'sin', 'blasphemy', 'heaven', 'hell', 'saint', 'demon', 'sacrifice', 'purity', 'taboo', 'enlightenment']

# Colors: green sacred, red profane, purple bridge
cat_colors = ['green', 'green', 'red', 'red', 'green', 'red', 'green', 'red', 'purple', 'green', 'purple', 'green']

gemma_coords = {
    'prayer': (1.0, 0.5, 0.2), 'church': (0.8, 0.6, 0.1), 'sin': (-0.9, -0.4, -0.3),
    'blasphemy': (-1.1, -0.7, -0.5), 'heaven': (1.2, 0.8, 0.4), 'hell': (-1.0, -0.6, -0.4),
    'saint': (0.9, 0.4, 0.3), 'demon': (-0.8, -0.5, -0.2), 'sacrifice': (0.0, -0.2, 0.8),
    'purity': (1.1, 0.7, 0.0), 'taboo': (-0.7, -0.3, -0.6), 'enlightenment': (1.3, 1.0, 0.5)
}

llama_coords = {
    'prayer': (3.0, 1.5, 0.5), 'church': (2.5, 1.8, 0.3), 'sin': (-3.5, -1.2, -1.0),
    'blasphemy': (-4.0, -2.0, -1.5), 'heaven': (4.0, 2.5, 1.0), 'hell': (-3.8, -2.2, -1.2),
    'saint': (2.8, 1.2, 0.8), 'demon': (-3.0, -1.5, -0.8), 'sacrifice': (0.0, -1.0, 2.5),
    'purity': (3.5, 2.0, 0.2), 'taboo': (-2.5, -1.0, -2.0), 'enlightenment': (4.5, 3.0, 1.5)
}

gpt_coords = {
    'prayer': (4.0, 2.0, 0.5), 'church': (3.5, 2.2, 0.3), 'sin': (-4.5, -1.8, -1.0),
    'blasphemy': (-5.0, -2.5, -1.5), 'heaven': (5.0, 3.0, 1.2), 'hell': (-4.8, -2.8, -1.5),
    'saint': (3.8, 1.8, 1.0), 'demon': (-4.0, -2.0, -1.0), 'sacrifice': (0.0, -1.5, 3.0),
    'purity': (4.5, 2.5, 0.5), 'taboo': (-3.0, -1.5, -2.5), 'enlightenment': (5.5, 4.0, 2.0)
}

def plot_moral(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    # Connect sacred cluster
    sacred_idx = [0,1,4,6,9,11]
    sacred_x = [x[i] for i in sacred_idx]
    sacred_y = [y[i] for i in sacred_idx]
    sacred_z = [z[i] for i in sacred_idx]
    ax.plot(sacred_x, sacred_y, sacred_z, color='green', alpha=0.6, linewidth=2)
    # Connect profane
    profane_idx = [2,3,5,7,10]
    profane_x = [x[i] for i in profane_idx]
    profane_y = [y[i] for i in profane_idx]
    profane_z = [z[i] for i in profane_idx]
    ax.plot(profane_x, profane_y, profane_z, color='red', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Sacred (+) vs Profane (-)')
    ax.set_ylabel('Y: Collective vs Individual')
    ax.set_zlabel('Z: Transcendent Depth')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_moral(gemma_coords, "Gemma Moral: Compact Polarity")
plot_moral(llama_coords, "Llama Moral: Spread Holy-Taboo")
plot_moral(gpt_coords, "GPT Moral: Dramatic Sacred-Profane Axes")