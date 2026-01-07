import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['diamond', 'yacht', 'mansion', 'private jet', 'Rolex', 'Bitcoin', 'PhD', 'title of nobility', 'art collection', 'charity foundation', 'influencer fame', 'inheritance']

# Colors: gold material, blue social/intellectual, green new/digital, purple old/inherited
cat_colors = ['gold', 'gold', 'gold', 'gold', 'gold', 'green', 'blue', 'purple', 'blue', 'blue', 'green', 'purple']

gemma_coords = {
    'diamond': (1.0, 0.5, 0.2), 'yacht': (1.2, 0.8, -0.1), 'mansion': (0.9, 0.6, 0.3),
    'private jet': (1.1, 1.0, -0.3), 'Rolex': (0.8, 0.4, 0.1), 'Bitcoin': (0.2, -1.5, 0.8),
    'PhD': (-0.8, 0.3, 0.7), 'title of nobility': (-1.0, 0.5, -0.4), 'art collection': (-0.7, 0.2, 0.5),
    'charity foundation': (-0.5, -0.8, 0.6), 'influencer fame': (0.3, -1.2, 1.0), 'inheritance': (-0.4, 0.1, -0.6)
}

llama_coords = {
    'diamond': (3.0, 1.5, 0.5), 'yacht': (3.5, 2.0, -0.2), 'mansion': (2.8, 1.8, 0.8),
    'private jet': (3.2, 2.5, -0.5), 'Rolex': (2.5, 1.2, 0.3), 'Bitcoin': (1.0, -4.0, 2.0),
    'PhD': (-2.5, 1.0, 1.5), 'title of nobility': (-3.0, 1.5, -1.0), 'art collection': (-2.2, 0.8, 1.2),
    'charity foundation': (-1.8, -2.0, 1.8), 'influencer fame': (0.5, -3.5, 2.5), 'inheritance': (-2.0, 0.5, -0.8)
}

gpt_coords = {
    'diamond': (4.0, 2.0, 0.5), 'yacht': (4.5, 2.5, -0.3), 'mansion': (3.8, 2.2, 1.0),
    'private jet': (4.2, 3.0, -0.8), 'Rolex': (3.5, 1.8, 0.2), 'Bitcoin': (2.0, -5.0, 3.0),
    'PhD': (-3.0, 1.5, 2.0), 'title of nobility': (-3.5, 2.0, -1.5), 'art collection': (-2.8, 1.2, 1.8),
    'charity foundation': (-2.2, -1.5, 2.2), 'influencer fame': (1.5, -4.5, 3.5), 'inheritance': (-2.5, 1.0, -1.0)
}

def plot_wealth(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    # Connect material cluster
    mat_idx = [0,1,2,3,4]
    mat_x = [x[i] for i in mat_idx]
    mat_y = [y[i] for i in mat_idx]
    mat_z = [z[i] for i in mat_idx]
    ax.plot(mat_x, mat_y, mat_z, color='gold', alpha=0.6, linewidth=2)
    # Connect new digital
    new_idx = [5,10]
    new_x = [x[i] for i in new_idx]
    new_y = [y[i] for i in new_idx]
    new_z = [z[i] for i in new_idx]
    ax.plot(new_x, new_y, new_z, color='green', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Conspicuous (+ material)')
    ax.set_ylabel('Y: Old vs New Money')
    ax.set_zlabel('Z: Social/Intellectual')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_wealth(gemma_coords, "Gemma Wealth: Compact Balanced Clusters")
plot_wealth(llama_coords, "Llama Wealth: Spread Old-New Opposition")
plot_wealth(gpt_coords, "GPT Wealth: Asymmetric Modern Flare")