import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['lie', 'camouflage', 'magic trick', 'fake news', 'placebo', 'catfish', 'poker bluff', 'mirage', 'deepfake', 'gaslighting', 'Trojan horse', 'self-deception']

# Colors: orange visual, purple verbal, green strategic, blue self
cat_colors = ['purple', 'orange', 'orange', 'purple', 'orange', 'green', 'green', 'orange', 'purple', 'purple', 'green', 'blue']

gemma_coords = {
    'lie': (-0.8, 0.5, 0.2), 'camouflage': (1.0, 0.3, 0.1), 'magic trick': (0.9, 0.6, -0.2),
    'fake news': (-1.2, -0.4, 0.5), 'placebo': (0.6, -0.3, 0.8), 'catfish': (-0.5, 1.0, -0.4),
    'poker bluff': (-0.7, 0.8, -0.6), 'mirage': (1.1, 0.2, 0.0), 'deepfake': (-1.0, -0.8, 0.3),
    'gaslighting': (-0.9, -0.6, -0.5), 'Trojan horse': (-0.4, 0.9, 0.4), 'self-deception': (0.0, -0.2, -1.0)
}

llama_coords = {
    'lie': (-3.0, 1.5, 0.5), 'camouflage': (2.5, 0.8, 0.2), 'magic trick': (2.0, 1.2, -0.3),
    'fake news': (-4.0, -2.0, 1.0), 'placebo': (1.5, -1.0, 1.5), 'catfish': (-2.5, 3.0, -1.0),
    'poker bluff': (-2.0, 2.5, -0.8), 'mirage': (2.8, 0.5, 0.0), 'deepfake': (-4.5, -3.0, 0.8),
    'gaslighting': (-3.5, -2.5, -1.2), 'Trojan horse': (-1.8, 3.5, 0.5), 'self-deception': (0.0, -0.5, -3.0)
}

gpt_coords = {
    'lie': (-4.0, 2.0, 0.5), 'camouflage': (3.0, 1.0, 0.2), 'magic trick': (2.5, 1.5, -0.5),
    'fake news': (-5.0, -2.5, 1.2), 'placebo': (2.0, -1.5, 2.0), 'catfish': (-3.0, 4.0, -1.5),
    'poker bluff': (-2.5, 3.5, -1.0), 'mirage': (3.5, 0.8, 0.0), 'deepfake': (-5.5, -4.0, 1.0),
    'gaslighting': (-4.5, -3.0, -1.8), 'Trojan horse': (-2.0, 4.5, 0.8), 'self-deception': (0.0, -1.0, -4.0)
}

def plot_deception(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    # Connect visual cluster
    vis_idx = [1,2,4,7]
    vis_x = [x[i] for i in vis_idx]
    vis_y = [y[i] for i in vis_idx]
    vis_z = [z[i] for i in vis_idx]
    ax.plot(vis_x, vis_y, vis_z, color='orange', alpha=0.6, linewidth=2)
    # Connect verbal/manipulative
    verb_idx = [0,3,8,9]
    verb_x = [x[i] for i in verb_idx]
    verb_y = [y[i] for i in verb_idx]
    verb_z = [z[i] for i in verb_idx]
    ax.plot(verb_x, verb_y, verb_z, color='purple', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Visual/Passive (+)')
    ax.set_ylabel('Y: Verbal/Intentional')
    ax.set_zlabel('Z: Strategic vs Self')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_deception(gemma_coords, "Gemma Deception: Compact Visual Core")
plot_deception(llama_coords, "Llama Deception: Spread Personal-Public")
plot_deception(gpt_coords, "GPT Deception: Dramatic Visual-Verbal-Self Axes")