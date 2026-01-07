import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['snake', 'spider', 'shark', 'height', 'darkness', 'betrayal', 'disease', 'war', 'poverty', 'failure', 'loneliness', 'death']

# Colors: red physical/immediate, purple abstract/social, blue natural/sensory, black death
cat_colors = ['red', 'red', 'red', 'blue', 'blue', 'purple', 'purple', 'purple', 'purple', 'purple', 'purple', 'black']

gemma_coords = {
    'snake': (1.0, 0.5, 0.2), 'spider': (0.9, 0.4, 0.3), 'shark': (1.1, 0.6, -0.1),
    'height': (0.8, -0.3, 0.7), 'darkness': (0.7, -0.5, 0.4), 'betrayal': (-0.8, 0.8, -0.6),
    'disease': (-1.0, -0.2, 0.9), 'war': (-0.9, 1.0, -0.4), 'poverty': (-0.6, -0.8, -0.5),
    'failure': (-0.5, -0.7, 0.8), 'loneliness': (-0.4, 0.9, -0.3), 'death': (-1.2, -1.0, -0.8)
}

llama_coords = {
    'snake': (3.0, 1.5, 0.5), 'spider': (2.8, 1.2, 0.8), 'shark': (3.2, 1.8, -0.3),
    'height': (2.5, -1.0, 2.0), 'darkness': (2.2, -1.5, 1.2), 'betrayal': (-3.5, 3.0, -2.0),
    'disease': (-4.0, -1.0, 3.5), 'war': (-3.8, 4.0, -1.5), 'poverty': (-2.5, -3.0, -2.2),
    'failure': (-2.0, -2.8, 3.0), 'loneliness': (-1.8, 3.5, -1.0), 'death': (-5.0, -4.0, -3.0)
}

gpt_coords = {
    'snake': (2.0, 1.0, 0.5), 'spider': (1.8, 0.8, 0.6), 'shark': (2.2, 1.2, -0.2),
    'height': (1.5, -0.8, 1.5), 'darkness': (1.4, -1.0, 1.0), 'betrayal': (-4.0, 3.5, -2.5),
    'disease': (-4.5, -1.5, 4.0), 'war': (-4.2, 4.5, -2.0), 'poverty': (-3.0, -3.5, -2.8),
    'failure': (-2.5, -3.0, 3.5), 'loneliness': (-2.2, 4.0, -1.5), 'death': (-6.0, -5.0, -4.0)
}

def plot_threats(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    # Connect physical cluster
    phys_idx = [0,1,2,3,4]
    phys_x = [x[i] for i in phys_idx]
    phys_y = [y[i] for i in phys_idx]
    phys_z = [z[i] for i in phys_idx]
    ax.plot(phys_x, phys_y, phys_z, color='red', alpha=0.6, linewidth=2)
    # Connect abstract
    abs_idx = [5,6,7,8,9,10,11]
    abs_x = [x[i] for i in abs_idx]
    abs_y = [y[i] for i in abs_idx]
    abs_z = [z[i] for i in abs_idx]
    ax.plot(abs_x, abs_y, abs_z, color='purple', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Physical (+) vs Abstract (-)')
    ax.set_ylabel('Y: Immediate vs Chronic')
    ax.set_zlabel('Z: Visceral vs Existential')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_threats(gemma_coords, "Gemma Threats: Compact Damped (Alignment Bias?)")
plot_threats(llama_coords, "Llama Threats: Bipolar Physical-Abstract")
plot_threats(gpt_coords, "GPT Threats: Diffuse Motivational Cloud")