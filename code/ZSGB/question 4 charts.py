import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'gold']

# Rainbow + special
hue_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'gold']

gemma_coords = {
    'red': (1.2, 0.3, 0.1), 'orange': (1.0, 0.8, 0.2), 'yellow': (0.5, 1.1, 0.4),
    'green': (-0.6, 0.9, 0.3), 'blue': (-1.1, -0.2, 0.2), 'purple': (-0.8, -0.8, 0.1),
    'pink': (0.8, -0.4, -0.1), 'brown': (0.4, 0.5, -0.6), 'black': (-0.2, -0.1, -1.2),
    'white': (0.1, 0.0, 1.1), 'gray': (0.0, 0.0, 0.0), 'gold': (0.9, 1.0, 0.5)
}

llama_coords = {
    'red': (4.0, 1.0, 0.5), 'orange': (3.5, 2.5, 0.8), 'yellow': (2.0, 4.0, 1.2),
    'green': (-2.5, 3.5, 1.0), 'blue': (-4.5, -1.0, 0.5), 'purple': (-3.0, -3.0, 0.2),
    'pink': (3.0, -1.5, -0.5), 'brown': (1.5, 2.0, -2.0), 'black': (-1.0, -0.5, -4.0),
    'white': (0.5, 0.2, 4.5), 'gray': (0.0, 0.0, 0.5), 'gold': (3.2, 3.8, 1.5)
}

gpt_coords = {
    'red': (5.0, 0.0, 0.0), 'orange': (4.0, 3.0, 0.5), 'yellow': (0.0, 5.0, 1.0),
    'green': (-4.0, 3.0, 0.8), 'blue': (-5.0, 0.0, 0.0), 'purple': (-3.0, -4.0, -0.5),
    'pink': (4.0, -2.0, -0.2), 'brown': (2.0, 2.0, -1.5), 'black': (0.0, 0.0, -5.0),
    'white': (0.0, 0.0, 5.0), 'gray': (0.0, 0.0, 0.0), 'gold': (4.5, 4.0, 1.2)
}

def plot_colors(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=hue_colors, s=200)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=12, color=hue_colors[i])
    # Connect hue wheel approximate
    wheel_idx = [0,1,2,3,4,5,6,0]  # red to pink back to red
    wheel_x = [x[i] for i in wheel_idx]
    wheel_y = [y[i] for i in wheel_idx]
    wheel_z = [z[i] for i in wheel_idx]
    ax.plot(wheel_x, wheel_y, wheel_z, color='gray', alpha=0.5, linestyle='--')
    ax.set_xlabel('X: Red-Green Opponency')
    ax.set_ylabel('Y: Blue-Yellow Opponency')
    ax.set_zlabel('Z: Brightness (Black-White)')
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    plt.show()

plot_colors(gemma_coords, "Gemma Colors: Compact Hue Wheel + Achromatic Spine")
plot_colors(llama_coords, "Llama Colors: Spread Cylinder Rainbow")
plot_colors(gpt_coords, "GPT Colors: Precise Perceptual Cone")