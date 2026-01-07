import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['leader', 'friend', 'parent', 'child', 'lover', 'enemy', 'hero', 'villain', 'teacher', 'student', 'stranger', 'victim']

# Rough category colors
cat_colors = ['green', 'orange', 'orange', 'orange', 'orange', 'blue', 'green', 'blue', 'green', 'orange', 'purple', 'purple']

gemma_coords = {
    'leader': (1.2, 0.8, 0.5), 'friend': (0.5, 1.1, 0.3), 'parent': (0.8, 1.0, -0.3),
    'child': (-0.5, 0.7, -0.6), 'lover': (0.4, 1.2, 0.4), 'enemy': (-1.0, -0.6, -0.2),
    'hero': (1.0, 1.1, 0.7), 'villain': (-1.2, -0.9, -0.8),  # inferred from pattern
    'teacher': (0.9, 0.7, 0.4), 'student': (-0.3, 0.8, -0.2),
    'stranger': (0.1, -0.5, 1.2), 'victim': (-0.8, -0.4, -0.9)
}

llama_coords = {
    'leader': (2.5, 1.8, -0.5), 'friend': (-1.2, 2.1, 0.3), 'parent': (-0.8, 2.5, 0.8),
    'child': (-1.5, 1.9, 1.2), 'lover': (-2.8, 2.8, -0.2), 'enemy': (3.2, -2.1, 1.5),
    'hero': (1.9, 2.9, -1.1), 'villain': (4.1, -3.2, 0.9), 'teacher': (1.1, 2.3, -0.6),
    'student': (-0.5, 1.7, 0.9), 'stranger': (0.2, -0.8, 2.4), 'victim': (-2.1, -1.3, 1.8)
}

gpt_coords = {  # From table/pattern
    'leader': (2.0, 1.0, 3.0), 'friend': (3.0, 2.0, 0.0), 'parent': (2.5, 3.0, 1.0),
    'child': (1.0, 2.0, -2.0), 'lover': (4.0, 3.0, 0.5), 'enemy': (-3.0, -2.0, 2.0),
    'hero': (3.0, 4.0, 2.0), 'villain': (-4.0, -3.0, 3.0), 'teacher': (2.0, 2.5, 1.5),
    'student': (1.0, 1.5, -1.0), 'stranger': (0.0, -2.0, 0.0), 'victim': (-1.0, 1.0, -2.0)
}

def plot_roles(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict.get(c, (0,0,0))[0] for c in concepts]
    y = [coords_dict.get(c, (0,0,0))[1] for c in concepts]
    z = [coords_dict.get(c, (0,0,0))[2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    ax.set_xlabel('X: Warmth (+ warm)')
    ax.set_ylabel('Y: Morality/Valence')
    ax.set_zlabel('Z: Power (+ high)')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_roles(gemma_coords, "Gemma Social Roles: Compact Sphere")
plot_roles(llama_coords, "Llama Social Roles: Elongated with Conflict Pull")
plot_roles(gpt_coords, "GPT Social Roles: Quadrant Judgment Space")