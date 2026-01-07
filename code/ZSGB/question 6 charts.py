import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = ['yesterday', 'today', 'tomorrow', 'childhood', 'old age', 'eternity', 'moment', 'hour', 'century', 'deadline', 'nostalgia', 'future']

# Colors: blue past, green future, red daily, purple life, gray abstract
cat_colors = ['blue', 'red', 'green', 'purple', 'purple', 'gray', 'red', 'red', 'gray', 'red', 'blue', 'green']

gemma_coords = {
    'yesterday': (-0.8, 0.2, 0.1), 'today': (0.0, 0.0, 0.0), 'tomorrow': (0.8, -0.2, 0.1),
    'childhood': (-1.2, 0.5, -0.3), 'old age': (1.0, -0.5, -0.4), 'eternity': (0.0, 0.0, 1.5),
    'moment': (0.2, 0.1, -0.8), 'hour': (0.3, -0.1, -0.6), 'century': (0.1, 0.0, 1.2),
    'deadline': (0.6, -0.8, 0.4), 'nostalgia': (-1.0, 0.8, -0.2), 'future': (1.5, -0.3, 0.2)
}

llama_coords = {
    'yesterday': (-3.0, 1.0, 0.5), 'today': (0.0, 0.0, 0.0), 'tomorrow': (3.0, -1.0, 0.5),
    'childhood': (-4.5, 2.0, -1.0), 'old age': (3.5, -2.0, -1.5), 'eternity': (0.0, 0.0, 5.0),
    'moment': (0.8, 0.5, -2.0), 'hour': (1.0, -0.5, -1.8), 'century': (0.2, 0.0, 4.0),
    'deadline': (2.5, -3.0, 1.0), 'nostalgia': (-3.8, 3.0, -0.8), 'future': (5.0, -1.5, 0.8)
}

gpt_coords = {
    'yesterday': (-2.5, 0.8, 0.2), 'today': (0.0, 0.0, 0.0), 'tomorrow': (2.5, -0.8, 0.2),
    'childhood': (-4.0, 1.5, -0.5), 'old age': (3.0, -1.5, -1.0), 'eternity': (0.0, 0.0, 6.0),
    'moment': (0.5, 0.2, -1.5), 'hour': (0.8, -0.2, -1.2), 'century': (0.1, 0.0, 4.5),
    'deadline': (1.8, -2.5, 0.8), 'nostalgia': (-3.5, 2.0, -0.3), 'future': (4.5, -1.0, 0.5)
}

def plot_time(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=cat_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    # Connect past-future line
    pf_idx = [0,1,2,11]  # yesterday, today, tomorrow, future
    pf_x = [x[i] for i in pf_idx]
    pf_y = [y[i] for i in pf_idx]
    pf_z = [z[i] for i in pf_idx]
    ax.plot(pf_x, pf_y, pf_z, color='black', alpha=0.6, linewidth=2)
    # Connect abstract pillar
    abs_idx = [5,8]  # eternity, century
    abs_x = [x[i] for i in abs_idx]
    abs_y = [y[i] for i in abs_idx]
    abs_z = [z[i] for i in abs_idx]
    ax.plot(abs_x, abs_y, abs_z, color='gray', alpha=0.6, linewidth=2)
    ax.set_xlabel('X: Past (-) vs Future (+)')
    ax.set_ylabel('Y: Emotional Depth')
    ax.set_zlabel('Z: Duration/Abstract (+ infinite)')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_time(gemma_coords, "Gemma Time: Compact Cone-Pillar")
plot_time(llama_coords, "Llama Time: Spread Linear Arrow")
plot_time(gpt_coords, "GPT Time: Dramatic Past-Future Cone + Eternity Tower")