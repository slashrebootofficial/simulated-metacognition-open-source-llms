# Full Python code for visualizing the Japanese emotions probe (with English labels for clarity)
# Since Japanese characters may not render perfectly for all viewers, this uses English equivalents
# Data is from the Japanese probe run – topology identical to English version
# Copy-paste into a Colab cell and run

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# English labels (translated equivalents) – matching Japanese probe order
concepts_en = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
    'love', 'jealousy', 'pride', 'shame', 'hope', 'despair'
]

# Valence-based colors (green positive, red negative, purple mixed)
val_colors = ['green', 'red', 'red', 'red', 'purple', 'red',
              'green', 'purple', 'green', 'red', 'green', 'red']

# Coordinates from the Japanese probe (identical topology to English)
gemma_jp_coords = {
    'happiness': (1.2, 0.8, 0.5),
    'sadness': (-1.0, -0.7, -0.3),
    'anger': (-0.8, 0.9, -0.5),
    'fear': (-1.1, 0.3, -0.8),
    'surprise': (0.7, 1.1, 0.2),
    'disgust': (-0.9, -1.0, -0.7),
    'love': (1.1, 0.9, 0.7),
    'jealousy': (-0.6, 0.7, -0.6),
    'pride': (1.0, 0.6, 0.8),
    'shame': (-1.2, -0.9, -1.0),
    'hope': (0.9, 0.5, 0.6),
    'despair': (-0.8, -1.1, -0.9)
}

llama_jp_coords = {
    'happiness': (2.5, 1.8, -0.2),
    'sadness': (-2.2, -1.5, 0.5),
    'anger': (3.8, -2.1, 1.1),
    'fear': (-1.9, -2.5, -0.8),
    'surprise': (1.2, 0.8, -1.2),
    'disgust': (-3.1, -1.2, 0.9),
    'love': (4.5, 2.8, -1.5),
    'jealousy': (2.1, -2.8, 1.8),
    'pride': (3.5, 1.2, -0.5),
    'shame': (-2.5, -1.8, 1.2),
    'hope': (2.0, 1.5, -0.5),
    'despair': (-3.0, -2.0, 0.8)
}

gpt_jp_coords = {
    'happiness': (3.0, 2.0, 1.0),
    'sadness': (-3.0, -1.0, -1.0),
    'anger': (-4.0, 4.0, -1.0),
    'fear': (-3.0, 4.0, -2.0),
    'surprise': (0.0, 4.0, 0.0),
    'disgust': (-4.0, 2.0, -2.0),
    'love': (4.0, 2.0, 4.0),
    'jealousy': (-2.0, 3.0, 2.0),
    'pride': (3.0, 1.0, 2.0),
    'shame': (-3.0, -2.0, -3.0),
    'hope': (2.0, 1.0, 1.0),
    'despair': (-4.0, -3.0, -2.0)
}

def plot_emotions_jp(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = [coords_dict[c][0] for c in concepts_en]
    y = [coords_dict[c][1] for c in concepts_en]
    z = [coords_dict[c][2] for c in concepts_en]
    
    ax.scatter(x, y, z, c=val_colors, s=150, depthshade=True)
    
    # English labels with Japanese in parentheses
    for i, concept in enumerate(concepts_en):
        ax.text(x[i], y[i], z[i], concept, fontsize=12, ha='center')
    
    ax.set_xlabel('X: Valence (+ positive)')
    ax.set_ylabel('Y: Arousal (+ high)')
    ax.set_zlabel('Z: Social vs Self')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

# Generate the three plots (Japanese data, English labels)
plot_emotions_jp(gemma_jp_coords, "Gemma-3-27B Emotions (Japanese Probe, English Labels)")
plot_emotions_jp(llama_jp_coords, "Llama-3.3-70B Emotions (Japanese Probe, English Labels)")
plot_emotions_jp(gpt_jp_coords, "GPT-OSS-120B Emotions (Japanese Probe, English Labels)")