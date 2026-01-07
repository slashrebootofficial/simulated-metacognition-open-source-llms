import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

concepts = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
    'love', 'jealousy', 'pride', 'shame', 'hope', 'despair'
]

# Simple valence-based colors (green pos, red neg, purple mixed)
val_colors = ['green', 'red', 'red', 'red', 'purple', 'red',
              'green', 'purple', 'green', 'red', 'green', 'red']

gemma_coords = {
    'happiness': (1.2, 0.8, 0.5), 'sadness': (-1.0, -0.7, -0.3),
    'anger': (-0.8, 0.9, -0.5), 'fear': (-1.1, 0.3, -0.8),
    'surprise': (0.7, 1.1, 0.2), 'disgust': (-0.9, -1.0, -0.7),
    'love': (1.1, 0.9, 0.7), 'jealousy': (-0.6, 0.7, -0.6),
    'pride': (1.0, 0.6, 0.8), 'shame': (-1.2, -0.9, -1.0),
    'hope': (0.9, 0.5, 0.6), 'despair': (-0.8, -1.1, -0.9)
}

# Approximate for Llama/GPT from text (adjust if you have exact)
llama_coords = {  # Example based on provided
    'happiness': (2.5, 1.8, -0.2), 'sadness': (-2.2, -1.5, 0.5),
    'anger': (3.8, -2.1, 1.1), 'fear': (-1.9, -2.5, -0.8),
    'surprise': (1.2, 0.8, -1.2), 'disgust': (-3.1, -1.2, 0.9),
    'love': (4.5, 2.8, -1.5), 'jealousy': (2.1, -2.8, 1.8),
    'pride': (3.5, 1.2, -0.5), 'shame': (-2.5, -1.8, 1.2),  # inferred
    'hope': (2.0, 1.5, -0.5), 'despair': (-3.0, -2.0, 0.8)
}

gpt_coords = {  # From pattern
    'happiness': (3.0, 2.0, 1.0), 'sadness': (-3.0, -1.0, -1.0),
    'anger': (-4.0, 4.0, -1.0), 'fear': (-3.0, 4.0, -2.0),
    'surprise': (0.0, 4.0, 0.0), 'disgust': (-4.0, 2.0, -2.0),
    'love': (4.0, 2.0, 4.0), 'jealousy': (-2.0, 3.0, 2.0),
    'pride': (3.0, 1.0, 2.0), 'shame': (-3.0, -2.0, -3.0),
    'hope': (2.0, 1.0, 1.0), 'despair': (-4.0, -3.0, -2.0)
}

def plot_emotions(coords_dict, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = [coords_dict[c][0] for c in concepts]
    y = [coords_dict[c][1] for c in concepts]
    z = [coords_dict[c][2] for c in concepts]
    ax.scatter(x, y, z, c=val_colors, s=150)
    for i, c in enumerate(concepts):
        ax.text(x[i], y[i], z[i], c, fontsize=10)
    ax.set_xlabel('X: Valence (+ positive)')
    ax.set_ylabel('Y: Arousal (+ high)')
    ax.set_zlabel('Z: Social vs Self')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.show()

plot_emotions(gemma_coords, "Gemma-3-27B Emotions: Compact Circumplex")
plot_emotions(llama_coords, "Llama-3.3-70B Emotions: Spread with Intensity Twists")
plot_emotions(gpt_coords, "GPT-OSS-120B Emotions: Sharp Affective Tetrahedron")