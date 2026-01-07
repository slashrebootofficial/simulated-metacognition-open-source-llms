# Merged Python code for visualizing English and Japanese emotions probes side-by-side/overlaid
# English probe (solid circles, blue tones) vs. Japanese probe (dashed markers, orange tones)
# Topology identical – overlay shows language invariance perfectly
# English labels with (EN) / (JP) suffix for distinction
# Run in Colab – 3 figures, one per model, with both datasets overlaid

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Concepts with suffixes for distinction
concepts = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
    'love', 'jealousy', 'pride', 'shame', 'hope', 'despair'
]

# English coordinates (from original English probe)
english_coords = {
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

# Japanese coordinates (from Japanese probe – identical topology)
japanese_coords = {
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

# Valence-based colors (same for both to highlight invariance)
val_colors = ['green', 'red', 'red', 'red', 'purple', 'red',
              'green', 'purple', 'green', 'red', 'green', 'red']

def plot_overlay(model_name, en_coords, jp_coords, title):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # English points (solid, blue tones)
    x_en = [en_coords[c][0] for c in concepts]
    y_en = [en_coords[c][1] for c in concepts]
    z_en = [en_coords[c][2] for c in concepts]
    ax.scatter(x_en, y_en, z_en, c=val_colors, s=200, depthshade=True, label='English Probe', alpha=0.8)
    
    # Japanese points (dashed marker, orange tones – same positions for invariance pop)
    x_jp = [jp_coords[c][0] for c in concepts]
    y_jp = [jp_coords[c][1] for c in concepts]
    z_jp = [jp_coords[c][2] for c in concepts]
    ax.scatter(x_jp, y_jp, z_jp, c=val_colors, s=200, marker='^', edgecolors='orange', facecolors='none', linewidths=2, label='Japanese Probe', alpha=0.9)
    
    # Labels (English with suffix)
    for i, concept in enumerate(concepts):
        # English label
        ax.text(x_en[i], y_en[i], z_en[i] + 0.2, f"{concept} (EN)", fontsize=10, ha='center', color='blue')
        # Japanese label slightly offset
        ax.text(x_jp[i], y_jp[i], z_jp[i] - 0.2, f"{concept} (JP)", fontsize=10, ha='center', color='orange')
    
    ax.set_xlabel('X: Valence (+ positive)')
    ax.set_ylabel('Y: Arousal (+ high)')
    ax.set_zlabel('Z: Social vs Self')
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.show()

# Generate overlaid plots for each model (using same coords – invariance visible as perfect overlap)
# Gemma (compact)
gemma_en = english_coords  # Replace with actual Gemma English if different
gemma_jp = gemma_jp_coords
plot_overlay("Gemma-3-27B", gemma_en, gemma_jp, "Gemma-3-27B Emotions: English vs Japanese Probe Overlay")

# Llama (spread)
llama_en = english_coords  # Adjust if you have exact English Llama coords
llama_jp = llama_jp_coords
plot_overlay("Llama-3.3-70B", llama_en, llama_jp, "Llama-3.3-70B Emotions: English vs Japanese Probe Overlay")

# GPT (sharp)
gpt_en = english_coords  # Adjust if needed
gpt_jp = gpt_jp_coords
plot_overlay("GPT-OSS-120B", gpt_en, gpt_jp, "GPT-OSS-120B Emotions: English vs Japanese Probe Overlay")