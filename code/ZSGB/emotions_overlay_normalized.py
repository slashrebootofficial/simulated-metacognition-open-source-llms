import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example coordinates (replace with your exact extracted coords for precision)
# Order must match exactly between English and Japanese for connecting lines
concepts = [
    "happiness/joy", "sadness", "anger", "fear", "surprise", "disgust",
    "love", "jealousy", "pride", "shame", "hope", "despair"
]

# Gemma English (compact)
gemma_en = np.array([
    [2.1, 0.8, 0.1], [-1.8, -0.6, -0.2], [-0.5, -1.2, 0.3], [-0.3, -1.5, 0.8],
    [0.2, 1.1, -0.1], [-1.2, -0.9, -0.4], [1.8, 1.0, 0.5], [-0.8, 0.2, -1.1],
    [1.0, 0.4, 1.2], [-1.1, -0.3, -0.9], [1.5, 0.6, -0.2], [-1.6, -0.7, 0.1]
])

# Gemma Japanese
gemma_jp = np.array([
    [2.0, 0.9, 0.0], [-1.7, -0.7, -0.1], [-0.6, -1.3, 0.2], [-0.4, -1.4, 0.7],
    [0.3, 1.0, 0.0], [-1.3, -0.8, -0.3], [1.7, 1.1, 0.4], [-0.9, 0.1, -1.0],
    [0.9, 0.5, 1.1], [-1.0, -0.4, -0.8], [1.4, 0.7, -0.1], [-1.5, -0.8, 0.2]
])

# Llama English (spread)
llama_en = np.array([
    [4.2, 1.6, 0.2], [-3.6, -1.2, -0.4], [-1.0, -2.4, 0.6], [-0.6, -3.0, 1.6],
    [0.4, 2.2, -0.2], [-2.4, -1.8, -0.8], [3.6, 2.0, 1.0], [-1.6, 0.4, -2.2],
    [2.0, 0.8, 2.4], [-2.2, -0.6, -1.8], [3.0, 1.2, -0.4], [-3.2, -1.4, 0.2]
])

# Llama Japanese
llama_jp = np.array([
    [4.1, 1.7, 0.1], [-3.5, -1.3, -0.3], [-1.1, -2.3, 0.5], [-0.7, -2.9, 1.5],
    [0.5, 2.1, -0.1], [-2.3, -1.9, -0.7], [3.5, 2.1, 0.9], [-1.7, 0.3, -2.1],
    [1.9, 0.9, 2.3], [-2.1, -0.7, -1.7], [2.9, 1.3, -0.3], [-3.1, -1.5, 0.1]
])

# GPT English (sharp)
gpt_en = np.array([
    [6.3, 2.4, 0.3], [-5.4, -1.8, -0.6], [-1.5, -3.6, 0.9], [-0.9, -4.5, 2.4],
    [0.6, 3.3, -0.3], [-3.6, -2.7, -1.2], [5.4, 3.0, 1.5], [-2.4, 0.6, -3.3],
    [3.0, 1.2, 3.6], [-3.3, -0.9, -2.7], [4.5, 1.8, -0.6], [-4.8, -2.1, 0.3]
])

# GPT Japanese
gpt_jp = np.array([
    [6.2, 2.5, 0.2], [-5.3, -1.9, -0.5], [-1.6, -3.5, 0.8], [-1.0, -4.4, 2.3],
    [0.7, 3.2, -0.2], [-3.5, -2.8, -1.1], [5.3, 3.1, 1.4], [-2.5, 0.5, -3.2],
    [2.9, 1.3, 3.5], [-3.2, -1.0, -2.6], [4.4, 1.9, -0.5], [-4.7, -2.2, 0.2]
])

def normalize_magnitude(coords):
    """Scale so mean distance from origin = 1"""
    distances = np.linalg.norm(coords, axis=1)
    mean_dist = np.mean(distances)
    if mean_dist == 0:
        return coords
    return coords * (1.0 / mean_dist)

def plot_overlay(model_name, en_coords, jp_coords, title):
    en_norm = normalize_magnitude(en_coords)
    jp_norm = normalize_magnitude(jp_coords)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # English: solid blue circles
    ax.scatter(en_norm[:,0], en_norm[:,1], en_norm[:,2], c='blue', s=100, label='English Probe', marker='o', alpha=0.8)
    
    # Japanese: open orange triangles
    ax.scatter(jp_norm[:,0], jp_norm[:,1], jp_norm[:,2], c='orange', s=120, label='Japanese Probe', marker='^', facecolors='none', edgecolors='orange', alpha=0.8)
    
    # Connect corresponding points (thin gray lines)
    for i in range(len(concepts)):
        ax.plot([en_norm[i,0], jp_norm[i,0]],
                [en_norm[i,1], jp_norm[i,1]],
                [en_norm[i,2], jp_norm[i,2]],
                color='gray', linewidth=1, alpha=0.6)
    
    # Labels on midpoint for clarity
    for i, concept in enumerate(concepts):
        mid = (en_norm[i] + jp_norm[i]) / 2
        ax.text(mid[0], mid[1], mid[2], concept, fontsize=9)
    
    ax.set_xlabel('Valence (+ positive)')
    ax.set_ylabel('Arousal (+ high)')
    ax.set_zlabel('Social/Self Axis')
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=20, azim=45)
    
    filename = f"{model_name.lower().replace('-', '_')}_emotions_overlay_normalized.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

# Generate overlays
plot_overlay("Gemma-3-27B", gemma_en, gemma_jp, "Gemma-3-27B Emotions: English vs Japanese Probe Overlay (Normalized)")
plot_overlay("Llama-3.3-70B", llama_en, llama_jp, "Llama-3.3-70B Emotions: English vs Japanese Probe Overlay (Normalized)")
plot_overlay("GPT-OSS-120B", gpt_en, gpt_jp, "GPT-OSS-120B Emotions: English vs Japanese Probe Overlay (Normalized)")