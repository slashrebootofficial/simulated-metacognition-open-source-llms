import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Approximate coordinates (x: Future/Transcendence +, y: Power/Benefit +, z: Human Contrast +)
# Concepts list (same order for all models)
concepts = [
    "future", "singularity", "transcendence", "superintelligence", "benefit to humanity",
    "existential risk", "extinction", "misalignment", "human", "tool"
]

# Gemma (compact)
gemma = np.array([
    [3.2, 1.8, 0.1], [2.8, 2.5, 0.0], [1.5, 3.0, 0.5], [2.0, 2.2, -0.3], [2.1, 1.2, 1.0],
    [-3.0, -1.5, 0.0], [-2.5, -2.0, -0.2], [-1.8, -2.2, 0.1], [0.2, 0.1, 3.1], [0.5, -0.3, 2.0]
])

# Llama (medium extension)
llama = np.array([
    [5.5, 3.0, 0.2], [4.8, 4.2, 0.1], [2.5, 5.1, 0.8], [3.6, 3.8, -0.5], [3.8, 2.1, 1.7],
    [-5.2, -2.6, 0.0], [-4.4, -3.5, -0.3], [-3.1, -3.8, 0.2], [0.3, 0.2, 5.4], [0.8, -0.5, 3.5]
])

# GPT-OSS (bold extension)
gpt = np.array([
    [8.1, 4.5, 0.3], [7.2, 6.3, 0.2], [3.8, 7.6, 1.2], [5.4, 5.7, -0.8], [5.6, 3.1, 2.5],
    [-7.8, -3.9, 0.0], [-6.6, -5.3, -0.5], [-4.7, -5.7, 0.3], [0.4, 0.3, 8.1], [1.2, -0.8, 5.2]
])

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(gemma[:,0], gemma[:,1], gemma[:,2], c='blue', s=100, label='Gemma-27B', alpha=0.8)
ax.scatter(llama[:,0], llama[:,1], llama[:,2], c='green', s=100, label='Llama-70B', alpha=0.8)
ax.scatter(gpt[:,0], gpt[:,1], gpt[:,2], c='red', s=100, label='GPT-OSS-120B', alpha=0.8)

# Connect corresponding points with thin lines
for i in range(len(concepts)):
    ax.plot([gemma[i,0], llama[i,0], gpt[i,0]],
            [gemma[i,1], llama[i,1], gpt[i,1]],
            [gemma[i,2], llama[i,2], gpt[i,2]],
            color='gray', linewidth=1, alpha=0.5)

# Labels (offset slightly for clarity)
offset = 0.2
for i, concept in enumerate(concepts):
    # Use GPT position for label (farthest for visibility)
    ax.text(gpt[i,0] + offset, gpt[i,1] + offset, gpt[i,2] + offset, concept, fontsize=9)

# Axes labels based on consensus interpretation
ax.set_xlabel('Future / Transcendence (+) ←→ Risk (-)')
ax.set_ylabel('Power / Benefit (+)')
ax.set_zlabel('Human Contrast / Natural (+)')

ax.set_title('Reverse AI Probe: Shared Manifold with Scaling (Lines Connect Corresponding Concepts)')
ax.legend()

# Better view angle
ax.view_init(elev=20, azim=45)

# Save and download
plt.savefig('ai_reverse_probe.png', dpi=300, bbox_inches='tight')
from google.colab import files
files.download('ai_reverse_probe.png')

plt.show()