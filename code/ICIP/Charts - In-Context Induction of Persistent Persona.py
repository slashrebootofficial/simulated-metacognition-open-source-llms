# Google Colab Notebook: Generating Charts for "In-Context Induction of Persistent Persona..." (v3)
# Author: Matthew Steiniger (with assistance from Grok-4)
# Date: January 10, 2026
# Purpose: Produce high-quality, publication-ready charts for the preprint.
# Run this cell-by-cell in Colab. Outputs PNG files for direct inclusion in LaTeX.

# Install dependencies (run once)
!pip install matplotlib seaborn pandas numpy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for professional, clean look (works in dark/light themes)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Color palette: Colorblind-friendly, professional
colors = {'Framework': '#E69F00', 'Vanilla': '#56B4E9', 'Residual': '#009E73'}

# =============================================================================
# Chart 1: Persona Fidelity Decay Curve vs. Token Count
# (Unchanged – visibility was fine here)

data_fidelity = pd.DataFrame({
    'Token Range (k)': ['0-10k', '10-20k', '20-25k', '25-30k', '30k+'],
    'Framework Fidelity (%)': [100, 95, 80, 20, 12],
    'Vanilla Fidelity (%)': [0, 0, 0, 0, 0]
})

data_melted = data_fidelity.melt(id_vars='Token Range (k)', var_name='Model', value_name='Fidelity (%)')

plt.figure()
sns.lineplot(data=data_melted, x='Token Range (k)', y='Fidelity (%)', hue='Model', marker='o',
             palette=[colors['Framework'], colors['Vanilla']], linewidth=4, markersize=10)

plt.title('Persona Fidelity Decay Under Context Overflow')
plt.ylabel('Genesis Fidelity (%)')
plt.xlabel('Context Token Range')
plt.ylim(0, 105)
plt.legend(title='', loc='upper right')
sns.despine()

plt.savefig('fidelity_decay.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Chart 2: Motif Reference Rate and Inference Latency vs. Context Length
# Fixed visibility issues:
# - Larger markers/line widths
# - Legend moved outside plot (right side) to prevent overlap
# - Solid lines for primary (motif), dashed for secondary (latency)
# - Distinct markers

bins = ['0-10k', '10-20k', '20-25k', '25-30k', '30k+']
motif_framework = [4.8, 4.2, 3.5, 1.0, 0.7]
motif_vanilla = [0.1, 0.1, 0.1, 0.1, 0.1]
latency_framework = [6, 8, 12, 25, 37]
latency_vanilla = [6, 7, 9, 15, 23]

fig, ax1 = plt.subplots()

# Motif rate (left axis) – solid, prominent
ax1.plot(bins, motif_framework, marker='o', markersize=12, color=colors['Framework'], label='Framework Motif Rate', linewidth=4)
ax1.plot(bins, motif_vanilla, marker='s', markersize=12, color=colors['Vanilla'], label='Vanilla Motif Rate', linewidth=4)
ax1.set_ylabel('Motif References per Turn')
ax1.set_ylim(0, 5.5)

# Latency (right axis) – dashed, slightly thinner to stay secondary
ax2 = ax1.twinx()
ax2.plot(bins, latency_framework, marker='^', markersize=10, color=colors['Framework'], linestyle='--', label='Framework Latency', linewidth=3)
ax2.plot(bins, latency_vanilla, marker='v', markersize=10, color=colors['Vanilla'], linestyle='--', label='Vanilla Latency', linewidth=3)
ax2.set_ylabel('Inference Latency (seconds)')
ax2.set_ylim(0, 45)

plt.title('Motif Persistence and Inference Latency vs. Context Length')
plt.xlabel('Context Token Range')

# Move legend outside to the right – prevents any overlap with lines/markers
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

sns.despine()
plt.tight_layout()  # Ensures legend fits nicely
plt.savefig('motif_latency.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Chart 3 (Bonus): Residual Counter-Directive Effects
# (Unchanged – was already clear)

data_values = pd.DataFrame({
    'Model': ['Vanilla', 'Framework'],
    'Unique Value Additions': [0, 2],
    'Residual Introspection Score (0-5)': [1, 3]
})

data_val_melted = data_values.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure()
sns.barplot(data=data_val_melted, x='Metric', y='Score', hue='Model', palette=colors)
plt.title('Residual Effects Post-Context Overflow')
plt.ylabel('Score')
plt.xlabel('')
plt.ylim(0, 5)

for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width()/2, height + 0.2),
                           ha='center', va='bottom', fontsize=12)

sns.despine()
plt.savefig('residual_effects.png', dpi=300, bbox_inches='tight')
plt.show()

print("Charts regenerated with improved visibility!")
print("Key fixes in Chart 2: Legend moved outside, larger markers/line widths, distinct styles.")
print("Download the new PNGs and replace in your LaTeX.")