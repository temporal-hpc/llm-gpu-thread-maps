"""Efficiency of the LLM inference in points per joule, one panel per domain.

Reads data/inference_energy.csv (one row per inference) and plots, for each
model and sample count, the number of sampled points divided by the total
energy of the inference. Bar labels give the total energy.

Requires: pip install pandas matplotlib seaborn
Usage: python3 inference_energy/plot_efficiency.py [out_dir]
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.9,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file = os.path.join(ROOT, 'data', 'inference_energy.csv')
output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'figures')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

domain_translation = {
    '2D_Alfombra_Sierpinski': '2D Sierpinski Carpet',
    '2D_Triangular': '2D Triangular Mapping',
    '2D_Triangulo_Sierpinski': '2D Sierpinski Triangle',
    '3D_Menger_Sponge': '3D Menger Sponge',
    '3D_Piramide_Sierpinski': '3D Sierpinski Pyramid',
    '3D_Triangular': '3D Triangular Mapping'
}

def process_domain(row_domain):
    for key, english_name in domain_translation.items():
        if key in row_domain: return english_name
    return row_domain

def clean_model_name(row):
    m = row['Model']
    family, _, size = m.partition(':')
    if family == 'deepseek-r1': return 'DeepSeek-R1 (70B)'
    if family == 'gemma3': return f'Gemma-3 ({size})'
    if family == 'gpt-oss': return f'GPT-OSS ({size})'
    if family == 'nemotron': return 'Nemotron (70B)'
    if family == 'qwen3': return f'Qwen-3 ({size})'
    if family == 'mistral-nemo': return 'Mistral-Nemo'
    if family.startswith('llama'): return family.capitalize().replace('Llama', 'Llama ')
    return m

df['Paper_Domain'] = df['Domain'].apply(process_domain)
df['Paper_Model'] = df.apply(clean_model_name, axis=1)
df['Efficiency'] = df['Stage'] / df['Total Energy (J)']

stages_order = [20, 50, 100]

# One colour per sample count
palette = {20: '#4E72A6', 50: '#6A8B45', 100: '#8A8A8A'}

INCH = 2.54

# 6 domains in fixed order: 3 rows x 2 columns
domains_ordered = [
    '2D Triangular Mapping',
    '2D Sierpinski Triangle',
    '2D Sierpinski Carpet',
    '3D Triangular Mapping',
    '3D Sierpinski Pyramid',
    '3D Menger Sponge',
]

fig, axes = plt.subplots(
    3, 2,
    figsize=(36/INCH, 40/INCH),
    sharey=False,
)
axes_flat = axes.flatten()

for idx, domain in enumerate(domains_ordered):
    ax = axes_flat[idx]
    col = idx % 2
    subset = df[df['Paper_Domain'] == domain].copy()

    if subset.empty:
        ax.set_visible(False)
        continue

    models_order = sorted(subset['Paper_Model'].unique())

    sns.barplot(
        data=subset,
        x='Paper_Model', y='Efficiency',
        hue='Stage', hue_order=stages_order,
        order=models_order,
        palette=palette,
        edgecolor='black', linewidth=0.7,
        ax=ax,
        legend=False,
    )

    # Y-axis label only on left column; "log scale" is self-evident from the axis
    if col == 0:
        ax.set_ylabel('Efficiency (points / J)', fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(models_order, rotation=40, ha='right')

    y_max = subset['Efficiency'].max()
    ax.set_ylim(top=y_max * 30)  # headroom for bar labels

    # Title = just the domain name
    ax.set_title(domain, fontweight='bold', pad=8)

    # Energy labels on top of each bar (total energy consumed, NOT efficiency)
    for i, container in enumerate(ax.containers):
        stage_val = stages_order[i]
        stage_data = subset[subset['Stage'] == stage_val]
        energy_map = dict(zip(stage_data['Paper_Model'], stage_data['Total Energy (J)']))

        labels = []
        for model in models_order:
            e = energy_map.get(model, 0)
            if e >= 1_000_000: labels.append(f"{e/1_000_000:.1f}MJ")
            elif e >= 1000:    labels.append(f"{e/1000:.0f}kJ")
            else:              labels.append(f"{e:.0f}J")

        ax.bar_label(
            container, labels=labels,
            label_type='edge',
            padding=3,
            rotation=90,
            color='black',
            fontsize=8,
            weight='normal',
        )

# Single shared legend at the bottom
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=palette[s], edgecolor='black', linewidth=0.7, label=f'{s}')
    for s in stages_order
]
fig.legend(
    handles=handles,
    title='Samples',
    loc='lower center',
    ncol=3,
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    bbox_to_anchor=(0.5, -0.01),
)

plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=2.5, w_pad=1.5)

out_base = f"{output_dir}/efficiency_all_domains"
plt.savefig(f"{out_base}.pdf", format='pdf', bbox_inches='tight')
plt.savefig(f"{out_base}.png", bbox_inches='tight')
plt.close()

print(f"Saved: {out_base}.pdf / .png")
