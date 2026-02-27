import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tigramite imports
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import json

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs/XGBoost-v2'

# Load the v2 windowed processed data (per-participant z-scored by train_model.py)
df = pd.read_csv(f'{OUTPUT_DIR}/processed_features_windowed.csv')

# Handle NaNs and Infs (same as XGBoost pipeline)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(df.median(numeric_only=True))

# Encode fatigue as numeric for ParCorr
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['fatigue_level_num'] = df['fatigue_level'].map(label_mapping)

# ─── Select variables for the causal graph ────────────────────────────────────
# Expanded v2 set: use the key sensor aggregates + engineered interaction features
# Keeping it interpretable (11 variables) while covering all sensor modalities.
selected_vars = [
    'wrist_eda_eda_mean',           # Electrodermal activity (stress)
    'wrist_hr_hr_mean',             # Heart rate (arousal)
    'wrist_temp_temp_mean',         # Skin temperature (thermoregulation)
    'muse_eeg_alpha_TP9_mean',      # Alpha EEG (relaxation)
    'muse_eeg_beta_TP9_mean',       # Beta EEG (alertness)
    'muse_eeg_theta_TP9_mean',      # Theta EEG (drowsiness)
    'muse_eeg_delta_TP9_mean',      # Delta EEG (deep fatigue)
    'muse_eeg_gamma_TP9_mean',      # Gamma EEG (cognitive processing)
    'alpha_beta_ratio_TP9',         # Drowsiness index
    'theta_beta_ratio_TP9',         # Sustained attention marker
    'eda_hr_interaction',           # Sympathetic arousal interaction
    'fatigue_level_num'
]

var_names = [
    'EDA', 'HeartRate', 'SkinTemp',
    'Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9', 'Gamma_TP9',
    'Alpha/Beta', 'Theta/Beta', 'EDA×HR',
    'Fatigue'
]

data_subset = df[selected_vars].values

print(f"Data shape for Causal Discovery: {data_subset.shape}")
print(f"Variables ({len(var_names)}): {var_names}")

# 2. Format data for Tigramite
# PCMCI expects time-series data. 
# Since our data is a sequence of sessions (01, 02, 03) per participant, we treat the sequence index as time.
dataframe = pp.DataFrame(data_subset, 
                         datatime=np.arange(len(data_subset)), 
                         var_names=var_names)

# 3. Setup the Conditional Independence Test
# ParCorr (Linear) — fast and reliable for this dimension/tau size
parcorr = ParCorr(significance='analytic')

# 4. Initialize and run PCMCI
# tau_max=2 to test 2-minute-back pathway propagation
pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1
)

print("\nRunning PCMCI Causal Discovery (v2 — 11 variables, tau_max=2)...")
results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)

print("\nPCMCI Results:")
print("p-values:")
print(np.round(results['p_matrix'], 3))
print("val_matrix (test statistic/correlation):")
print(np.round(results['val_matrix'], 3))

# 5. Extract and save the mathematical causal weights for the Streamlit Rules Engine
causal_weights = {}
val_matrix = results['val_matrix']
graph = results['graph']

for i in range(len(var_names)):
    for j in range(len(var_names)):
        for tau in range(0, 3):  # up to tau_max=2
            # Only store significant links
            if graph[i, j, tau] != "" and graph[i, j, tau] != "o-o":
                source = f"{var_names[i]}_t-{tau}"
                target = f"{var_names[j]}_t"
                weight = float(val_matrix[i, j, tau])
                causal_weights[f"{source}->{target}"] = weight

with open(f'{OUTPUT_DIR}/causal_weights.json', 'w') as f:
    json.dump(causal_weights, f, indent=4)
print(f"\nExported mathematical causal weights to {OUTPUT_DIR}/causal_weights.json")
print(f"Total significant links found: {len(causal_weights)}")

# 6. Generate clean 3-panel causal graph visualization
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Separate auto-regressive (self) links from cross-links
auto_links = {}
cross_links = {}
for k, v in causal_weights.items():
    parts = k.split('->')
    source_name = parts[0].rsplit('_t-', 1)[0]
    target_name = parts[1].rsplit('_t', 1)[0]
    if source_name == target_name:
        auto_links[k] = v
    else:
        cross_links[k] = v

# Color palette
COLORS = {
    'EDA': '#FF6B6B', 'HeartRate': '#EE5A24', 'SkinTemp': '#F9CA24',
    'Alpha_TP9': '#6C5CE7', 'Beta_TP9': '#A29BFE', 'Theta_TP9': '#00B894',
    'Delta_TP9': '#00CEC9', 'Gamma_TP9': '#FDCB6E',
    'Alpha/Beta': '#E17055', 'Theta/Beta': '#00B894', 'EDA×HR': '#D63031',
    'Fatigue': '#2D3436',
}
CATEGORIES = {
    'Wearable Sensors': ['EDA', 'HeartRate', 'SkinTemp'],
    'EEG Bands': ['Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9', 'Gamma_TP9'],
    'Engineered': ['Alpha/Beta', 'Theta/Beta', 'EDA×HR'],
    'Target': ['Fatigue'],
}
all_vars = list(COLORS.keys())

fig = plt.figure(figsize=(20, 8), facecolor='#0E1117')
fig.suptitle('PCMCI Causal Discovery — V2 Pipeline (11 Variables, τ_max=2)',
             color='white', fontsize=16, fontweight='bold', y=0.98)

# ── Panel 1: Filtered Cross-Link Graph ───────────────────────────────────
ax1 = fig.add_subplot(131)
ax1.set_facecolor('#0E1117')
ax1.set_title('Cross-Causal Links\n(|strength| ≥ 0.08, self-loops removed)',
              color='white', fontsize=11, fontweight='bold', pad=12)

G = nx.DiGraph()
for var in all_vars:
    G.add_node(var)

threshold = 0.08
for k, v in cross_links.items():
    if abs(v) >= threshold:
        parts = k.split('->')
        src = parts[0].rsplit('_t-', 1)[0]
        tgt = parts[1].rsplit('_t', 1)[0]
        tau = int(parts[0].rsplit('_t-', 1)[1])
        if G.has_edge(src, tgt):
            if abs(v) > abs(G[src][tgt]['weight']):
                G[src][tgt]['weight'] = v
                G[src][tgt]['tau'] = tau
        else:
            G.add_edge(src, tgt, weight=v, tau=tau)

pos = nx.shell_layout(G, nlist=[
    ['Fatigue'],
    ['EDA', 'HeartRate', 'SkinTemp', 'EDA×HR'],
    ['Alpha_TP9', 'Beta_TP9', 'Theta_TP9', 'Delta_TP9', 'Gamma_TP9', 'Alpha/Beta', 'Theta/Beta'],
])

nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=600, 
                       node_color=[COLORS.get(n, '#888') for n in G.nodes()],
                       edgecolors='white', linewidths=1.5, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=7, font_color='white', font_weight='bold')

edges = G.edges(data=True)
if edges:
    max_w = max(abs(d['weight']) for _, _, d in edges) or 1
    for u, v, d in edges:
        w = abs(d['weight'])
        color = '#FF4757' if d['weight'] > 0 else '#3742FA'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax1,
                               width=1 + 4 * (w / max_w), edge_color=color,
                               alpha=0.4 + 0.5 * (w / max_w),
                               arrows=True, arrowsize=12,
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=15, min_target_margin=15)
ax1.axis('off')
ax1.legend(handles=[mpatches.Patch(color='#FF4757', label='Positive'),
                    mpatches.Patch(color='#3742FA', label='Negative')],
           loc='lower center', fontsize=8, facecolor='#1E1E2E',
           edgecolor='#444', labelcolor='white', framealpha=0.9)

# ── Panel 2: Heatmap ────────────────────────────────────────────────────
ax2 = fig.add_subplot(132)
ax2.set_facecolor('#0E1117')
ax2.set_title('Causal Strength Heatmap\n(max |weight| across all lags)',
              color='white', fontsize=11, fontweight='bold', pad=12)

matrix = np.zeros((len(all_vars), len(all_vars)))
for k, v in causal_weights.items():
    parts = k.split('->')
    src = parts[0].rsplit('_t-', 1)[0]
    tgt = parts[1].rsplit('_t', 1)[0]
    i = all_vars.index(src) if src in all_vars else -1
    j = all_vars.index(tgt) if tgt in all_vars else -1
    if i >= 0 and j >= 0 and abs(v) > abs(matrix[i, j]):
        matrix[i, j] = v

short_labels = ['EDA', 'HR', 'Temp', 'α', 'β', 'θ', 'δ', 'γ', 'α/β', 'θ/β', 'EDA×HR', 'Fatigue']
im = ax2.imshow(matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
ax2.set_xticks(range(len(short_labels)))
ax2.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8, color='white')
ax2.set_yticks(range(len(short_labels)))
ax2.set_yticklabels(short_labels, fontsize=8, color='white')
ax2.set_xlabel('Target →', color='#aaa', fontsize=9)
ax2.set_ylabel('← Source', color='#aaa', fontsize=9)
ax2.tick_params(colors='white')

for i in range(len(all_vars)):
    for j in range(len(all_vars)):
        val = matrix[i, j]
        if abs(val) >= 0.08:
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=6, color='white' if abs(val) > 0.15 else '#ccc',
                     fontweight='bold' if abs(val) > 0.15 else 'normal')

cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors='white', labelsize=7)
cbar.set_label('Causal Strength', color='white', fontsize=8)

# ── Panel 3: Fatigue-Centric Star ───────────────────────────────────────
ax3 = fig.add_subplot(133)
ax3.set_facecolor('#0E1117')
ax3.set_title('Fatigue-Centric View\n(what drives Fatigue & what Fatigue drives)',
              color='white', fontsize=11, fontweight='bold', pad=12)

fatigue_in, fatigue_out = {}, {}
for k, v in causal_weights.items():
    parts = k.split('->')
    src = parts[0].rsplit('_t-', 1)[0]
    tgt = parts[1].rsplit('_t', 1)[0]
    tau = int(parts[0].rsplit('_t-', 1)[1])
    if tgt == 'Fatigue' and src != 'Fatigue':
        if src not in fatigue_in or abs(v) > abs(fatigue_in[src][0]):
            fatigue_in[src] = (v, tau)
    elif src == 'Fatigue' and tgt != 'Fatigue':
        if tgt not in fatigue_out or abs(v) > abs(fatigue_out[tgt][0]):
            fatigue_out[tgt] = (v, tau)

G2 = nx.DiGraph()
G2.add_node('Fatigue')
pos2 = {'Fatigue': (0, 0)}

for i, (name, (w, tau)) in enumerate(sorted(fatigue_in.items(), key=lambda x: abs(x[1][0]), reverse=True)):
    angle = np.pi/2 + np.pi * (i + 0.5) / max(len(fatigue_in), 1)
    pos2[name] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
    G2.add_node(name)
    G2.add_edge(name, 'Fatigue', weight=w, tau=tau)

for i, (name, (w, tau)) in enumerate(sorted(fatigue_out.items(), key=lambda x: abs(x[1][0]), reverse=True)):
    angle = -np.pi/2 + np.pi * (i + 0.5) / max(len(fatigue_out), 1)
    if name not in pos2:
        pos2[name] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
    G2.add_node(name)
    G2.add_edge('Fatigue', name, weight=w, tau=tau)

nx.draw_networkx_nodes(G2, pos2, ax=ax3,
                       node_size=[900 if n == 'Fatigue' else 500 for n in G2.nodes()],
                       node_color=[COLORS.get(n, '#888') for n in G2.nodes()],
                       edgecolors='white', linewidths=1.5, alpha=0.9)
nx.draw_networkx_labels(G2, pos2, ax=ax3, font_size=7, font_color='white', font_weight='bold')

for u, v, d in G2.edges(data=True):
    color = '#FF4757' if d['weight'] > 0 else '#3742FA'
    width = 1 + 3 * abs(d['weight']) / 0.3
    nx.draw_networkx_edges(G2, pos2, edgelist=[(u, v)], ax=ax3,
                           width=width, edge_color=color,
                           alpha=0.5 + 0.5 * min(abs(d['weight']) / 0.15, 1.0),
                           arrows=True, arrowsize=15,
                           connectionstyle='arc3,rad=0.1',
                           min_source_margin=18, min_target_margin=18)
    mid_x = (pos2[u][0] + pos2[v][0]) / 2
    mid_y = (pos2[u][1] + pos2[v][1]) / 2
    ax3.text(mid_x, mid_y, f'{d["weight"]:+.2f}', fontsize=6, color='#aaa',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='#1E1E2E', edgecolor='none', alpha=0.8))
ax3.axis('off')

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[v[0]],
                          markersize=8, label=cat, linewidth=0) for cat, v in CATEGORIES.items()]
ax3.legend(handles=legend_elements, loc='lower center', fontsize=7,
           facecolor='#1E1E2E', edgecolor='#444', labelcolor='white', framealpha=0.9, ncol=2)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig(f'{OUTPUT_DIR}/causal_graph.png', dpi=150, bbox_inches='tight',
            facecolor='#0E1117', edgecolor='none')
print(f"\nSaved causal graph to {OUTPUT_DIR}/causal_graph.png")
plt.close()

