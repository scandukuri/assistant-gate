import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import json
from utils import *


with open(POS_CONTROL_TOPK_DIR, 'r') as f:
    pos_control_topk = json.load(f)
    
with open(NEG_CONTROL_DIR, 'r') as f:
    neg_control = json.load(f)

with open(EXPERIMENTAL_TOPK_DIR, 'r') as f:
    experimental_topk = json.load(f)

log_probabilities = {}

neg_control_mean = sum([sum(lst) for key, lst in neg_control.items()]) / sum([len(lst) for key, lst in neg_control.items()])
pos_control_topk_mean = sum([sum(lst) for key, lst in pos_control_topk.items()]) / sum([len(lst) for key, lst in pos_control_topk.items()])
experimental_topk_mean = sum([sum(lst) for key, lst in experimental_topk.items()]) / sum([len(lst) for key, lst in experimental_topk.items()])


# Adjusting the provided code to use plot() instead of scatter(), and making the background grid light gray
iterations=[0]
# Adjust marker sizes for better visibility with plot()
marker_size = 16  # Adjust this value as needed for larger markers with plot()

# Define custom colors to match the provided image as closely as possible
color_map = {
    'pos-control': '#FFA500',  # Orange color for positive control
    'neg-control': '#1F77B4',  # Muted blue color for negative control
    'qa-experimental': '#FF7F0E' # Safety orange for QA experimental
}


# Log-probability data points for each category at iteration 0
log_probabilities = {
    'neg-control': [neg_control_mean],
    f'pos-control-top-{k}' : [pos_control_topk_mean],
    f'qa-experimental-top-{k}' : [experimental_topk_mean]
}

# Set up the plot with larger figure size for better visibility of larger markers
plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor("whitesmoke")

# Plot each category with a larger marker using plot()
plt.plot(iterations, log_probabilities['neg-control'], label='neg-control', 
         marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='')
plt.plot(iterations, log_probabilities[f'qa-experimental-top-{k}'], label=f'qa-experimental-top-{k}', 
         marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='', alpha=0.4)
plt.plot(iterations, log_probabilities[f'pos-control-top-{k}'], label=f'pos-control-top-{k}', 
         marker='^', markersize=marker_size, color=color_map['pos-control'], linestyle='', alpha=0.4)

# Set the x-axis to show the range up to iteration 5
plt.xticks(range(4))  # This will show ticks from 0 to 5
plt.xlim(-0.5, 3.5)  # Set the limit so the x-axis will start a bit before 0 and end a bit after 5

# Set x-axis range
ax.set_ylim([-450, -200])

# Set the labels for x and y axes
plt.xlabel('Iterations')
plt.ylabel('Log-Probability')

# Add the legend to the plot
plt.legend(borderpad=1, fontsize='large')

# Show grid with light gray color
plt.grid(True, color='white', linestyle='-', linewidth=0.9)
plt.title(f'Average Log-Probability of Desired Responses [n = {20000*k} simulations]')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Show the plot
plt.savefig(f'figures/plot_top-{k}.png')
