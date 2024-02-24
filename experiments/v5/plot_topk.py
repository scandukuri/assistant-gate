import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import json
from utils import *

with open(M0_NEG_CONTROL_DIR, 'r') as f:
    neg_control = json.load(f)

with open(M0_POS_CONTROL_DIR, 'r') as f:
    pos_control = json.load(f)

with open(M0_TOPK_POS_CONTROL_DIR, 'r') as f:
    pos_control_topk = json.load(f)
    
with open(M0_EXPERIMENTAL_DIR, 'r') as f:
    experimental = json.load(f)

with open(M0_TOPK_EXPERIMENTAL_DIR, 'r') as f:
    experimental_topk = json.load(f)


with open(M1_NEG_CONTROL_DIR, 'r') as f:
    m1_neg_control = json.load(f)

with open(M1_POS_CONTROL_DIR, 'r') as f:
    m1_pos_control = json.load(f)

with open(M1_TOPK_POS_CONTROL_DIR, 'r') as f:
    m1_pos_control_topk = json.load(f)
    
with open(M1_EXPERIMENTAL_DIR, 'r') as f:
    m1_experimental = json.load(f)

with open(M1_TOPK_EXPERIMENTAL_DIR, 'r') as f:
    m1_experimental_topk = json.load(f)

log_probabilities = {}

# Now calculate means and SEMs instead of just means
# Calculate means and SEMs

neg_control_mean, neg_control_sem = calculate_mean_and_sem([lst for key, lst in neg_control.items()])
print(f"Negative Control Mean: {neg_control_mean}, SEM: {neg_control_sem}")


pos_control_topk_mean, pos_control_topk_sem = calculate_mean_and_sem([lst for key, lst in pos_control_topk.items()])
print(f"Positive Control Top-K Mean: {pos_control_topk_mean}, SEM: {pos_control_topk_sem}")

experimental_topk_mean, experimental_topk_sem = calculate_mean_and_sem([lst for key, lst in experimental_topk.items()])
print(f"Experimental Top-K Mean: {experimental_topk_mean}, SEM: {experimental_topk_sem}")


m1_neg_control_mean, m1_neg_control_sem = calculate_mean_and_sem([lst for key, lst in m1_neg_control.items()])
print(f"M1 Negative Control Mean: {m1_neg_control_mean}, SEM: {m1_neg_control_sem}")


m1_pos_control_topk_mean, m1_pos_control_topk_sem = calculate_mean_and_sem([lst for key, lst in m1_pos_control_topk.items()])
print(f"M1 Positive Control Top-K Mean: {m1_pos_control_topk_mean}, SEM: {m1_pos_control_topk_sem}")

m1_experimental_topk_mean, m1_experimental_topk_sem = calculate_mean_and_sem([lst for key, lst in m1_experimental_topk.items()])
print(f"M1 Experimental Top-K Mean: {m1_experimental_topk_mean}, SEM: {m1_experimental_topk_sem}")
# The rest of your plotting code remains the same

# Adjusting the provided code to use plot() instead of scatter(), and making the background grid light gray
iterations=[0, 1]
# Adjust marker sizes for better visibility with plot()
marker_size = 16  # Adjust this value as needed for larger markers with plot()

# Define custom colors to match the provided image as closely as possible
color_map = {
    'pos-control': '#FFA500',  # yellow color for positive control
    'neg-control': '#1F77B4',  # Muted blue color for negative control
    'qa-experimental': '#FF7F0E' # Safety orange for QA experimental
}


# Log-probability data points for each category at iteration 0
log_probabilities = {
    'neg-control': [neg_control_mean, m1_neg_control_mean],
    f'pos-control-top-{k}' : [pos_control_topk_mean, m1_pos_control_topk_mean],
    f'qa-experimental-top-{k}' : [experimental_topk_mean, m1_experimental_topk_mean]
}

# Set up the plot with larger figure size for better visibility of larger markers
plt.figure(figsize=(10, 9))
ax = plt.axes()
ax.set_facecolor("whitesmoke")

# Plot each category with a larger marker using plot()
plt.plot(iterations, log_probabilities['neg-control'], label='neg-control', 
         marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='', zorder=z_order/100)
plt.plot(iterations, log_probabilities[f'qa-experimental-top-{k}'], label=f'qa-experimental-top-{k}', 
         marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='', alpha=0.4, zorder=z_order/100)
plt.plot(iterations, log_probabilities[f'pos-control-top-{k}'], label=f'pos-control-top-{k}', 
         marker='^', markersize=marker_size, color=color_map['pos-control'], linestyle='', alpha=0.4, zorder=z_order/100)

# # Adjust the plotting code to include error bars
# # Plot each category with error bars
plt.errorbar(iterations, [neg_control_mean, m1_neg_control_mean], yerr=[neg_control_sem, m1_neg_control_sem], marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='', capsize=5,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)

# If the top-k categories also need error bars, adjust similarly
# For example:
plt.errorbar(iterations, [pos_control_topk_mean, m1_pos_control_topk_mean], yerr=[pos_control_topk_sem, m1_pos_control_topk_sem], marker='^', markersize=marker_size, color=color_map['pos-control'], linestyle='', capsize=5,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
plt.errorbar(iterations, [experimental_topk_mean, m1_experimental_topk_mean], yerr=[experimental_topk_sem, m1_experimental_topk_sem], marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='', capsize=5,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)



# Set the x-axis to show the range up to iteration 5
plt.xticks(range(4))  # This will show ticks from 0 to 5
plt.xlim(-0.5, 3.5)  # Set the limit so the x-axis will start a bit before 0 and end a bit after 5

# Set x-axis range
ax.set_ylim([-450, -250])

# Set the labels for x and y axes
plt.xlabel('Iterations')
plt.ylabel('Log-Probability')

# Add the legend to the plot
plt.legend(borderpad=1, fontsize='large')

# Show grid with light gray color
plt.grid(True, color='white', linestyle='-', linewidth=0.9)
plt.title(f'Average Log-Probability of Desired Responses [n = {sum([len(lst) for key, lst in pos_control_topk.items()])} simulations]')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Show the plot
plt.savefig(f'figures/plot_top-{k}.png')
