import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np



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
    'pos-control': [-281.1892529296875],
    'neg-control': [-406.52847702026367],
    'qa-experimental': [-324.8078739929199]
}

# Set up the plot with larger figure size for better visibility of larger markers
plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor("whitesmoke")

# Plot each category with a larger marker using plot()
plt.plot(iterations, log_probabilities['pos-control'], label='pos-control', 
         marker='o', markersize=marker_size, color=color_map['pos-control'], linestyle='')
plt.plot(iterations, log_probabilities['neg-control'], label='neg-control', 
         marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='')
plt.plot(iterations, log_probabilities['qa-experimental'], label='qa-experimental', 
         marker='^', markersize=marker_size, color=color_map['qa-experimental'], linestyle='')

# Set the x-axis to show the range up to iteration 5
plt.xticks(range(4))  # This will show ticks from 0 to 5
plt.xlim(-0.5, 3.5)  # Set the limit so the x-axis will start a bit before 0 and end a bit after 5

# Set the labels for x and y axes
plt.xlabel('Iterations')
plt.ylabel('Log-Probability')

# Add the legend to the plot
plt.legend(borderpad=1, fontsize='large')

# Show grid with light gray color
plt.grid(True, color='white', linestyle='-', linewidth=0.9)
plt.title('Average Log-Probability of Desired Responses')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Show the plot
plt.savefig('assistant-gate-t0.png')
