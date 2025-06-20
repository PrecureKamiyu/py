import matplotlib.pyplot as plt
import numpy as np

ns = [10, 20, 30]

delays_mip = [
    26.046,
    35.065,
    50.203,
]
delays_proposed = [
    21.799,
    29.61,
    29.218,
]

# categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
# group1_values = [20, 35, 30, 40] # e.g., Sales for Product A
# group2_values = [25, 32, 34, 38] # e.g., Sales for Product B

# 2. Set up positions and bar width
x = np.arange(len(ns))  # the label locations (0, 1, 2, 3...)
width = 0.35  # the width of the bars

# Create the figure and an axes object
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size for better readability

# 3. Plot the bars for each group, shifting their x-positions
rects1 = ax.bar(x - width/2, delays_mip, width, label='cMIP-L1', color='skyblue')
rects2 = ax.bar(x + width/2, delays_proposed, width, label='Proposed Method', color='lightcoral')

# 4. Add labels, title, and legend
ax.set_xlabel('Number of placed servers')
ax.set_ylabel('Delay')
# ax.set_title('Comparison of Group 1 and Group 2 Data across Categories')
ax.set_xticks(x) # Set tick locations to be at the center of the grouped bars
ax.set_xticklabels(ns) # Use your category labels for the x-axis ticks
ax.legend()

# Optional: Add data labels on top of the bars for clarity
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

# Adjust layout to prevent labels from overlapping
fig.tight_layout()

plt.show()
