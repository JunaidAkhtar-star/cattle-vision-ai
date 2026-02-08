import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(10,6))

# Hide axes
ax.axis("off")

# Define box properties
box_props = dict(boxstyle="round,pad=0.5", facecolor="#DDEEFF", edgecolor="black")

# Define steps
steps = [
    "FLW Upload Image",
    "Preprocessing\n(Resize & Normalize)",
    "AI Model (ResNet18)\nBreed Prediction",
    "Results Processing\n(Softmax + Confidence Threshold)",
    "Visualization\n(Charts & Scores)",
    "Data Stored in BPA Database"
]

# Draw boxes and arrows
y = 0.9
positions = []
for step in steps:
    ax.text(0.5, y, step, ha="center", va="center", fontsize=12,
            bbox=box_props)
    positions.append(y)
    y -= 0.15

# Draw arrows
for i in range(len(steps)-1):
    ax.annotate("",
                xy=(0.5, positions[i]-0.07),
                xytext=(0.5, positions[i+1]+0.07),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

# Save flowchart
flowchart_path = "/mnt/data/AI_Breed_Identification_Flowchart.png"
plt.savefig(flowchart_path, bbox_inches="tight")
plt.close()

flowchart_path
