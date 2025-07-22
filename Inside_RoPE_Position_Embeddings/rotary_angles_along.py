import numpy as np
import matplotlib.pyplot as plt

# ----------- EDIT YOUR PARAGRAPH -----------
paragraph = "hello my name is Shubham Anand"
# -------------------------------------------

d = 64                               # hidden size (even for RoPE)
tokens  = paragraph.split()
seq_len = len(tokens)

# ---- RoPE: first-frequency angles (1 rad / token) ----
angles = np.arange(seq_len) * 1.0    # radians (only for the first frequency or first pair or the pair with 0 pair_index)

# Coordinates on the unit circle
x = np.cos(angles)
y = np.sin(angles)

# ---- Plot ----
plt.figure(figsize=(6, 6))

# Embedding points (no labels here)
plt.scatter(x, y, marker='x', zorder=3)

# Unit circle for reference
plt.gca().add_artist(plt.Circle((0, 0), 1,
                                fill=False, linestyle="--", zorder=1))

# Light dashed grid
plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)

# Parameters controlling label placement
token_r  = 0.60   # radial position for token labels (60 % of radius)
angle_r  = 0.15   # radial position for angle labels near origin

for xi, yi, theta, tok in zip(x, y, angles, tokens):
    # Dotted ray from origin to embedding point
    plt.plot([0, xi], [0, yi], linestyle=":", linewidth=1, zorder=2)

    # Token label along the ray (not at the endpoint)
    plt.text(token_r * np.cos(theta),
             token_r * np.sin(theta),
             tok,
             ha="center", va="center", fontsize=9)

    # Angle label closer to the origin
    plt.text(angle_r * np.cos(theta),
             angle_r * np.sin(theta),
             f"{np.degrees(theta):.1f}°",
             ha="center", va="center", fontsize=7)

plt.title("RoPE – Token Labels Along Rays with Grid")
plt.xlabel("cos θ")
plt.ylabel("sin θ")
plt.axis("equal")         # equal aspect ratio
plt.show()
