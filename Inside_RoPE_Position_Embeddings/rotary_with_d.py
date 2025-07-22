import numpy as np
import matplotlib.pyplot as plt

# ---------- RoPE Configurations ----------

paragraph  = "hello my name is Shubham Anand and I am a Computer Vision Engineer I work for OpenCV University I am working simultaneously on multiple projects like with LeRobot and then writing short and long blogs also and then handling the Deep Learning in PyTorch course also so"
d          = 64          # hidden (or head) size, must be even
pair_idx   = 7           # which (cos,sin) pair to plot: 0 … d/2-1

# ----------------------------

tokens   = paragraph.split()
p        = np.arange(len(tokens))           # positions 0,1,2,…

# RoPE angle schedule ⟶ depends on both i and d
base     = 10000 ** (2*pair_idx / d)
angles   = p / base                         # radians

# Coordinates for that pair
x, y = np.cos(angles), np.sin(angles)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, marker="x")

# Dotted rays with token labels mid-way
for xi, yi, tok in zip(x, y, tokens):
    plt.plot([0, xi], [0, yi], ":", lw=1)
    plt.text(0.6*xi, 0.6*yi, tok, ha="center", va="center")

# Unit circle and grid
plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, ls="--"))
plt.grid(ls="--", lw=0.4, alpha=0.4)

plt.title(f"RoPE pair {pair_idx} of d={d} (dim {2*pair_idx}/{2*pair_idx+1})")
plt.xlabel("cos θ"); plt.ylabel("sin θ")
plt.axis("equal"); plt.show()


