import numpy as np
import matplotlib.pyplot as plt
import icp
import matplotlib.animation as animation
import copy # Needed for deep copies if modifying arrays in place later



# Define two sets of points A and B (same as your example)
t = np.linspace(0, 2*np.pi, 10)
A = np.column_stack((t, np.sin(t)))

# Define the rotation angle
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array(((c, -s), (s, c)))

# Define translation vector
translation_vector = np.array([[2, 0]])

# Apply the transformation and add randomness to get B
np.random.seed(42) # for reproducible randomness
randomness = 0.3 * np.random.rand(10, 2)
B = np.dot(rotation_matrix, A.T).T + translation_vector + randomness

# --- Run ICP and get history ---
max_iter = 20
tolerance = 0.0001 # Lower tolerance for smoother convergence potentially
T_final, history_A, history_error, iters = icp.iterative_closest_point_visual(A, B, max_iterations=max_iter, tolerance=tolerance)

print(f'Converged/Stopped after {iters} iterations.')
print(f'Final Mean Error: {history_error[-1]:.4f}')
print('Final Transformation:')
print(np.round(T_final, 3))

# --- Create Animation ---
fig, ax = plt.subplots()

# Plot target points (static)
ax.scatter(B[:, 0], B[:, 1], color='blue', label='Target B', marker='x')

# Plot initial source points
scatter_A = ax.scatter(history_A[0][:, 0], history_A[0][:, 1], color='red', label='Source A (moving)')
title = ax.set_title(f'Iteration 0, Mean Error: N/A')
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.axis('equal') # Important for visualizing rotations correctly

# Determine plot limits based on all points across all iterations
all_points = np.vstack([B] + history_A)
min_vals = np.min(all_points, axis=0)
max_vals = np.max(all_points, axis=0)
range_vals = max_vals - min_vals
margin = 0.1 * range_vals # Add 10% margin
ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])


# Animation update function
def update(frame):
    # Update source points position
    scatter_A.set_offsets(history_A[frame])
    # Update title
    error_str = f"{history_error[frame-1]:.4f}" if frame > 0 else "N/A" # Error calculated *after* step
    title.set_text(f'Iteration {frame}, Mean Error: {error_str}')
    # Return the artists that were modified
    return scatter_A, title,

# Create the animation
# Number of frames is number of states stored (initial + iterations)
# Interval is milliseconds between frames (e.g., 500ms = 0.5s)
ani = animation.FuncAnimation(fig, update, frames=len(history_A), 
                              interval=500, blit=True, repeat=False)

# # --- Save the animation ---
# # You might need to install ffmpeg: conda install ffmpeg or sudo apt-get install ffmpeg
# try:
#     output_filename = 'icp_iterations.mp4'
#     ani.save(output_filename, writer='ffmpeg', fps=2) # fps = frames per second
#     print(f"Animation saved successfully as {output_filename}")
# except Exception as e:
#     print(f"Error saving animation: {e}")
#     print("Ensure ffmpeg is installed and accessible in your system's PATH.")
#     print("Alternatively, try saving as GIF (may require 'imagemagick' or 'pillow'):")
#     # try:
#     #     ani.save('icp_iterations.gif', writer='pillow', fps=2)
#     #     print("Animation saved successfully as icp_iterations.gif")
#     # except Exception as e_gif:
#     #     print(f"Error saving GIF: {e_gif}")


# Display the final plot (optional, animation already shows it)
plt.figure()
plt.scatter(history_A[-1][:, 0], history_A[-1][:, 1], color='red', label='Final A')
plt.scatter(B[:, 0], B[:, 1], color='blue', label='Target B', marker='x')
plt.legend()
plt.title(f"Final Alignment after {iters} iterations")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()





""""
Reference: https://github.com/OmarJItani/Iterative-Closest-Point-Algorithm/tree/main
"""
