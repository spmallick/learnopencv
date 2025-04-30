

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# Keep best_fit_transform and nearest_neighbor functions as they were in your original code
def best_fit_transform(A, B):
    """
    Calculates the best-fit transform that maps points A onto points B.
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
    Output:
        T: (m+1)x(m+1) homogeneous transformation matrix
    """
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()
    return T

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # Ensure shapes are compatible for KNN, although they don't strictly need to be identical N
    assert src.shape[1] == dst.shape[1] 
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


# --- Modified ICP function to store history ---
def iterative_closest_point_visual(A, B, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B.
    Stores intermediate results for visualization.
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T_final: final homogeneous transformation that maps A on to B
        intermediate_A: List containing A transformed at each iteration (N x m arrays)
        intermediate_errors: List containing the mean error at each iteration
        i: number of iterations to converge
    '''

    # Check dimensions
    # Allow N to differ, but dimensions (m) must match
    assert A.shape[1] == B.shape[1] 

    # get number of dimensions
    m = A.shape[1]
    
    # --- Store History ---
    intermediate_A = [np.copy(A)] # Store initial state
    intermediate_errors = []
    # --- Store History ---

    # make points homogeneous, copy them to maintain the originals
    # Use np.copy() for src to allow modification without affecting original A
    src_h = np.ones((m+1, A.shape[0])) 
    src_h[:m, :] = np.copy(A.T)
    
    # Target points (B) remain fixed, use non-homogeneous for KNN
    dst = np.copy(B) # Non-homogeneous target points for KNN

    prev_error = float('inf') # Initialize with infinity
    T_cumulative = np.identity(m+1) # To accumulate transformations correctly

    for i in range(max_iterations):
        # Current source points (non-homogeneous)
        current_src = src_h[:m, :].T 

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(current_src, dst)

        # compute the transformation between the current source and nearest destination points
        # Use the subset of B (dst) corresponding to the nearest neighbors found
        T_step = best_fit_transform(current_src, dst[indices, :])

        # update the current source points *in homogeneous coordinates*
        src_h = np.dot(T_step, src_h)
        
        # --- Store History ---
        intermediate_A.append(src_h[:m, :].T) # Store transformed A for this iteration
        # --- Store History ---

        # check error (stop if error is less than specified tolerance)
        mean_error = np.mean(distances)
        intermediate_errors.append(mean_error) # Store error for this iteration
        
        # Use absolute difference check for convergence
        if np.abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {i+1} with error difference {np.abs(prev_error - mean_error)}")
            break
            
        prev_error = mean_error
        
        # Accumulate transformation
        T_cumulative = np.dot(T_step, T_cumulative)


    # Calculate the *final* transformation from the *original* A to the *final* src position
    # This accounts for the accumulated transform
    T_final = best_fit_transform(A, src_h[:m, :].T)
    
    # If loop finished due to max_iterations without converging based on tolerance
    if i == max_iterations - 1:
         print(f"Reached max iterations ({max_iterations})")

    return T_final, intermediate_A, intermediate_errors, i + 1 # Return i+1 for actual count



"""Reference: https://github.com/OmarJItani/Iterative-Closest-Point-Algorithm/tree/main
"""



