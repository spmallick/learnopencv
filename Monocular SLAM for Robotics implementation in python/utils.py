import numpy as np

def read_calibration_file(calib_file_path):
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
    return lines

def extract_intrinsic_matrix(calib_lines, camera_id='P0'):
    # Find the line corresponding to the camera_id
    for line in calib_lines:
        if line.startswith(camera_id):
            # Split the line and convert to float
            values = line.strip().split()[1:]
            values = [float(val) for val in values]
            # Reshape to 3x4 matrix (the last column is for extrinsics, we need the 3x3 part)
            P = np.array(values).reshape(3, 4)
            K = P[:3, :3]
            return K
    return None

def main():
    calib_file_path = "../data/data_odometry_gray/dataset/sequences/00/calib.txt"
    calib_lines = read_calibration_file(calib_file_path)
    intrinsic_matrix = extract_intrinsic_matrix(calib_lines, camera_id='P0')
    
    if intrinsic_matrix is not None:
        print("Intrinsic Matrix (K):")
        print(intrinsic_matrix)
    else:
        print("Intrinsic matrix not found for the specified camera ID.")

if __name__ == "__main__":
    main()
