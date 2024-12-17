import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import platform

# Initialize global variables for the transformation
transformation_init = False
t_f = None
R_f = None

def load_poses(file_path):
    """
    Load poses from a text file. Each line in the file should represent a pose as a 3x4 matrix.
    """
    translations = []
    rotations = []

    with open(file_path, 'r') as file:
        for line in file:
            # line = tartan2kittiSingle(line)
            elements = list(map(float, line.split()))
            # Extract translation and rotation
            translations.append([elements[3], elements[7], elements[11]])
            rotations.append([
                [elements[0], elements[1], elements[2]],
                [elements[4], elements[5], elements[6]],
                [elements[8], elements[9], elements[10]]
            ])

    translations = np.array(translations)
    rotations = np.array(rotations)
    return translations, rotations

def load_poses_tartan(poses):
    """
    Load poses from a text file. Each line in the file should represent a pose as a 3x4 matrix.
    """
    translations = []
    rotations = []

    
    for line in poses:
        # print(str(line))
        # print(type(line))
        line = ' '.join(f'{x:.8e}' for x in line)
        elements = list(map(float, line.split()))
        # Extract translation and rotation
        translations.append([elements[3], elements[7], elements[11]])
        rotations.append([
            [elements[0], elements[1], elements[2]],
            [elements[4], elements[5], elements[6]],
            [elements[8], elements[9], elements[10]]
        ])

    translations = np.array(translations)
    rotations = np.array(rotations)
    return translations, rotations

def calc_offset(t, R, t_f, R_f, ground_truth_t, ground_truth_R):
    """
    Calculate translation and rotation errors between the estimated and ground truth poses.
    """
    # Compute translation difference
    translation_difference = np.array(t) - np.array(t_f)
    translation_error = np.linalg.norm(translation_difference)

    # Compute rotation difference
    rotation_difference = np.dot(R, R_f.T)
    rotation_error_axis_angle = np.arccos((np.trace(rotation_difference) - 1) / 2.0)
    rotation_error_degrees = np.degrees(rotation_error_axis_angle)

    print(f"Translation Error: {translation_error} units")
    print(f"Rotation Error: {rotation_error_degrees} degrees")

    return translation_error, rotation_error_degrees

def getAbsoluteScale(frame_id, sequence_id):
    # import time
    # tgas = time.time()
    i = 0
    if platform.system() == "Linux":
        f = open("/mnt/d/loop_closure_detection/datasets/Kitti/data_odometry_poses/dataset/poses/02.txt", "r")
    else:
        f = open("D:/loop_closure_detection/datasets/Kitti/data_odometry_poses/dataset/poses/02.txt", "r")
    # print(f.read())

    x = 0
    y = 0
    z = 0
    x_prev = 0
    y_prev = 0
    z_prev = 0

    for line in f:
        if i <= frame_id:
            z_prev = z
            y_prev = y
            x_prev = x
            term = line.split()
            # print("term len: {}".format(len(term)))
            # print("term: {}".format(term))
            for j in range(len(term)):
                # print("terM: {}".format(term[j]))
                z = float(term[j])
                # print("z={}".format(z))
                if j == 7:
                    y = z
                if j == 3:
                    x = z

            i += 1
    f.close()
    del line, f, j,term, i
    # print(f"Get Absolute Scale time: {time.time() - tgas}")
    return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))



def update_line(t, R, ground_truth_t, ground_truth_R, file_num):
    """
    Update the current pose and compute the offset errors.
    """
    global t_f, R_f, transformation_init

    if not transformation_init:
        R_f = R
        t_f = t
        transformation_init = True
    else:
        # scale = 1  # Assuming a scale factor of 1
        scale = getAbsoluteScale(file_num, 0)
        t_f = t_f + scale * R_f.dot(t)
        R_f = R.dot(R_f)

    return calc_offset(t, R, t_f, R_f, ground_truth_t, ground_truth_R)


def tartan2kitti(traj):
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pos_quat2SE(pose).reshape(3,4)
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(ttt[:3,:].reshape(12))
        
    return np.array(new_traj)

def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE

def tartan2kittiSingle(traj):
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    

    
    tt = np.eye(4)
    tt[:3,:] = pos_quat2SE(traj).reshape(3,4)
    ttt=T.dot(tt).dot(T_inv)
        
    return ttt[:3,:].reshape(12)


def main():
    # Paths to ground truth and estimated pose files
    ground_truth_file = 'D:/loop_closure_detection/datasets/Kitti/data_odometry_poses/dataset/poses/02.txt'
    estimated_file = 'D:/loop_closure_detection/tartanvo/results/kitti_tartanvo_1914.txt'

    # Load poses
    ground_truth_t, ground_truth_R = load_poses(ground_truth_file)

    poselist = np.loadtxt(estimated_file).astype(np.float32)
    if poselist.shape[1]==7:
        poselist = tartan2kitti(poselist).astype(np.float32)
    estimated_t, estimated_R = load_poses_tartan(poselist)

    # DataFrame to store errors
    errors = pd.DataFrame(columns=['Translation_offset', 'Rotation_offset'])

    # Iterate through poses and compute errors
    for i in range(min(len(ground_truth_t), len(estimated_t))):
        t = estimated_t[i]
        R = estimated_R[i]
        gt_t = ground_truth_t[i]
        gt_R = ground_truth_R[i]

        translation_error, rotation_error = update_line(t, R, gt_t, gt_R, i)
        errors = pd.concat([errors, pd.DataFrame({'Translation_offset': [translation_error], 'Rotation_offset': [rotation_error]})], ignore_index=True)

    # Save errors to a CSV file
    errors.to_csv('pose_errors.csv', index=False)
    print("Errors saved to pose_errors.csv")

if __name__ == '__main__':
    main()
