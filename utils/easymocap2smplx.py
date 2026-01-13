import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_alignment(Th, Rh_vec, pelvis_local, corrected_trans):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert Rh to matrix for rotation
    rmat = R.from_rotvec(Rh_vec).as_matrix()

    # 1. Calculate World Pelvis via EasyMocap Method
    # EasyMocap rotates the local pelvis then translates it by Th
    world_pelvis_em = Th + np.dot(rmat, pelvis_local)

    # 2. Canonical SMPLX Pelvis
    # In SMPLX, the transl parameter IS the world position of the pelvis
    world_pelvis_sx = corrected_trans

    # Plotting
    ax.scatter(world_pelvis_em[0], world_pelvis_em[1], world_pelvis_em[2], 
               c='r', s=100, label='EasyMocap Calculated Pelvis', marker='o')
    ax.scatter(world_pelvis_sx[0], world_pelvis_sx[1], world_pelvis_sx[2], 
               c='b', s=50, label='SMPLX "transl" Position', marker='x')

    # Draw a line between origin and pelvis to show vector
    ax.plot([Th[0], world_pelvis_em[0]], [Th[1], world_pelvis_em[1]], [Th[2], world_pelvis_em[2]], 
            'g--', label='Offset Vector (R @ Jp)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Coordinate Alignment Verification')
    ax.legend()
    plt.savefig('alignment.png')

def get_full_hand_pose(pca_coeffs, model_path, side='left'):
    data = np.load(model_path, allow_pickle=True)
    
    # 1. Get the components (6, 45)
    comp_key = 'hands_componentsl' if side == 'left' else 'hands_componentsr'
    components = data[comp_key][:6] 
    
    # 2. Get the mean pose (45,)
    mean_key = 'hands_meanl' if side == 'left' else 'hands_meanr'
    mean_pose = data[mean_key] 
    
    # 3. Weighted Sum (The Dot Product)
    # pca_coeffs shape (6,) dot components shape (6, 45) -> (45,)
    full_pose = np.dot(pca_coeffs, components) + mean_pose
    return full_pose


def get_pelvis_offset(betas, model_path): 
    """
    Calculates the local coordinates of the pelvis joint based on body shape (betas).
    """
    data = np.load(model_path, allow_pickle=True)
    
    # v_template: (10475, 3), shapedirs: (10475, 3, 10), J_regressor: (55, 10475)
    v_template = data['v_template']
    shapedirs = data['shapedirs'][:, :, :10] # Only take first 10 betas
    j_regressor = data['J_regressor']
    
    # 1. Calculate shaped vertices: V_beta = V_template + sum(beta_i * shapedir_i)
    v_shaped = v_template + np.einsum('vcd,d->vc', shapedirs, betas)
    
    # 2. Get joints: J = J_regressor @ V_shaped
    # Pelvis is index 0 in the SMPL-X joint set
    joints = np.matmul(j_regressor, v_shaped)
    pelvis_local = joints[0] 
    
    return pelvis_local

def convert_easymocap_to_smplx(json_data, smplx_default_path, zero_hands=False):
    poses = np.array(json_data['poses']).reshape(-1)
    Rh_vec = np.array(json_data['Rh']).reshape(3)
    Th = np.array(json_data['Th']).reshape(3)
    expression = np.array(json_data['expression']).reshape(10)
    betas = np.array(json_data['shapes']).reshape(10)

    body_pose = poses[3:66]    
    left_hand_pca_pose = poses[66:72]
    right_hand_pca_pose = poses[72:78]
    jaw_pose = poses[78:81]
    left_eye_pose = poses[81:84]
    right_eye_pose = poses[84:87]

    if zero_hands:
        left_hand_pose = np.zeros(45)
        right_hand_pose = np.zeros(45)
    else:
        left_hand_pose = get_full_hand_pose(left_hand_pca_pose, smplx_default_path, 'left')
        right_hand_pose = get_full_hand_pose(right_hand_pca_pose, smplx_default_path, 'right')

    # --- ACCURATE PELVIS ESTIMATION (TH -> TRANS) ---
    # 1. Get the local pelvis position based on current body shape
    pelvis_local = get_pelvis_offset(betas, smplx_default_path)
    
    # 2. Convert Rh axis-angle to a 3x3 rotation matrix
    rmat = R.from_rotvec(Rh_vec).as_matrix()
    
    # 3. Apply the alignment formula:
    # EasyMocap: World = Th + R * Mesh
    # SMPLX: World = Trans + R * (Mesh - Pelvis) + Pelvis
    # Equating at Pelvis: Th + R * Pelvis = Trans + Pelvis
    # Trans = Th + R * Pelvis - Pelvis
    corrected_trans = Th + np.dot(rmat, pelvis_local) - pelvis_local

    smplx_params = {
        'global_orient': Rh_vec,
        'body_pose': body_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'left_eye_pose': left_eye_pose,
        'right_eye_pose': right_eye_pose,
        'betas': betas,
        'expression': expression,
        'transl': corrected_trans  # This is now accurately aligned for canonical SMPLX
    }

    return smplx_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--default-model", type=str, default="SMPLX_NEUTRAL.npz")
    parser.add_argument("--zero-hands", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    with open(args.json_path, "r") as f: # read json file containing EasyMocap data
        data = json.load(f)[0] # only a single frame
    smplx_params = convert_easymocap_to_smplx(data, args.default_model, args.zero_hands)
    print(data['Th'], data['Rh'])
    if args.visualize:
        visualize_alignment(np.array(data['Th']).flatten(), np.array(data['Rh']).flatten(), get_pelvis_offset(smplx_params['betas'], args.default_model), smplx_params['transl'])

# -------------------------------
# Example usage:
# -------------------------------
# import json
# with open('smpl/000000.json', 'r') as f:
#     data = json.load(f)
# smplx_params = convert_easymocap_to_smplx(data)
# print(smplx_params)
