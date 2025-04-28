import os
import argparse
import torch
import numpy as np
import open3d as o3d
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# --- Main Function ---
def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # Load point cloud from PCD file
    pcd_path = os.path.join(data_dir, '/root/anygrasp_sdk/grasp_detection/scene.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32) if pcd.has_colors() else np.zeros_like(points)

    # Workspace filter
    mask = (points[:, 2] > 0) & (points[:, 2] < 1.0)
    points = points[mask]
    colors = colors[mask]
    print("Point bounds:", points.min(axis=0), points.max(axis=0))
    print("Points after filtering:", points.shape[0])

    # Define workspace limits (camera frame)
    lims = [-1.5, 2.0, -2.0, 1.8, -2.0, 2.0]

    # Run AnyGrasp
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims,
                                   apply_object_mask=True,
                                   dense_grasp=False,
                                   collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return

    # Grasp processing
    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:100]
    print("Top grasp scores:\n", gg_pick.scores)
    print("Best grasp score:", gg_pick[0].score)
    print("\n--- Grasp Poses (6-DoF) ---")

    # --- Save Best Grasp to TXT ---
    best_grasp = gg_pick[12]
    position = best_grasp.translation
    rotation_matrix = best_grasp.rotation_matrix
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    # Combine position and quaternion into one array
    pose_data = np.hstack((position, quaternion))

    # Save to file
    output_txt_path = os.path.join(data_dir, 'best_grasp.txt')
    np.savetxt(output_txt_path, pose_data.reshape(1, -1), fmt='%.8f')
    print(f"Best grasp saved to {output_txt_path}")

    # --- Save all grasps to a text file ---
    all_grasps = []
    all_scores = []

    for i, grasp in enumerate(gg_pick):  # gg_pick is a GraspGroup
        position = grasp.translation
        rotation_matrix = grasp.rotation_matrix
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Save pose in (x, y, z, qx, qy, qz, qw) format
        pose_line = np.hstack((position, quaternion))
        all_grasps.append(pose_line)
        all_scores.append(grasp.score)

        print(f"Grasp {i}:")
        print(f"  Position    : {position}")
        print(f"  Quaternion  : {quaternion}")
        print("-" * 40)

    output_all_grasps_path = os.path.join(data_dir, 'all_grasps.txt')
    np.savetxt(output_all_grasps_path, np.array(all_grasps), fmt='%.8f')
    print(f"Saved all grasp poses to {output_all_grasps_path}")

    # --- New: Save all scores ---
    output_scores_path = os.path.join(data_dir, 'all_scores.txt')
    np.savetxt(output_scores_path, np.array(all_scores), fmt='%.8f')
    print(f"Saved all grasp scores to {output_scores_path}")

    # --- Visualization ---
    grippers = gg.to_open3d_geometry_list()
    trans_mat = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])

    # cloud.transform(trans_mat)
    # for g in grippers:
    #     g.transform(trans_mat)

    output_pcd_path = os.path.join(data_dir, 'processed_scene.pcd')
    o3d.io.write_point_cloud(output_pcd_path, cloud)
    print(f"Processed point cloud saved to {output_pcd_path}")

    # --- Sample points from grippers and merge with cloud ---

    # Sample points from each gripper mesh
    gripper_points = []
    for g in grippers:
        sampled = g.sample_points_uniformly(number_of_points=300)
        gripper_points.append(sampled)

    # Merge all gripper point clouds
    all_grippers_pcd = o3d.geometry.PointCloud()
    for gp in gripper_points:
        all_grippers_pcd += gp

    # Merge scene cloud + all gripper points
    combined_cloud = cloud + all_grippers_pcd

    # Save final point cloud
    output_pcd_path = os.path.join(data_dir, 'grasp_results.pcd')
    o3d.io.write_point_cloud(output_pcd_path, combined_cloud)
    print(f"Saved merged point cloud (scene + grippers) to {output_pcd_path}")


    if cfgs.debug:
        o3d.visualization.draw_geometries([cloud, *grippers], window_name="All Grasp Candidates")
        o3d.visualization.draw_geometries([cloud, grippers[0]], window_name="Top Grasp Only")



# --- Entry Point ---
if __name__ == '__main__':
    demo('/root/anygrasp_sdk/grasp_detection/example_data')

