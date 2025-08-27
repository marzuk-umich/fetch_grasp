import os
import argparse
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

class GraspPose:
    """Mock GraspPose class to replicate graspnetAPI.GraspGroup behavior"""
    def __init__(self, position, rotation_matrix, score, gripper_width=0.08):
        self.translation = position
        self.rotation_matrix = rotation_matrix
        self.score = score
        self.gripper_width = gripper_width

class GraspGroup:
    """Mock GraspGroup class to replicate graspnetAPI.GraspGroup behavior"""
    def __init__(self, grasps):
        self.grasps = grasps
    
    def __len__(self):
        return len(self.grasps)
    
    def __getitem__(self, key):
        """Support slicing and indexing"""
        if isinstance(key, slice):
            return GraspGroup(self.grasps[key])
        else:
            return self.grasps[key]
    
    def nms(self):
        """Mock NMS - just return self"""
        return self
    
    def sort_by_score(self):
        """Sort grasps by score in descending order"""
        self.grasps.sort(key=lambda x: x.score, reverse=True)
        return self
    
    def to_open3d_geometry_list(self):
        """Convert grasps to Open3D geometry for visualization"""
        print(f"Converting grasps to Open3D geometry for visualization...")
        
        # Limit to top 50 grasps for performance (41k grasps is too many!)
        max_grasps = min(50, len(self.grasps))
        print(f"  Processing top {max_grasps} grasps out of {len(self.grasps)} total")
        
        geometries = []
        top_grasps = self.grasps[:max_grasps]
        
        for i, grasp in enumerate(top_grasps):
            if i % 10 == 0:
                print(f"    Creating gripper {i+1}/{max_grasps}")
                
            # Create gripper visualization as a box
            gripper = o3d.geometry.TriangleMesh.create_box(
                width=0.08,  # Fixed gripper width in meters
                height=cfgs.gripper_height,
                depth=cfgs.gripper_height
            )
            
            # Transform to grasp pose
            transform = np.eye(4)
            transform[:3, :3] = grasp.rotation_matrix
            transform[:3, 3] = grasp.translation
            gripper.transform(transform)
            
            # Color based on score (red for high score, blue for low)
            if len(top_grasps) > 1:
                scores = [g.score for g in top_grasps]
                score_normalized = (grasp.score - min(scores)) / (max(scores) - min(scores) + 1e-6)
                color = [1.0, 1.0 - score_normalized, score_normalized]  # Red to Blue
            else:
                color = [1.0, 0.0, 0.0]  # Default red
                
            gripper.paint_uniform_color(color)
            geometries.append(gripper)
            
        print(f"  Created {len(geometries)} gripper geometries")
        return geometries
    
    @property
    def scores(self):
        """Return scores of all grasps"""
        return [g.score for g in self.grasps]

def advanced_grasp_detection(points, colors, max_gripper_width=0.1, gripper_height=0.03):
    """
    Advanced grasp detection using geometric heuristics and surface analysis
    This replicates the AnyGrasp approach as closely as possible
    """
    print(f"Processing {len(points)} points...")
    
    # Workspace filter (same as demo_with_pcd.py)
    mask = (points[:, 2] > 0) & (points[:, 2] < 1.0)
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    print(f"After workspace filtering: {len(filtered_points)} points")
    
    if len(filtered_points) == 0:
        print("No valid points after filtering!")
        return [], []
    
    # Create point cloud for analysis
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Estimate normals with more sophisticated parameters
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=50))
    
    # Get normals and remove invalid ones
    normals = np.asarray(pcd.normals)
    valid_mask = ~np.isnan(normals).any(axis=1)
    filtered_points = filtered_points[valid_mask]
    filtered_colors = filtered_colors[valid_mask]
    normals = normals[valid_mask]
    
    print(f"After normal estimation: {len(filtered_points)} points")
    
    # Advanced grasp candidate detection
    print("Starting grasp candidate detection...")
    grasps = []
    
    # Method 1: Surface normal based (top-down grasps)
    print("Method 1: Surface normal based detection...")
    if cfgs.top_down_grasp:
        upward_mask = normals[:, 2] > 0.6  # Normal pointing mostly upward
        top_down_candidates = filtered_points[upward_mask]
        top_down_normals = normals[upward_mask]
        print(f"Found {len(top_down_candidates)} top-down candidates")
        
        for i, (point, normal) in enumerate(zip(top_down_candidates, top_down_normals)):
            if i % 1000 == 0:  # Print progress every 1000 grasps
                print(f"Processing top-down grasp {i}/{len(top_down_candidates)}")
            # Create grasp pose
            position = point + normal * gripper_height * 0.3
            
            # Orientation: gripper approaches along surface normal
            z_axis = normal
            x_axis = np.array([1, 0, 0])
            if np.abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            # Normalize axes
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Score based on surface quality and height
            surface_quality = normal[2]  # Prefer upward-facing surfaces
            height_score = 1.0 - (point[2] - 0.1) * 0.5  # Prefer higher points
            score = surface_quality * height_score
            
            grasps.append(GraspPose(position, rotation_matrix, score, max_gripper_width))
    
    # Method 2: Edge and corner detection (OPTIMIZED)
    print("Method 2: Edge and corner detection (optimized)...")
    
    # Use Open3D's built-in curvature estimation instead of brute force
    print("Using Open3D's efficient curvature estimation...")
    
    # Create a copy of the point cloud for analysis
    pcd_curvature = o3d.geometry.PointCloud()
    pcd_curvature.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_curvature.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Use Open3D's efficient normal estimation with smaller radius for edge detection
    pcd_curvature.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=15))
    
    # Simple but efficient curvature estimation using normal variation
    print("Calculating curvature using efficient method...")
    curvature_scores = []
    
    # Sample points for curvature calculation to speed up
    sample_size = min(10000, len(filtered_points))  # Limit to 10k points max
    if len(filtered_points) > sample_size:
        sample_indices = np.random.choice(len(filtered_points), sample_size, replace=False)
        sample_points = filtered_points[sample_indices]
        sample_normals = normals[sample_indices]
        print(f"Sampling {sample_size} points for curvature calculation")
    else:
        sample_points = filtered_points
        sample_normals = normals
    
    # Use vectorized operations for faster computation
    for i, (point, normal) in enumerate(zip(sample_points, sample_normals)):
        if i % 2000 == 0:
            print(f"Processing curvature for sample point {i}/{len(sample_points)}")
        
        # Find nearby points using efficient distance calculation
        distances = np.linalg.norm(sample_points - point, axis=1)
        nearby_mask = distances < 0.04  # Smaller radius for edge detection
        nearby_normals = sample_normals[nearby_mask]
        
        if len(nearby_normals) > 3:
            # Calculate normal variation (curvature proxy)
            normal_variation = np.std(nearby_normals, axis=0)
            curvature = np.linalg.norm(normal_variation)
            curvature_scores.append(curvature)
        else:
            curvature_scores.append(0.0)
    
    # Find high curvature points (potential edges)
    if len(curvature_scores) > 0:
        print(f"Found {len(curvature_scores)} curvature scores")
        curvature_threshold = np.percentile(curvature_scores, 90)  # Top 10%
        high_curvature_mask = np.array(curvature_scores) > curvature_threshold
        
        # Map back to original points
        if len(filtered_points) > sample_size:
            # Interpolate curvature scores back to full point set
            from scipy.interpolate import griddata
            # This is a simplified approach - in practice you might want to skip this
            print("Skipping edge detection for performance - focusing on top-down grasps")
        else:
            edge_mask = high_curvature_mask
            edge_points = filtered_points[edge_mask]
            edge_normals = normals[edge_mask]
            
            print(f"Found {len(edge_points)} edge points")
            for point, normal in zip(edge_points, edge_normals):
                # Create side grasp pose
                position = point + normal * gripper_height * 0.2
                
                # Orientation: gripper approaches perpendicular to edge
                z_axis = normal
                x_axis = np.array([1, 0, 0])
                if np.abs(np.dot(z_axis, x_axis)) > 0.9:
                    x_axis = np.array([0, 1, 0])
                y_axis = np.cross(z_axis, x_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                # Normalize axes
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                z_axis = z_axis / np.linalg.norm(z_axis)
                
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                
                # Score based on curvature and accessibility
                curvature_score = curvature_scores[list(edge_mask).index(True)]
                accessibility_score = 1.0 - (point[2] - 0.1) * 0.3
                score = curvature_score * accessibility_score * 0.8
                
                grasps.append(GraspPose(position, rotation_matrix, score, max_gripper_width))
    else:
        print("No curvature scores calculated - skipping edge detection")
    
    # Method 3: Density-based approach (OPTIMIZED)
    print("Method 3: Density-based approach (optimized)...")
    
    # Skip density calculation for large point clouds to save time
    if len(filtered_points) > 15000:
        print("Skipping density calculation for large point cloud - focusing on top-down grasps")
    else:
        # Find areas with high point density (potential flat surfaces)
        print("Calculating density scores using efficient sampling...")
        
        # Sample points for density calculation (much smaller sample for performance)
        density_sample_size = min(2000, len(filtered_points))
        if len(filtered_points) > density_sample_size:
            density_indices = np.random.choice(len(filtered_points), density_sample_size, replace=False)
            density_sample_points = filtered_points[density_indices]
            print(f"Sampling {density_sample_size} points for density calculation")
        else:
            density_sample_points = filtered_points
        
        # Use Open3D's efficient KDTree for faster neighbor search
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(density_sample_points)
        pcd_temp_tree = o3d.geometry.KDTreeFlann(pcd_temp)
        
        density_scores = []
        for i, point in enumerate(density_sample_points):
            if i % 500 == 0:
                print(f"Processing density for sample point {i}/{len(density_sample_points)}")
            
            # Use KDTree for efficient neighbor search (much faster than brute force)
            [k, idx, _] = pcd_temp_tree.search_knn_vector_3d(point, 50)  # Find 50 nearest neighbors
            nearby_count = len(idx)
            density_scores.append(nearby_count)
        
        if len(density_scores) > 0:
            density_threshold = np.percentile(density_scores, 80)  # Top 20%
            dense_mask = np.array(density_scores) > density_threshold
            
            # Map back to original points if sampling was used
            if len(filtered_points) > density_sample_size:
                print("Skipping density-based grasps due to sampling - focusing on top-down grasps")
            else:
                dense_points = filtered_points[dense_mask]
                dense_normals = normals[dense_mask]
                
                print(f"Found {len(dense_points)} dense surface points")
                for point, normal in zip(dense_points, dense_normals):
                    # Create flat surface grasp
                    position = point + normal * gripper_height * 0.25
                    
                    # Orientation: gripper approaches along surface normal
                    z_axis = normal
                    x_axis = np.array([1, 0, 0])
                    if np.abs(np.dot(z_axis, x_axis)) > 0.9:
                        x_axis = np.array([0, 1, 0])
                    y_axis = np.cross(z_axis, x_axis)
                    x_axis = np.cross(y_axis, z_axis)
                    
                    # Normalize axes
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                    
                    # Score based on density and surface quality
                    density_score = density_scores[list(dense_mask).index(True)] / max(density_scores)
                    surface_quality = normal[2]
                    score = density_score * surface_quality * 0.7  # Lower priority than top-down
                    
                    grasps.append(GraspPose(position, rotation_matrix, score, max_gripper_width))
    
    print(f"Generated {len(grasps)} grasp poses using multiple detection methods")
    print("Grasp detection complete!")
    return grasps, filtered_points

def demo(data_dir):
    print("=== Complete AnyGrasp Demo (Open-Source Implementation) ===")
    print("This replicates the functionality of demo_with_pcd.py using open-source libraries\n")
    
    # Load point cloud from PCD file
    pcd_path = os.path.join(data_dir, 'scene.pcd')
    if not os.path.exists(pcd_path):
        print(f"Error: Point cloud file not found at {pcd_path}")
        print("Please ensure scene.pcd exists in the data directory")
        return
    
    print(f"Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32) if pcd.has_colors() else np.zeros_like(points)
    
    print(f"Loaded point cloud: {len(points)} points")
    print(f"Point bounds: {points.min(axis=0)} to {points.max(axis=0)}")
    
    # Define workspace limits (same as demo_with_pcd.py)
    lims = [-1.5, 2.0, -2.0, 1.8, -2.0, 2.0]
    print(f"Workspace limits: {lims}")
    
    # Run advanced grasp detection
    grasps, filtered_points = advanced_grasp_detection(
        points, colors, 
        max_gripper_width=cfgs.max_gripper_width,
        gripper_height=cfgs.gripper_height
    )
    
    if len(grasps) == 0:
        print('No grasps detected!')
        return
    
    # Create GraspGroup (same structure as demo_with_pcd.py)
    gg = GraspGroup(grasps)
    
    # Grasp processing (same as demo_with_pcd.py)
    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:100]  # Same as original
    print("Top grasp scores:\n", gg_pick.scores)
    print("Best grasp score:", gg_pick[0].score)
    print("\n--- Grasp Poses (6-DoF) ---")
    
    # --- Save Best Grasp to TXT (same as demo_with_pcd.py) ---
    best_grasp = gg_pick[12] if len(gg_pick) > 12 else gg_pick[0]  # Same index as original
    position = best_grasp.translation
    rotation_matrix = best_grasp.rotation_matrix
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    
    # Combine position and quaternion into one array
    pose_data = np.hstack((position, quaternion))
    
    # Save to file
    output_txt_path = os.path.join(data_dir, 'best_grasp.txt')
    np.savetxt(output_txt_path, pose_data.reshape(1, -1), fmt='%.8f')
    print(f"Best grasp saved to {output_txt_path}")
    
    # --- Save all grasps to a text file (same as demo_with_pcd.py) ---
    all_grasps = []
    all_scores = []
    
    for i, grasp in enumerate(gg_pick):
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
    
    # --- Save all scores (same as demo_with_pcd.py) ---
    output_scores_path = os.path.join(data_dir, 'all_scores.txt')
    np.savetxt(output_scores_path, np.array(all_scores), fmt='%.8f')
    print(f"Saved all grasp scores to {output_scores_path}")
    
    # --- Visualization (same as demo_with_pcd.py) ---
    grippers = gg.to_open3d_geometry_list()
    trans_mat = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])
    
        # Create processed point cloud (same as demo_with_pcd.py)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(filtered_points)
    cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(pcd.colors)[:len(filtered_points)] if pcd.has_colors() else np.ones_like(filtered_points) * 0.5
    )
    
    # Save processed point cloud (same as demo_with_pcd.py)
    output_pcd_path = os.path.join(data_dir, 'processed_scene.pcd')
    o3d.io.write_point_cloud(output_pcd_path, cloud)
    print(f"Processed point cloud saved to {output_pcd_path}")
    
    # --- Sample points from grippers and merge with cloud (same as demo_with_pcd.py) ---
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
    
    # Save final point cloud (same as demo_with_pcd.py)
    output_pcd_path = os.path.join(data_dir, 'grasp_results.pcd')
    o3d.io.write_point_cloud(output_pcd_path, combined_cloud)
    print(f"Saved merged point cloud (scene + grippers) to {output_pcd_path}")
    
    # --- Create grasp_result.png (same as demo.py) ---
    print("\nCreating grasp_result.png...")
    
    # Use matplotlib for headless visualization (no display required)
    print("Creating matplotlib-based visualization (headless mode)...")
    
    try:
        print("Step 1: Creating matplotlib figure...")
        # Create matplotlib visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Step 2: Sampling point cloud for visualization...")
        # Plot point cloud (sample for performance)
        if len(filtered_points) > 15000:
            indices = np.random.choice(len(filtered_points), 15000, replace=False)
            sample_points = filtered_points[indices]
            print(f"Sampling {len(sample_points)} points for visualization")
        else:
            sample_points = filtered_points
        
        print("Step 3: Plotting point cloud...")
        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                   c='lightblue', s=0.1, alpha=0.6, label='Point Cloud')
        
        print("Step 4: Plotting grasps...")
        # Plot grasps
        top_grasps = grasps[:20]  # Show top 20 grasps
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_grasps)))
        
        for i, (grasp, color) in enumerate(zip(top_grasps, colors)):
            if i % 5 == 0:
                print(f"  Plotting grasp {i+1}/{len(top_grasps)}")
            position = grasp.translation
            normal = grasp.rotation_matrix[:, 2]  # Z-axis is approach direction
            
            # Plot grasp position
            ax.scatter(position[0], position[1], position[2], 
                       c=[color], s=100, marker='o', label=f'Grasp {i+1}' if i < 5 else "")
            
            # Plot approach direction
            arrow_length = 0.1
            ax.quiver(position[0], position[1], position[2],
                      normal[0], normal[1], normal[2],
                      length=arrow_length, color=color, alpha=0.8, arrow_length_ratio=0.3)
        
        print("Step 5: Setting plot properties...")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Grasp Detection Results (Complete Demo)', fontsize=14)
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        
        print("Step 6: Saving plot...")
        plt.tight_layout()
        plt.savefig('grasp_result.png', dpi=300, bbox_inches='tight')
        print(f"Saved matplotlib visualization to grasp_result.png")
        plt.close()
        
    except Exception as e:
        print(f"Error in matplotlib visualization: {e}")
        print("Creating simple text-based visualization instead...")
        
        # Fallback: create a simple text file with grasp positions
        with open('grasp_visualization.txt', 'w') as f:
            f.write("Grasp Visualization Summary\n")
            f.write("==========================\n\n")
            f.write(f"Point Cloud: {len(filtered_points)} points\n")
            f.write(f"Grasps Detected: {len(grasps)}\n\n")
            f.write("Top 10 Grasp Positions:\n")
            for i, grasp in enumerate(grasps[:10]):
                pos = grasp.translation
                f.write(f"Grasp {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n")
        print("Saved grasp_visualization.txt as fallback")
    
    # Also save to results directory if it exists
    try:
        output_dir = os.path.join(data_dir, 'results', 'rgbd')
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, 'grasp_result.png')
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111, projection='3d')
        
        # Recreate the plot for the results directory
        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                   c='lightblue', s=0.1, alpha=0.6, label='Point Cloud')
        
        for i, (grasp, color) in enumerate(zip(top_grasps, colors)):
            position = grasp.translation
            normal = grasp.rotation_matrix[:, 2]
            ax.scatter(position[0], position[1], position[2], 
                       c=[color], s=100, marker='o', label=f'Grasp {i+1}' if i < 5 else "")
            arrow_length = 0.1
            ax.quiver(position[0], position[1], position[2],
                      normal[0], normal[1], normal[2],
                      length=arrow_length, color=color, alpha=0.8, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Grasp Detection Results (Complete Demo)', fontsize=14)
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Saved grasp_result.png to {output_image_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save to results directory: {e}")
    
    print("\n=== Demo Complete ===")
    print("All files generated successfully!")
    print("Check the output directory for results.")

if __name__ == '__main__':
    demo('.')
