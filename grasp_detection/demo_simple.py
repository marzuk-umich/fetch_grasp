import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def simple_grasp_detection(points, colors, max_gripper_width=0.1, gripper_height=0.03):
    """
    Simple grasp detection using geometric heuristics
    This is a simplified version for demonstration purposes
    """
    print(f"Processing {len(points)} points...")
    
    # Filter points by height (remove floor and ceiling)
    mask = (points[:, 2] > 0.05) & (points[:, 2] < 0.8)
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    print(f"After height filtering: {len(filtered_points)} points")
    
    if len(filtered_points) == 0:
        print("No valid points after filtering!")
        return [], []
    
    # Find potential grasp points using surface normals and curvature
    # For simplicity, we'll use a basic approach based on point density
    
    # Create a KD-tree for efficient nearest neighbor search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # Get normals
    normals = np.asarray(pcd.normals)
    
    # Find points with upward-facing normals (potential grasp surfaces)
    upward_mask = normals[:, 2] > 0.7  # Normal pointing mostly upward
    grasp_candidates = filtered_points[upward_mask]
    grasp_normals = normals[upward_mask]
    
    print(f"Found {len(grasp_candidates)} potential grasp points")
    
    if len(grasp_candidates) == 0:
        print("No grasp candidates found!")
        return [], []
    
    # Simple grasp pose generation
    grasps = []
    for i, (point, normal) in enumerate(zip(grasp_candidates, grasp_normals)):
        # Create a simple grasp pose
        # Position: slightly above the surface point
        position = point + normal * gripper_height * 0.5
        
        # Orientation: align gripper with surface normal
        # For simplicity, we'll create a basic rotation matrix
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
        
        # Simple grasp score based on height and normal alignment
        score = normal[2] * (1.0 - point[2] * 0.5)  # Prefer higher points with upward normals
        
        grasps.append({
            'position': position,
            'rotation_matrix': rotation_matrix,
            'score': score,
            'normal': normal
        })
    
    # Sort grasps by score
    grasps.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Generated {len(grasps)} grasp poses")
    return grasps, filtered_points

def create_gripper_visualization(grasp, gripper_width=0.08, gripper_height=0.03):
    """Create a simple gripper visualization"""
    position = grasp['position']
    rotation = grasp['rotation_matrix']
    
    # Create gripper geometry (simplified as a box)
    gripper = o3d.geometry.TriangleMesh.create_box(
        width=gripper_width, 
        height=gripper_height, 
        depth=gripper_height
    )
    
    # Transform gripper to grasp pose
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    gripper.transform(transform)
    
    return gripper

def create_png_visualization(points, grasps, output_path='grasp_result.png'):
    """Create a 3D matplotlib visualization and save as PNG"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud (sample points for performance)
    if len(points) > 10000:
        # Sample points for visualization
        indices = np.random.choice(len(points), 10000, replace=False)
        sample_points = points[indices]
    else:
        sample_points = points
    
    # Plot point cloud
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
               c='lightblue', s=0.1, alpha=0.6, label='Point Cloud')
    
    # Plot top grasps
    top_grasps = grasps[:10]  # Show top 10 grasps
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_grasps)))
    
    for i, (grasp, color) in enumerate(zip(top_grasps, colors)):
        position = grasp['position']
        normal = grasp['normal']
        
        # Plot grasp position
        ax.scatter(position[0], position[1], position[2], 
                   c=[color], s=100, marker='o', label=f'Grasp {i+1}' if i < 3 else "")
        
        # Plot approach direction (normal vector)
        arrow_length = 0.1
        end_point = position + normal * arrow_length
        ax.quiver(position[0], position[1], position[2],
                  normal[0], normal[1], normal[2],
                  length=arrow_length, color=color, alpha=0.8, arrow_length_ratio=0.3)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Grasp Detection Results\n(Simplified Demo)', fontsize=14)
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()

def demo(data_dir):
    print("=== Simple Grasp Detection Demo ===")
    print("Note: This is a simplified version using only open-source libraries")
    print("For full AnyGrasp functionality, you need the commercial SDK\n")
    
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
    
    # Run simple grasp detection
    grasps, filtered_points = simple_grasp_detection(
        points, colors, 
        max_gripper_width=cfgs.max_gripper_width,
        gripper_height=cfgs.gripper_height
    )
    
    if len(grasps) == 0:
        print('No grasps detected!')
        return
    
    # Display top grasps
    print(f"\n=== Top 5 Grasp Candidates ===")
    for i, grasp in enumerate(grasps[:5]):
        print(f"Grasp {i+1}:")
        print(f"  Position: {grasp['position']}")
        print(f"  Score: {grasp['score']:.4f}")
        print(f"  Normal: {grasp['normal']}")
        print("-" * 40)
    
    # Save grasp data
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save top grasps
    top_grasps = grasps[:10]
    grasp_data = []
    for grasp in top_grasps:
        quaternion = R.from_matrix(grasp['rotation_matrix']).as_quat()
        pose_data = np.hstack((grasp['position'], quaternion))
        grasp_data.append(pose_data)
    
    grasp_data = np.array(grasp_data)
    output_path = os.path.join(output_dir, 'simple_grasps.txt')
    np.savetxt(output_path, grasp_data, fmt='%.8f')
    print(f"\nSaved {len(top_grasps)} grasps to: {output_path}")
    
    # Save scores
    scores = [g['score'] for g in top_grasps]
    scores_path = os.path.join(output_dir, 'simple_grasp_scores.txt')
    np.savetxt(scores_path, scores, fmt='%.8f')
    print(f"Saved grasp scores to: {scores_path}")
    
    # Create visualization
    if cfgs.debug:
        print("\nCreating visualization...")
        
        # Create point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        vis_pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[:len(filtered_points)] if pcd.has_colors() else np.ones_like(filtered_points) * 0.5
        )
        
        # Create gripper visualizations
        grippers = []
        for grasp in top_grasps[:5]:  # Show top 5 grasps
            gripper = create_gripper_visualization(grasp, cfgs.max_gripper_width, cfgs.gripper_height)
            gripper.paint_uniform_color([1, 0, 0])  # Red color for grasps
            grippers.append(gripper)
        
        # Save processed point cloud
        output_pcd_path = os.path.join(output_dir, 'processed_scene_simple.pcd')
        o3d.io.write_point_cloud(output_pcd_path, vis_pcd)
        print(f"Saved processed point cloud to: {output_pcd_path}")
        
        # Save gripper mesh
        if grippers:
            combined_grippers = grippers[0]
            for gripper in grippers[1:]:
                combined_grippers += gripper
            
            gripper_path = os.path.join(output_dir, 'simple_grippers.ply')
            o3d.io.write_triangle_mesh(gripper_path, combined_grippers)
            print(f"Saved gripper visualization to: {gripper_path}")
        
        # Create PNG visualization
        create_png_visualization(filtered_points, top_grasps, 'grasp_result.png')
        
        print("\nVisualization files saved. You can view them with Open3D or other 3D viewers.")
        print("Note: Interactive visualization requires a display environment.")

if __name__ == '__main__':
    demo('.')
