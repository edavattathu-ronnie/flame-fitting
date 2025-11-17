import numpy as np
import trimesh
import os

def visualize_flame_alignment(scan, flame_model, landmarks_3d, output_dir, stage="before_fitting"):
    """
    Create visualization files to check FLAME-scan alignment
    
    Args:
        scan: TrimeshWrapper of scan mesh
        flame_model: FLAME model (chumpy object)
        landmarks_3d: 3D landmarks for reference
        output_dir: Directory to save visualization files
        stage: "before_fitting", "after_rigid", "after_full"
    """
    
    print(f"📊 Creating alignment visualization ({stage})...")
    
    # Get current FLAME mesh vertices
    if hasattr(flame_model, 'r'):
        flame_vertices = flame_model.r.copy()
    else:
        flame_vertices = flame_model.copy()
    
    flame_faces = flame_model.f if hasattr(flame_model, 'f') else flame_model[1]
    
    # Create visualization meshes
    scan_mesh = trimesh.Trimesh(vertices=scan.v, faces=scan.f)
    flame_mesh = trimesh.Trimesh(vertices=flame_vertices, faces=flame_faces)
    
    # Color the meshes differently
    scan_mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray
    flame_mesh.visual.face_colors = [255, 100, 100, 255]  # Red
    
    # Save individual meshes
    scan_path = os.path.join(output_dir, f"scan_{stage}.ply")
    flame_path = os.path.join(output_dir, f"flame_{stage}.ply")
    
    scan_mesh.export(scan_path)
    flame_mesh.export(flame_path)
    
    # Create combined scene
    scene = trimesh.Scene([scan_mesh, flame_mesh])
    combined_path = os.path.join(output_dir, f"alignment_{stage}.ply")
    scene.export(combined_path)
    
    # Create landmarks visualization
    landmarks_spheres = []
    for i, lmk in enumerate(landmarks_3d):
        sphere = trimesh.primitives.Sphere(radius=0.002, center=lmk)
        sphere.visual.face_colors = [0, 255, 0, 255]  # Green
        landmarks_spheres.append(sphere)
    
    # Combined scene with landmarks
    scene_with_landmarks = trimesh.Scene([scan_mesh, flame_mesh] + landmarks_spheres)
    landmarks_path = os.path.join(output_dir, f"alignment_with_landmarks_{stage}.ply")
    scene_with_landmarks.export(landmarks_path)
    
    # Print alignment statistics
    print_alignment_stats(scan.v, flame_vertices, landmarks_3d, stage)
    
    print(f"✅ Visualization saved:")
    print(f"   Combined: {combined_path}")
    print(f"   With landmarks: {landmarks_path}")
    
    return combined_path, landmarks_path

def print_alignment_stats(scan_vertices, flame_vertices, landmarks_3d, stage):
    """Print alignment statistics"""
    
    print(f"\n📊 ALIGNMENT STATISTICS ({stage.upper()})")
    print("-" * 50)
    
    # Bounding box comparison
    scan_bbox = np.array([np.min(scan_vertices, axis=0), np.max(scan_vertices, axis=0)])
    flame_bbox = np.array([np.min(flame_vertices, axis=0), np.max(flame_vertices, axis=0)])
    landmark_bbox = np.array([np.min(landmarks_3d, axis=0), np.max(landmarks_3d, axis=0)])
    
    scan_center = np.mean(scan_bbox, axis=0)
    flame_center = np.mean(flame_bbox, axis=0)
    landmark_center = np.mean(landmark_bbox, axis=0)
    
    scan_size = scan_bbox[1] - scan_bbox[0]
    flame_size = flame_bbox[1] - flame_bbox[0]
    
    print(f"Scan center:     [{scan_center[0]:>7.3f}, {scan_center[1]:>7.3f}, {scan_center[2]:>7.3f}]")
    print(f"FLAME center:    [{flame_center[0]:>7.3f}, {flame_center[1]:>7.3f}, {flame_center[2]:>7.3f}]")
    print(f"Landmark center: [{landmark_center[0]:>7.3f}, {landmark_center[1]:>7.3f}, {landmark_center[2]:>7.3f}]")
    
    print(f"\nScan size:       [{scan_size[0]:>7.3f}, {scan_size[1]:>7.3f}, {scan_size[2]:>7.3f}]")
    print(f"FLAME size:      [{flame_size[0]:>7.3f}, {flame_size[1]:>7.3f}, {flame_size[2]:>7.3f}]")
    
    # Distance between centers
    center_distance = np.linalg.norm(scan_center - flame_center)
    print(f"\nCenter distance: {center_distance:.6f}m")
    
    # Size ratio
    size_ratio = scan_size / (flame_size + 1e-8)  # Avoid division by zero
    print(f"Size ratios:     [{size_ratio[0]:>7.3f}, {size_ratio[1]:>7.3f}, {size_ratio[2]:>7.3f}]")
    
    # Alignment quality assessment
    print(f"\n🎯 ALIGNMENT ASSESSMENT:")
    if center_distance < 0.01:  # 1cm
        print("✅ Centers are well aligned")
    elif center_distance < 0.05:  # 5cm
        print("⚠️  Centers are somewhat aligned")
    else:
        print("❌ Centers are poorly aligned")
    
    if np.all(np.abs(size_ratio - 1.0) < 0.3):  # Within 30%
        print("✅ Sizes are well matched")
    elif np.all(np.abs(size_ratio - 1.0) < 0.5):  # Within 50%
        print("⚠️  Sizes are somewhat matched")
    else:
        print("❌ Sizes are poorly matched")

def create_alignment_checker():
    """
    Create a standalone alignment checker that can be called at any stage
    """
    
    def check_alignment(scan, flame_model, landmarks_3d, output_dir, stage="check"):
        """Check and visualize current alignment"""
        
        safe_mkdir(output_dir)
        
        # Create visualizations
        combined_path, landmarks_path = visualize_flame_alignment(
            scan, flame_model, landmarks_3d, output_dir, stage)
        
        # Additional analysis
        flame_vertices = flame_model.r if hasattr(flame_model, 'r') else flame_model
        
        print(f"\n🔍 DETAILED ANALYSIS")
        print("-" * 50)
        
        # Check if models are facing same direction
        scan_front = np.mean(scan.v[scan.v[:, 2] > np.percentile(scan.v[:, 2], 90)], axis=0)
        flame_front = np.mean(flame_vertices[flame_vertices[:, 2] > np.percentile(flame_vertices[:, 2], 90)], axis=0)
        
        scan_back = np.mean(scan.v[scan.v[:, 2] < np.percentile(scan.v[:, 2], 10)], axis=0)
        flame_back = np.mean(flame_vertices[flame_vertices[:, 2] < np.percentile(flame_vertices[:, 2], 10)], axis=0)
        
        scan_direction = scan_front - scan_back
        flame_direction = flame_front - flame_back
        
        # Normalize directions
        scan_direction = scan_direction / np.linalg.norm(scan_direction)
        flame_direction = flame_direction / np.linalg.norm(flame_direction)
        
        # Calculate alignment
        direction_similarity = np.dot(scan_direction, flame_direction)
        
        print(f"Direction similarity: {direction_similarity:.3f}")
        if direction_similarity > 0.8:
            print("✅ Models are facing the same direction")
        elif direction_similarity > 0.0:
            print("⚠️  Models are somewhat aligned")
        elif direction_similarity > -0.8:
            print("🔄 Models might be rotated relative to each other")
        else:
            print("❌ Models are facing opposite directions")
        
        return {
            'combined_path': combined_path,
            'landmarks_path': landmarks_path,
            'direction_similarity': direction_similarity,
            'alignment_quality': 'good' if direction_similarity > 0.8 else 'poor'
        }
    
    return check_alignment

def safe_mkdir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Example usage function
def quick_alignment_check(scan, flame_model, landmarks_3d, output_dir="alignment_check"):
    """
    Quick function to check current alignment
    
    Usage:
        results = quick_alignment_check(scan, flame_model, landmarks_3d)
        print(f"Alignment quality: {results['alignment_quality']}")
    """
    
    checker = create_alignment_checker()
    return checker(scan, flame_model, landmarks_3d, output_dir)

if __name__ == "__main__":
    print("Alignment visualization tools loaded!")
    print("Use quick_alignment_check(scan, flame_model, landmarks_3d) to check alignment")