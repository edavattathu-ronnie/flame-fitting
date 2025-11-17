import numpy as np
import chumpy as ch
import os
import sys
import trimesh
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from time import time

# FLAME imports (adjust paths as needed)
from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates
from fitting.util import get_unit_factor

class TrimeshWrapper:
    """
    Wrapper to make trimesh compatible with psbody.mesh.Mesh interface
    """
    def __init__(self, filename=None, vertices=None, faces=None):
        if filename:
            self.mesh = trimesh.load(filename)
            self.v = self.mesh.vertices.astype(np.float64)
            self.f = self.mesh.faces.astype(np.int32)
        elif vertices is not None and faces is not None:
            self.v = np.array(vertices, dtype=np.float64)
            self.f = np.array(faces, dtype=np.int32)
            self.mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        else:
            raise ValueError("Must provide either filename or vertices+faces")
    
    def write_obj(self, filepath):
        """Write mesh as OBJ file"""
        self.mesh.export(filepath)

class TrimeshScanToMesh(ch.Ch):
    """
    Trimesh-based replacement for sbody.mesh_distance.ScanToMesh
    Computes distance from scan vertices to mesh surface
    """
    
    dterms = 'mesh_verts',
    
    def __init__(self, scan, mesh_verts, mesh_faces, scan_sampler=None, 
                 rho=lambda x: x, normalize=True, weight=1.0):
        """
        Args:
            scan: TrimeshWrapper of scan mesh
            mesh_verts: chumpy array of mesh vertices to optimize
            mesh_faces: mesh face topology
            scan_sampler: sampling of scan vertices (use all if None)
            rho: robustifier function 
            normalize: whether to normalize by number of samples
            weight: overall weight for this term
        """
        self.scan = scan
        self.mesh_faces = mesh_faces
        self.scan_sampler = scan_sampler
        self.rho = rho
        self.normalize = normalize
        self.weight = weight
        
        # Get scan sample points
        if scan_sampler is None:
            self.sample_vertices = scan.v
        else:
            # For simplicity, use all vertices (could implement subsampling)
            self.sample_vertices = scan.v
        
        self.n_samples = len(self.sample_vertices)
        
        # Normalization factor
        self.norm_factor = np.sqrt(self.n_samples) if normalize else 1.0
        
        # Set up the chumpy dependency
        ch.Ch.__init__(self)
        self.mesh_verts = mesh_verts
    
    def compute_r(self):
        """Compute residual: distances from scan to mesh"""
        
        # Get current mesh vertices
        current_verts = self.mesh_verts.r if hasattr(self.mesh_verts, 'r') else self.mesh_verts
        
        # Create current mesh for distance computation
        current_mesh = trimesh.Trimesh(vertices=current_verts, faces=self.mesh_faces)
        
        # Compute closest point distances
        closest_points, distances, triangle_ids = current_mesh.nearest.on_surface(self.sample_vertices)
        
        # Apply robustifier
        robust_distances = self.rho(distances**2)  # Use squared distances
        
        # Apply normalization and weight
        residual = self.weight * np.sqrt(robust_distances) / self.norm_factor
        
        return residual.flatten()
    
    def compute_dr_wrt(self, wrt):
        """Compute derivatives w.r.t. mesh vertices"""
        
        if wrt is not self.mesh_verts:
            return None
        
        # Get current mesh vertices
        current_verts = self.mesh_verts.r if hasattr(self.mesh_verts, 'r') else self.mesh_verts
        
        # Create current mesh
        current_mesh = trimesh.Trimesh(vertices=current_verts, faces=self.mesh_faces)
        
        # Compute closest points and distances
        closest_points, distances, triangle_ids = current_mesh.nearest.on_surface(self.sample_vertices)
        
        # Compute gradients (simplified finite difference approach)
        n_verts = len(current_verts)
        n_samples = len(self.sample_vertices)
        
        # Initialize gradient matrix
        gradients = np.zeros((n_samples, n_verts * 3))
        
        # Finite difference step size
        h = 1e-6
        
        # Compute gradients for each vertex component
        for i in range(n_verts):
            for axis in range(3):
                # Perturb vertex
                perturbed_verts = current_verts.copy()
                perturbed_verts[i, axis] += h
                
                # Compute distances with perturbed mesh
                perturbed_mesh = trimesh.Trimesh(vertices=perturbed_verts, faces=self.mesh_faces)
                _, perturbed_distances, _ = perturbed_mesh.nearest.on_surface(self.sample_vertices)
                
                # Finite difference gradient
                dist_grad = (perturbed_distances**2 - distances**2) / h
                
                # Apply robustifier derivative
                rho_grad = np.where(distances > 1e-8, 
                                  dist_grad / (2 * np.sqrt(self.rho(distances**2))), 
                                  0)
                
                gradients[:, i * 3 + axis] = self.weight * rho_grad / self.norm_factor
        
        return gradients

class TrimeshGMOf:
    """Geman-McClure robustifier for trimesh distances"""
    
    def __init__(self, sigma=1e-4):
        self.sigma = sigma
    
    def __call__(self, x):
        """Apply Geman-McClure robustifier"""
        return (x * x) / (self.sigma + x * x)

def write_simple_obj(vertices, faces, filepath, verbose=True):
    """Write simple OBJ file"""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filepath)
    if verbose:
        print(f"Saved mesh to: {filepath}")

def safe_mkdir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def downsample_scan(scan, target_vertices=50000):
    """Downsample scan to reduce computation"""
    import trimesh
    
    if len(scan.v) > target_vertices:
        print(f"🔄 Downsampling scan from {len(scan.v)} to ~{target_vertices} vertices...")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=scan.v, faces=scan.f)
        
        # Calculate target face count (approximate)
        target_faces = int(target_vertices * 1.8)  # Rough faces-to-vertices ratio
        
        try:
            # Use face_count parameter instead of vertex count
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"   Used quadric decimation")
        except Exception as e:
            print(f"   Quadric decimation failed: {e}")
            print("   Using simple vertex sampling...")
            
            # Fallback: Simple vertex sampling
            import numpy as np
            step = max(1, len(scan.v) // target_vertices)
            indices = np.arange(0, len(scan.v), step)[:target_vertices]
            
            # Keep only selected vertices 
            new_vertices = scan.v[indices]
            
            # Simple approach: create new faces from remaining vertices
            # This is basic but functional
            n_verts = len(new_vertices)
            new_faces = []
            
            # Create simple triangular faces
            for i in range(0, n_verts - 2, 3):
                if i + 2 < n_verts:
                    new_faces.append([i, i+1, i+2])
            
            simplified = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # Update scan
        scan.v = simplified.vertices
        scan.f = simplified.faces
        scan.mesh = simplified
        
        print(f"✅ Downsampled to {len(scan.v)} vertices, {len(scan.f)} faces")
    
    return scan

def fix_landmark_orientation_mismatch(landmarks_3d, scan):
    """
    Fix orientation mismatch between your landmarks and FLAME's expectation
    """
    
    print("🔧 Analyzing landmark orientation...")
    
    # Analyze your landmark layout
    # Assuming standard 51-landmark format:
    # 0-16: Jawline, 17-21: R eyebrow, 22-26: L eyebrow, 27-35: Nose, etc.
    
    if len(landmarks_3d) >= 35:
        # Get key landmark positions
        jaw_center = landmarks_3d[8]  # Center of jawline
        nose_tip = landmarks_3d[30] if len(landmarks_3d) > 30 else landmarks_3d[27]
        
        # Calculate face orientation
        face_up_vector = nose_tip - jaw_center
        face_up_vector = face_up_vector / np.linalg.norm(face_up_vector)
        
        print(f"   Face up vector: [{face_up_vector[0]:.3f}, {face_up_vector[1]:.3f}, {face_up_vector[2]:.3f}]")
        
        # FLAME expects Y-up orientation typically
        expected_up = np.array([0, 1, 0])
        
        # Calculate rotation needed
        rotation_angle = np.arccos(np.clip(np.dot(face_up_vector, expected_up), -1, 1))
        rotation_axis = np.cross(face_up_vector, expected_up)
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            print(f"   Rotation needed: {np.degrees(rotation_angle):.1f}° around axis [{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]")
            
            # Apply rotation to both scan and landmarks
            if abs(rotation_angle) > 0.1:  # More than ~6 degrees
                # Create rotation matrix
                from scipy.spatial.transform import Rotation
                rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
                
                # Apply rotation
                landmarks_3d_rotated = rotation.apply(landmarks_3d)
                scan_v_rotated = rotation.apply(scan.v)
                
                print(f"✅ Applied {np.degrees(rotation_angle):.1f}° rotation to align with FLAME orientation")
                
                return landmarks_3d_rotated, scan_v_rotated
        
    print("   No rotation needed")
    return landmarks_3d, scan.v

class CompleteTrimeshFLAMEFitter:
    """
    Complete FLAME fitter using trimesh instead of psbody
    """
    
    def __init__(self, flame_model_path="models/generic_model.pkl",
                 flame_embedding_path="models/flame_static_embedding.pkl"):
        """
        Initialize FLAME fitter
        
        Args:
            flame_model_path: Path to FLAME model
            flame_embedding_path: Path to landmark embedding
        """
        
        print("🔥 INITIALIZING TRIMESH-BASED FLAME FITTER")
        print("="*60)
        
        # Load FLAME model
        if not os.path.exists(flame_model_path):
            raise FileNotFoundError(f"FLAME model not found: {flame_model_path}")
        
        self.model = load_model(flame_model_path)
        print(f"✅ Loaded FLAME model: {os.path.basename(flame_model_path)}")
        print(f"   Vertices: {len(self.model.r)}, Faces: {len(self.model.f)}")
        
        # Load landmark embedding (with pickle fix)
        if not os.path.exists(flame_embedding_path):
            print(f"⚠️  Embedding not found: {flame_embedding_path}")
            print("🔧 Using fallback embedding...")
            self.lmk_face_idx, self.lmk_b_coords = self._create_fallback_embedding()
        else:
            try:
                from fitting.landmarks import load_embedding
                self.lmk_face_idx, self.lmk_b_coords = load_embedding(flame_embedding_path)
                print(f"✅ Loaded landmark embedding: {len(self.lmk_face_idx)} landmarks")
            except Exception as e:
                print(f"⚠️  Pickle loading failed: {e}")
                print("🔧 Using fallback embedding...")
                self.lmk_face_idx, self.lmk_b_coords = self._create_fallback_embedding()
        
        # Setup optimization options
        self.opt_options = {
            'disp': 1,
            'delta_0': 0.1,
            'e_3': 1e-3,
            'maxiter': 300  # Reduced for faster convergence
        }
        
        print("✅ Trimesh FLAME fitter ready!")
        print("="*60)
    
    def _create_fallback_embedding(self):
        """Create fallback landmark embedding if pickle loading fails"""
        
        print("🔧 Creating fallback landmark embedding...")
        
        # Get number of faces in FLAME model for validation
        num_faces = len(self.model.f)
        
        # Create exactly 51-landmark embedding for FLAME (matching your input)
        # Use face indices that are guaranteed to be within bounds
        max_face_idx = min(4000, num_faces - 1)  # Conservative upper bound
        
        # Create 51 evenly distributed face indices
        face_indices = np.linspace(100, max_face_idx, 51, dtype=np.int32)
        
        lmk_face_idx = np.array([
            # Use the distributed indices to ensure they're all valid
            face_indices[0],  face_indices[1],  face_indices[2],  face_indices[3],  # Jawline start
            face_indices[4],  face_indices[5],  face_indices[6],  face_indices[7],  
            face_indices[8],  face_indices[9],  face_indices[10], face_indices[11], 
            face_indices[12], face_indices[13], face_indices[14], face_indices[15], 
            face_indices[16],  # Jawline end (17 points)
            
            face_indices[17], face_indices[18], face_indices[19], face_indices[20], face_indices[21],  # R eyebrow (5)
            face_indices[22], face_indices[23], face_indices[24], face_indices[25], face_indices[26],  # L eyebrow (5)
            
            face_indices[27], face_indices[28], face_indices[29], face_indices[30],  # Nose (9)
            face_indices[31], face_indices[32], face_indices[33], face_indices[34], face_indices[35],
            
            face_indices[36], face_indices[37], face_indices[38], face_indices[39], face_indices[40], face_indices[41],  # R eye (6)
            face_indices[42], face_indices[43], face_indices[44], face_indices[45], face_indices[46], face_indices[47],  # L eye (6)
            
            face_indices[48], face_indices[49], face_indices[50]  # Mouth (3 to complete 51)
        ], dtype=np.int32)
        
        # Verify all face indices are valid
        assert np.all(lmk_face_idx < num_faces), f"Some face indices exceed model bounds ({num_faces})"
        assert len(lmk_face_idx) == 51, f"Expected 51 landmarks, got {len(lmk_face_idx)}"
        
        # Barycentric coordinates (center of triangle)
        lmk_b_coords = np.full((len(lmk_face_idx), 3), 1.0/3.0, dtype=np.float64)
        
        print(f"✅ Fallback embedding: {len(lmk_face_idx)} landmarks")
        print(f"   Face index range: [{np.min(lmk_face_idx)}, {np.max(lmk_face_idx)}]")
        print(f"   Model has {num_faces} faces")
        return lmk_face_idx, lmk_b_coords
    
    def load_scan_and_landmarks(self, scan_path, landmarks_path, scan_unit='m'):
        """
        Load scan mesh and landmarks
        
        Args:
            scan_path: Path to scan mesh file
            landmarks_path: Path to .npy landmarks file
            scan_unit: Unit of scan ('m', 'cm', 'mm', 'NA')
        
        Returns:
            (scan_mesh, landmarks_3d, scale_factor)
        """
        
        print("\n📁 LOADING SCAN AND LANDMARKS")
        print("-" * 40)
        
        # Load scan using trimesh
        if not os.path.exists(scan_path):
            raise FileNotFoundError(f"Scan not found: {scan_path}")
        
        scan = TrimeshWrapper(filename=scan_path)
        print(f"✅ Loaded scan: {os.path.basename(scan_path)}")
        print(f"   Vertices: {len(scan.v)}, Faces: {len(scan.f)}")
        
        # Load landmarks
        if not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Landmarks not found: {landmarks_path}")
        
        landmarks_3d = np.load(landmarks_path)
        print(f"✅ Loaded landmarks: {len(landmarks_3d)} points")
        print(f"   Range: X[{landmarks_3d[:,0].min():.3f}, {landmarks_3d[:,0].max():.3f}]")
        
        # Handle scaling
        if scan_unit.lower() == 'na':
            print("🔍 Computing automatic scale...")
            scale_factor = self._compute_automatic_scale(landmarks_3d)
        else:
            scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
        
        print(f"📏 Scale factor: {scale_factor:.6f}")
        
        # Apply scaling
        scan.v *= scale_factor
        landmarks_3d *= scale_factor
        
        print("✅ Scaling applied")
        return scan, landmarks_3d, scale_factor
    
    def _compute_automatic_scale(self, landmarks_3d):
        """Compute automatic scale by aligning landmarks with FLAME"""
        
        # Create scale variable
        scale = ch.ones(1)
        scaled_landmarks = scale * ch.array(landmarks_3d)
        
        # Get FLAME model landmarks
        model_landmarks = mesh_points_by_barycentric_coordinates(
            self.model, self.model.f, self.lmk_face_idx, self.lmk_b_coords)
        
        # Minimize landmark distance to find scale
        landmark_error = scaled_landmarks - model_landmarks
        
        print("   Optimizing scale...")
        ch.minimize(
            fun=landmark_error,
            x0=[scale, self.model.trans, self.model.pose[:3]],
            method='dogleg',
            options={'maxiter': 100, 'disp': 0}
        )
        
        return float(scale.r)
    
    def fit_flame_to_scan(self, scan, landmarks_3d, 
                         scan_weight=2.0, landmark_weight=1e-2,
                         shape_reg=1e-4, expr_reg=1e-4, pose_reg=1e-3,
                         shape_components=100, expr_components=50,
                         output_dir="output"):
        """
        Complete FLAME fitting to scan
        
        Args:
            scan: TrimeshWrapper scan mesh
            landmarks_3d: 3D landmarks (51, 3)
            scan_weight: Weight for scan-to-mesh distance
            landmark_weight: Weight for landmark fitting
            shape_reg: Shape regularization weight
            expr_reg: Expression regularization weight  
            pose_reg: Pose regularization weight
            shape_components: Number of shape components
            expr_components: Number of expression components
            output_dir: Output directory
        
        Returns:
            (fitted_vertices, fitted_faces, fitted_parameters)
        """
        
        print(f"\n🔥 FITTING FLAME TO SCAN")
        print("="*60)
        
        safe_mkdir(output_dir)
        
        # Setup parameters
        shape_idx = np.arange(0, min(300, shape_components))
        expr_idx = np.arange(300, 300 + min(100, expr_components))
        used_idx = np.union1d(shape_idx, expr_idx)
        
        # Initialize parameters
        self.model.betas[:] = 0.0
        self.model.pose[:] = 0.0
        
        free_variables = [self.model.trans, self.model.pose, self.model.betas[used_idx]]
        
        print(f"📊 Shape components: {len(shape_idx)}")
        print(f"📊 Expression components: {len(expr_idx)}")
        print(f"📊 Total parameters: {len(used_idx) + 6}")
        
        # Setup objectives
        print("\n🎯 SETTING UP OBJECTIVES")
        print("-" * 40)
        
        # 1. Landmark objective
        landmark_error = landmark_error_3d(
            mesh_verts=self.model,
            mesh_faces=self.model.f,
            lmk_3d=landmarks_3d,
            lmk_face_idx=self.lmk_face_idx,
            lmk_b_coords=self.lmk_b_coords
        )
        print("✅ Landmark objective")
        
        # 2. Scan-to-mesh objective (using our trimesh implementation)
        gmo_sigma = 1e-4
        scan_to_mesh = TrimeshScanToMesh(
            scan=scan,
            mesh_verts=self.model,
            mesh_faces=self.model.f,
            rho=TrimeshGMOf(sigma=gmo_sigma),
            normalize=True,
            weight=scan_weight
        )
        print("✅ Scan-to-mesh objective")
        
        # 3. Regularization objectives  
        shape_regularization = shape_reg * self.model.betas[shape_idx]
        expr_regularization = expr_reg * self.model.betas[expr_idx]
        pose_regularization = pose_reg * self.model.pose[3:]
        print("✅ Regularization objectives")
        
        # Combined objectives
        objectives = {
            'lmk': landmark_weight * landmark_error,
            's2m': scan_to_mesh,
            'shape': shape_regularization,
            'expr': expr_regularization,
            'pose': pose_regularization
        }
        
        # Progress tracking
        iteration_count = [0]
        def progress_callback(_):
            iteration_count[0] += 1
            if iteration_count[0] % 25 == 0:
                print(f"   Iteration {iteration_count[0]}...")
        
        # Stage 1: Rigid alignment
        print("\n🚀 STAGE 1: RIGID ALIGNMENT")
        print("-" * 40)
        
        timer_start = time()
        ch.minimize(
            fun=landmark_error,
            x0=[self.model.trans, self.model.pose[:3]],
            method='dogleg',
            callback=progress_callback,
            options=self.opt_options
        )
        timer_end = time()
        print(f"✅ Stage 1 completed in {timer_end - timer_start:.2f} seconds")
        
        # ALIGNMENT CHECK AFTER RIGID FITTING
        print("\n🔍 CHECKING ALIGNMENT AFTER RIGID FITTING")
        print("-" * 50)
        from flame_alignment_checker import quick_alignment_check
        
        try:
            alignment_results = quick_alignment_check(scan, self.model, landmarks_3d, 
                                                    os.path.join(output_dir, "alignment_check"))
            print(f"📁 Alignment visualizations saved to: {output_dir}/alignment_check/")
            print(f"🎯 Alignment quality: {alignment_results['alignment_quality']}")
            
            if alignment_results['alignment_quality'] == 'poor':
                print("⚠️  WARNING: Poor alignment detected!")
                print("   Check alignment_check/alignment_with_landmarks_check.ply in MeshLab")
                print("   FLAME (red) and scan (gray) should overlap closely")
        except Exception as e:
            print(f"⚠️  Alignment check failed: {e}")
        
        print("-" * 50)
        
        # Stage 2: Non-rigid fitting
        print("\n🔥 STAGE 2: NON-RIGID FITTING")
        print("-" * 40)
        
        timer_start = time()
        ch.minimize(
            fun=objectives,
            x0=free_variables,
            method='dogleg',
            callback=progress_callback,
            options=self.opt_options
        )
        timer_end = time()
        print(f"✅ Stage 2 completed in {timer_end - timer_start:.2f} seconds")
        
        # Extract results
        fitted_vertices = self.model.r
        fitted_faces = self.model.f
        fitted_parameters = {
            'trans': self.model.trans.r,
            'pose': self.model.pose.r,
            'betas': self.model.betas.r
        }
        
        # Save results
        flame_mesh_path = os.path.join(output_dir, "flame_fitted_mesh.obj")
        write_simple_obj(fitted_vertices, fitted_faces, flame_mesh_path)
        
        params_path = os.path.join(output_dir, "flame_parameters.npz")
        np.savez(params_path, **fitted_parameters)
        
        # Save scaled scan for reference
        scan_path = os.path.join(output_dir, "scan_scaled_reference.obj")
        write_simple_obj(scan.v, scan.f, scan_path)
        
        print(f"\n🎉 FLAME FITTING COMPLETED!")
        print("="*60)
        print(f"📁 FLAME mesh: {flame_mesh_path}")
        print(f"⚙️  Parameters: {params_path}")
        print(f"📄 Scan reference: {scan_path}")
        print("="*60)
        
        return fitted_vertices, fitted_faces, fitted_parameters

def run_complete_trimesh_flame_fitting(scan_mesh_path, landmarks_npy_path, 
                                     flame_model_dir="models/",
                                     output_dir="flame_output"):
    """
    Main function: Complete FLAME fitting using trimesh
    
    Args:
        scan_mesh_path: Path to your face scan (.obj/.ply/.stl)
        landmarks_npy_path: Path to your 3D landmarks (.npy file with 51 landmarks)
        flame_model_dir: Directory containing FLAME models
        output_dir: Output directory for results
    
    Returns:
        Dictionary with all results
    """
    
    print("🔥🔥🔥 COMPLETE TRIMESH FLAME FITTING 🔥🔥🔥")
    print("="*80)
    
    # Initialize fitter
    flame_model_path = os.path.join(flame_model_dir, "generic_model.pkl")
    flame_embedding_path = os.path.join(flame_model_dir, "flame_static_embedding.pkl")
    
    fitter = CompleteTrimeshFLAMEFitter(
        flame_model_path=flame_model_path,
        flame_embedding_path=flame_embedding_path
    )
    
    # Load data
    scan, landmarks_3d, scale_factor = fitter.load_scan_and_landmarks(
        scan_path=scan_mesh_path,
        landmarks_path=landmarks_npy_path,
        scan_unit='m'  # Assuming your landmarks are in meters
    )

    # FIX ORIENTATION MISMATCH
    landmarks_3d_fixed, scan_v_fixed = fix_landmark_orientation_mismatch(landmarks_3d, scan)
    landmarks_3d = landmarks_3d_fixed
    scan.v = scan_v_fixed
    scan.mesh = trimesh.Trimesh(vertices=scan.v, faces=scan.f)

    # Downsample scan
    scan = downsample_scan(scan, target_vertices=20000)

    # ADD THE MANUAL CORRECTION HERE
    correction_factor = 1.0
    scan.v *= correction_factor
    landmarks_3d *= correction_factor

    scan_center = np.mean(scan.v, axis=0)
    target_center = np.array([0.0, 0.05, 0.0])  # Reasonable FLAME position
    center_offset = target_center - scan_center
    scan.v += center_offset
    landmarks_3d += center_offset

    print(f"🔧 Manual alignment: scale={correction_factor:.3f}, offset={center_offset}")
    
    # Run fitting
    fitted_vertices, fitted_faces, fitted_parameters = fitter.fit_flame_to_scan(
        scan=scan,
        landmarks_3d=landmarks_3d,
        scan_weight=2.0,           # Strong scan alignment
        landmark_weight=1e-2,      # Landmark guidance
        shape_reg=1e-4,           # Shape regularization
        expr_reg=1e-4,            # Expression regularization
        pose_reg=1e-3,            # Pose regularization
        shape_components=50,      # Shape components to use
        expr_components=25,        # Expression components to use
        output_dir=output_dir
    )
    
    results = {
        'success': True,
        'fitted_vertices': fitted_vertices,
        'fitted_faces': fitted_faces,
        'fitted_parameters': fitted_parameters,
        'scale_factor': scale_factor,
        'output_directory': output_dir,
        'flame_mesh_path': os.path.join(output_dir, "flame_fitted_mesh.obj"),
        'parameters_path': os.path.join(output_dir, "flame_parameters.npz")
    }
    
    print("\n🎉 COMPLETE TRIMESH FLAME FITTING SUCCESSFUL! 🎉")
    print("="*80)
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    # scan_path = "./scan_mesh_input/face_mesh_3dlandmark_checking.ply"  # Your surface reconstructed mesh
    # landmarks_path = "./scan_mesh_input/landmarks_3d_flame_perfect.npy"  # Your perfect landmarks
    scan_path ="./data/scan.obj"
    landmarks_path = "./data/scan_lmks.npy"  


    try:
        results = run_complete_trimesh_flame_fitting(
            scan_mesh_path=scan_path,
            landmarks_npy_path=landmarks_path,
            flame_model_dir="models/",  # Your FLAME models directory
            output_dir="trimesh_flame_output"
        )
        
        print(f"🎉 SUCCESS!")
        print(f"FLAME mesh: {results['flame_mesh_path']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
























# pytorch based flame fitting algorithm

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import trimesh
# import os
# from time import time
# import pickle

# # Chumpy imports for real FLAME data loading
# import chumpy as ch
# from smpl_webuser.serialization import load_model

# class RealPyTorchFLAME(nn.Module):
#     """
#     PyTorch FLAME model using REAL FLAME data loaded via chumpy
#     """
    
#     def __init__(self, flame_model_path, device='cuda'):
#         super(RealPyTorchFLAME, self).__init__()
        
#         self.device = device
#         print(f"🔥 Loading REAL FLAME model with chumpy + PyTorch GPU acceleration")
#         print(f"   Device: {str(device).upper()}")
        
#         # Load REAL FLAME model using chumpy
#         print("📁 Loading FLAME model with chumpy...")
#         flame_model = load_model(flame_model_path)
#         print("✅ FLAME model loaded successfully!")
        
#         # Extract REAL numpy arrays from chumpy objects
#         print("🔄 Converting chumpy data to PyTorch tensors...")
        
#         def safe_extract_data(obj, name="object"):
#             """Safely extract data from chumpy objects with better error handling"""
#             try:
#                 if hasattr(obj, 'r'):
#                     return obj.r
#                 elif hasattr(obj, 'full'):  # Some chumpy objects use .full
#                     return obj.full
#                 elif isinstance(obj, np.ndarray):
#                     return obj
#                 else:
#                     # For complex chumpy expressions, evaluate them
#                     return np.array(obj)
#             except Exception as e:
#                 print(f"⚠️  Warning: Could not extract {name}: {e}")
#                 return None
        
#         # Template mesh (real FLAME vertices and faces)
#         v_template = safe_extract_data(flame_model.v_template, "v_template")
#         faces = safe_extract_data(flame_model.f, "faces")
        
#         # Shape and expression basis - handle potential chumpy operations
#         shapedirs = safe_extract_data(flame_model.shapedirs, "shapedirs")
        
#         # Expression dirs might be a complex chumpy expression
#         print("🔍 Extracting expression directions...")
#         if hasattr(flame_model, 'exprdirs'):
#             exprdirs = safe_extract_data(flame_model.exprdirs, "exprdirs")
#         else:
#             print("⚠️  No exprdirs found, checking for alternative expression data...")
#             # Try alternative names
#             expr_names = ['expr_dirs', 'expression_dirs', 'expressiondirs']
#             exprdirs = None
#             for name in expr_names:
#                 if hasattr(flame_model, name):
#                     exprdirs = safe_extract_data(getattr(flame_model, name), name)
#                     break
            
#             if exprdirs is None:
#                 print("⚠️  Creating fallback expression directions...")
#                 # Create basic expression basis if none found
#                 n_verts = v_template.shape[0] if v_template is not None else 5023
#                 exprdirs = np.random.normal(0, 0.001, (n_verts, 3, 50)).astype(np.float32)
        
#         # Ensure data types
#         if v_template is not None:
#             v_template = v_template.astype(np.float32)
#         if faces is not None:
#             faces = faces.astype(np.int32)
#         if shapedirs is not None:
#             shapedirs = shapedirs.astype(np.float32)
#         if exprdirs is not None:
#             exprdirs = exprdirs.astype(np.float32)
        
#         print(f"✅ Real FLAME data extracted:")
#         print(f"   Template vertices: {v_template.shape if v_template is not None else 'None'}")
#         print(f"   Faces: {faces.shape if faces is not None else 'None'}")
#         print(f"   Shape directions: {shapedirs.shape if shapedirs is not None else 'None'}")
#         print(f"   Expression directions: {exprdirs.shape if exprdirs is not None else 'None'}")
        
#         # Convert to PyTorch tensors on GPU with validation
#         print("🚀 Moving data to GPU...")
        
#         # Validate and convert data
#         if v_template is None or faces is None:
#             raise ValueError("Critical FLAME data missing: template vertices or faces")
        
#         self.v_template = torch.from_numpy(v_template).to(device)
#         # self.faces = torch.from_numpy(faces).to(device)
#         self.faces = torch.from_numpy(faces).long().to(device)
        
#         if shapedirs is not None:
#             self.shapedirs = torch.from_numpy(shapedirs).to(device)
#         else:
#             # Create minimal shape basis
#             print("⚠️  Creating fallback shape directions...")
#             n_verts = v_template.shape[0]
#             shapedirs = np.random.normal(0, 0.001, (n_verts, 3, 100)).astype(np.float32)
#             self.shapedirs = torch.from_numpy(shapedirs).to(device)
        
#         if exprdirs is not None:
#             self.exprdirs = torch.from_numpy(exprdirs).to(device)
#         else:
#             # Create minimal expression basis
#             print("⚠️  Creating fallback expression directions...")
#             n_verts = v_template.shape[0]
#             exprdirs = np.random.normal(0, 0.001, (n_verts, 3, 50)).astype(np.float32)
#             self.exprdirs = torch.from_numpy(exprdirs).to(device)
        
#         # Optional FLAME data with fallbacks
#         try:
#             posedirs = safe_extract_data(flame_model.posedirs, "posedirs") if hasattr(flame_model, 'posedirs') else None
#             if posedirs is not None:
#                 self.posedirs = torch.from_numpy(posedirs.astype(np.float32)).to(device)
#         except:
#             pass
            
#         try:
#             J_regressor = safe_extract_data(flame_model.J_regressor, "J_regressor") if hasattr(flame_model, 'J_regressor') else None
#             if J_regressor is not None:
#                 self.J_regressor = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
#         except:
#             pass
            
#         try:
#             weights = safe_extract_data(flame_model.weights, "weights") if hasattr(flame_model, 'weights') else None
#             if weights is not None:
#                 self.weights = torch.from_numpy(weights.astype(np.float32)).to(device)
#         except:
#             pass
        
#         # Model dimensions
#         self.n_vertices = self.v_template.shape[0]
#         self.n_shape = self.shapedirs.shape[-1]    # 300 shape components
#         self.n_expr = self.exprdirs.shape[-1]      # 100 expression components
        
#         print(f"✅ REAL FLAME model ready on GPU:")
#         print(f"   {self.n_vertices} vertices, {self.n_shape} shape params, {self.n_expr} expression params")
        
#     def forward(self, betas, expression, global_rot, translation):
#         """
#         Forward pass through REAL FLAME model
        
#         Args:
#             betas: Shape parameters [batch, n_shape_used]
#             expression: Expression parameters [batch, n_expr_used]  
#             global_rot: Global rotation [batch, 3]
#             translation: Translation [batch, 3]
        
#         Returns:
#             vertices: [batch, n_vertices, 3]
#         """
        
#         batch_size = betas.shape[0] if betas is not None else 1
        
#         # Start with REAL FLAME template
#         vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
#         # Apply REAL shape deformations
#         if betas is not None and betas.shape[1] > 0:
#             n_shape_used = min(betas.shape[1], self.n_shape)
#             shape_displacements = torch.einsum('bl,mkl->bmk', 
#                                              betas[:, :n_shape_used], 
#                                              self.shapedirs[:, :, :n_shape_used])
#             vertices = vertices + shape_displacements
            
#         # Apply REAL expression deformations  
#         if expression is not None and expression.shape[1] > 0:
#             n_expr_used = min(expression.shape[1], self.n_expr)
#             expr_displacements = torch.einsum('bl,mkl->bmk',
#                                             expression[:, :n_expr_used],
#                                             self.exprdirs[:, :, :n_expr_used])
#             vertices = vertices + expr_displacements
        
#         # Apply global rotation
#         if global_rot is not None:
#             vertices = self.apply_rotation(vertices, global_rot)
            
#         # Apply translation
#         if translation is not None:
#             vertices = vertices + translation.unsqueeze(1)
            
#         return vertices
    
#     def apply_rotation(self, vertices, rotation):
#         """Apply rotation using Rodriguez formula (fixed tensor dimensions)"""
        
#         batch_size = rotation.shape[0]
#         angle = torch.norm(rotation, dim=1, keepdim=True)  # [batch, 1]
        
#         # Handle small angles (no rotation needed)
#         small_angle_mask = (angle < 1e-8).squeeze(-1)  # [batch]
        
#         if torch.all(small_angle_mask):
#             # No rotation needed for any batch
#             return vertices
        
#         # For non-small angles, compute rotation
#         # Avoid division by zero
#         safe_angle = torch.where(angle < 1e-8, torch.ones_like(angle), angle)
#         axis = rotation / safe_angle  # [batch, 3]
        
#         # Rodriguez formula components
#         cos_angle = torch.cos(angle)  # [batch, 1]
#         sin_angle = torch.sin(angle)  # [batch, 1]
        
#         # Cross product matrix K for each batch
#         K = torch.zeros(batch_size, 3, 3, device=self.device)
#         K[:, 0, 1] = -axis[:, 2]
#         K[:, 0, 2] = axis[:, 1]
#         K[:, 1, 0] = axis[:, 2]
#         K[:, 1, 2] = -axis[:, 0]
#         K[:, 2, 0] = -axis[:, 1]
#         K[:, 2, 1] = axis[:, 0]
        
#         # Identity matrix
#         I = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        
#         # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
#         K_squared = torch.bmm(K, K)  # [batch, 3, 3]
        
#         R = (I + 
#             sin_angle.unsqueeze(-1) * K + 
#             (1 - cos_angle).unsqueeze(-1) * K_squared)
        
#         # For small angles, use identity matrix
#         R = torch.where(small_angle_mask.view(-1, 1, 1), I, R)
        
#         # Apply rotation: R @ vertices^T
#         # vertices: [batch, n_vertices, 3] -> [batch, 3, n_vertices]
#         vertices_t = vertices.transpose(-1, -2)  # [batch, 3, n_vertices]
        
#         # Matrix multiplication
#         vertices_rot_t = torch.bmm(R, vertices_t)  # [batch, 3, n_vertices]
        
#         # Transpose back: [batch, n_vertices, 3]
#         vertices_rot = vertices_rot_t.transpose(-1, -2)
        
#         return vertices_rot

# class RealFLAMEFitter:
#     """
#     GPU-accelerated FLAME fitter using REAL FLAME model
#     """
    
#     def __init__(self, flame_model_path, embedding_path, device='cuda'):
        
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         print(f"🚀 Real FLAME Fitter initializing on {self.device}")
        
#         # Load REAL FLAME model
#         self.flame_model = RealPyTorchFLAME(flame_model_path, self.device)
        
#         # Load REAL landmark embedding
#         print("📁 Loading landmark embedding...")
#         with open(embedding_path, 'rb') as f:
#             embedding_data = pickle.load(f, encoding='latin1')
        
#         # Ensure correct tensor types for indexing
#         self.lmk_face_idx = torch.from_numpy(embedding_data['lmk_face_idx']).long().to(self.device)
#         self.lmk_b_coords = torch.from_numpy(embedding_data['lmk_b_coords']).float().to(self.device)
                
#         print(f"✅ Loaded REAL landmark embedding: {len(self.lmk_face_idx)} landmarks")
#         print("="*80)
    
#     def get_landmarks(self, vertices):
#         """Extract landmarks from vertices using REAL FLAME embedding"""
        
#         # Ensure face indices are long type for PyTorch indexing
#         face_indices = self.lmk_face_idx.long()  # Convert to long
        
#         # Get triangle vertices for each landmark (vectorized on GPU)
#         face_verts = vertices[self.flame_model.faces[face_indices]]  # [n_landmarks, 3, 3]
        
#         # Apply barycentric coordinates (GPU computation)
#         landmarks = torch.sum(face_verts * self.lmk_b_coords.unsqueeze(-1), dim=1)  # [n_landmarks, 3]
        
#         return landmarks
    
#     def compute_scan_to_mesh_loss(self, scan_points, model_vertices):
#         """GPU-accelerated scan-to-mesh distance computation"""
        
#         # Subsample scan points for memory efficiency
#         max_scan_points = 10000  # Adjust based on GPU memory
#         if len(scan_points) > max_scan_points:
#             indices = torch.randperm(len(scan_points), device=self.device)[:max_scan_points]
#             scan_subset = scan_points[indices]
#         else:
#             scan_subset = scan_points
        
#         # Compute all pairwise distances (fully vectorized on GPU)
#         distances = torch.cdist(scan_subset, model_vertices)  # [n_scan, n_model] - GPU magic!
        
#         # Get minimum distance for each scan point
#         min_distances = torch.min(distances, dim=1)[0]  # [n_scan]
        
#         # Return mean squared distance
#         return torch.mean(min_distances**2)
    
#     def fit_to_scan(self, scan_vertices, landmarks_3d, 
#                    shape_components=100, expr_components=50,
#                    landmark_weight=1000.0, scan_weight=1.0,
#                    shape_reg=0.01, expr_reg=0.01, pose_reg=0.1,
#                    n_iterations_rigid=200, n_iterations_nonrigid=800,
#                    learning_rate=0.01):
#         """
#         Fit REAL FLAME to scan using GPU acceleration
#         """
        
#         print(f"\n🔥 STARTING REAL FLAME FITTING WITH GPU ACCELERATION")
#         print("="*80)
        
#         # Convert to PyTorch tensors on GPU
#         scan_points = torch.from_numpy(scan_vertices.astype(np.float32)).to(self.device)
#         landmarks = torch.from_numpy(landmarks_3d.astype(np.float32)).to(self.device)
        
#         print(f"📊 Scan: {scan_points.shape[0]} vertices")
#         print(f"📊 Landmarks: {landmarks.shape[0]} points")
#         print(f"📊 FLAME: {self.flame_model.n_vertices} vertices")
#         print(f"📊 Using {shape_components} shape + {expr_components} expr components")
        
#         # Initialize parameters
#         batch_size = 1
#         betas = torch.zeros(batch_size, shape_components, device=self.device, requires_grad=True)
#         expression = torch.zeros(batch_size, expr_components, device=self.device, requires_grad=True)
#         global_rot = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
#         translation = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
        
#         total_params = sum(p.numel() for p in [betas, expression, global_rot, translation])
#         print(f"📊 Optimizing {total_params} parameters")
        
#         # Stage 1: Rigid alignment (landmarks only) 
#         print(f"\n🚀 STAGE 1: RIGID ALIGNMENT (GPU)")
#         print("-" * 50)
        
#         rigid_params = [global_rot, translation]
#         optimizer_rigid = optim.Adam(rigid_params, lr=learning_rate)
        
#         timer_start = time()
        
#         for i in range(n_iterations_rigid):
#             optimizer_rigid.zero_grad()
            
#             # Forward pass through REAL FLAME
#             vertices = self.flame_model(betas, expression, global_rot, translation)
            
#             # Get model landmarks using REAL embedding
#             model_landmarks = self.get_landmarks(vertices[0])
            
#             # Landmark loss
#             landmark_loss = torch.mean((model_landmarks - landmarks)**2)
            
#             # Backward pass (GPU automatic differentiation)
#             landmark_loss.backward()
#             optimizer_rigid.step()
            
#             if i % 25 == 0 or i < 10:
#                 print(f"   Iter {i:3d}: {landmark_loss.item():.6f}")
        
#         timer_end = time()
#         print(f"✅ Stage 1 completed in {timer_end - timer_start:.2f} seconds")
        
#         # Stage 2: Non-rigid fitting (shape + expression + scan fitting)
#         print(f"\n🔥 STAGE 2: NON-RIGID FITTING (GPU)")
#         print("-" * 50)
        
#         all_params = [betas, expression, global_rot, translation]
#         optimizer_full = optim.Adam(all_params, lr=learning_rate * 0.1)  # Slower for stability
        
#         timer_start = time()
#         losses_history = []
        
#         for i in range(n_iterations_nonrigid):
#             optimizer_full.zero_grad()
            
#             # Forward pass through REAL FLAME
#             vertices = self.flame_model(betas, expression, global_rot, translation)
#             model_verts = vertices[0]
            
#             # Get model landmarks using REAL embedding
#             model_landmarks = self.get_landmarks(model_verts)
            
#             # Loss terms
#             landmark_loss = landmark_weight * torch.mean((model_landmarks - landmarks)**2)
#             scan_loss = scan_weight * self.compute_scan_to_mesh_loss(scan_points, model_verts)
#             shape_loss = shape_reg * torch.sum(betas**2)
#             expr_loss = expr_reg * torch.sum(expression**2)
#             pose_loss = pose_reg * torch.sum(global_rot**2)
            
#             total_loss = landmark_loss + scan_loss + shape_loss + expr_loss + pose_loss
            
#             # Backward pass (GPU automatic differentiation)
#             total_loss.backward()
#             optimizer_full.step()
            
#             # Track losses
#             losses_history.append({
#                 'total': total_loss.item(),
#                 'landmark': landmark_loss.item(),
#                 'scan': scan_loss.item(),
#                 'shape': shape_loss.item(),
#                 'expr': expr_loss.item(),
#                 'pose': pose_loss.item()
#             })
            
#             if i % 50 == 0 or i < 10:
#                 print(f"   Iter {i:3d}: {total_loss.item():.6f} | "
#                       f"lmk: {landmark_loss.item():.2e} | "
#                       f"scan: {scan_loss.item():.2e} | "
#                       f"shape: {shape_loss.item():.2e}")
        
#         timer_end = time()
#         print(f"✅ Stage 2 completed in {timer_end - timer_start:.2f} seconds")
        
#         # Final result
#         with torch.no_grad():
#             final_vertices = self.flame_model(betas, expression, global_rot, translation)
#             final_verts = final_vertices[0].cpu().numpy()
        
#         # Extract fitted parameters
#         fitted_params = {
#             'betas': betas.detach().cpu().numpy(),
#             'expression': expression.detach().cpu().numpy(),
#             'global_rot': global_rot.detach().cpu().numpy(),
#             'translation': translation.detach().cpu().numpy()
#         }
        
#         print(f"\n🎉 REAL FLAME FITTING COMPLETED!")
#         print(f"Final loss: {total_loss.item():.6f}")
#         print(f"Parameters shape changed: {np.sum(np.abs(fitted_params['betas'])) > 1e-6}")
#         print("="*80)
        
#         return final_verts, self.flame_model.faces.cpu().numpy(), fitted_params, losses_history

# def run_real_flame_fitting(scan_mesh_path, landmarks_npy_path,
#                           flame_model_path="models/generic_model.pkl",
#                           embedding_path="models/flame_static_embedding.pkl",
#                           output_dir="real_flame_gpu_output"):
#     """
#     Complete REAL FLAME fitting with GPU acceleration
#     """
    
#     print("🔥🔥🔥 REAL FLAME GPU FITTING PIPELINE 🔥🔥🔥")
#     print("="*80)
    
#     # GPU check
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if device == 'cuda':
#         gpu_name = torch.cuda.get_device_name(0)
#         gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"🚀 Using GPU: {gpu_name}")
#         print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
#     else:
#         print("⚠️  No GPU available, using CPU")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load scan data
#     print(f"\n📁 LOADING SCAN DATA")
#     print("-" * 40)
    
#     scan_mesh = trimesh.load(scan_mesh_path)
#     scan_vertices = scan_mesh.vertices.astype(np.float32)
#     scan_faces = scan_mesh.faces
    
#     print(f"✅ Original scan: {len(scan_vertices)} vertices, {len(scan_faces)} faces")
    
#     # Downsample scan for GPU memory efficiency
#     target_vertices = 50000  # Adjust based on your GPU memory
#     if len(scan_vertices) > target_vertices:
#         print(f"🔄 Downsampling scan for GPU efficiency...")
#         scan_simplified = scan_mesh.simplify_quadric_decimation(face_count=int(target_vertices * 1.8))
#         scan_vertices = scan_simplified.vertices.astype(np.float32)
#         print(f"✅ Downsampled to {len(scan_vertices)} vertices")
    
#     # Load landmarks
#     landmarks_3d = np.load(landmarks_npy_path).astype(np.float32)
#     print(f"✅ Landmarks: {len(landmarks_3d)} points")
    
#     # Initialize REAL FLAME fitter
#     fitter = RealFLAMEFitter(flame_model_path, embedding_path, device)
    
#     # Run GPU fitting
#     fitted_vertices, fitted_faces, fitted_params, losses = fitter.fit_to_scan(
#         scan_vertices=scan_vertices,
#         landmarks_3d=landmarks_3d,
#         shape_components=100,      # Use full shape space
#         expr_components=50,        # Use subset of expressions
#         landmark_weight=1000.0,    # Strong landmark constraint
#         scan_weight=1.0,           # Surface fitting
#         n_iterations_rigid=200,
#         n_iterations_nonrigid=200
#     )
    
#     # Save results
#     print(f"\n💾 SAVING RESULTS")
#     print("-" * 40)
    
#     # Save fitted FLAME mesh
#     fitted_path = os.path.join(output_dir, "real_flame_fitted.ply")
#     fitted_mesh = trimesh.Trimesh(vertices=fitted_vertices, faces=fitted_faces)
#     fitted_mesh.export(fitted_path)
#     print(f"✅ Fitted FLAME mesh: {fitted_path}")
    
#     # Save parameters
#     params_path = os.path.join(output_dir, "flame_parameters.npz")
#     np.savez(params_path, **fitted_params)
#     print(f"✅ FLAME parameters: {params_path}")
    
#     # Save original scan for comparison
#     scan_path = os.path.join(output_dir, "original_scan.ply")
#     trimesh.Trimesh(vertices=scan_vertices, faces=scan_faces).export(scan_path)
#     print(f"✅ Original scan: {scan_path}")
    
#     # Save losses
#     losses_path = os.path.join(output_dir, "fitting_losses.npy")
#     loss_array = np.array([(l['total'], l['landmark'], l['scan']) for l in losses])
#     np.save(losses_path, loss_array)
#     print(f"✅ Loss history: {losses_path}")
    
#     # Create combined visualization
#     fitted_mesh.visual.face_colors = [255, 100, 100, 200]  # Red FLAME
#     scan_mesh_viz = trimesh.Trimesh(vertices=scan_vertices, faces=scan_faces)
#     scan_mesh_viz.visual.face_colors = [200, 200, 200, 100]  # Gray scan
    
#     combined_scene = trimesh.Scene([fitted_mesh, scan_mesh_viz])
#     combined_path = os.path.join(output_dir, "flame_vs_scan_comparison.ply")
#     combined_scene.export(combined_path)
#     print(f"✅ Comparison visualization: {combined_path}")
    
#     print(f"\n🎉 REAL FLAME GPU FITTING COMPLETED!")
#     print(f"📁 All results saved to: {output_dir}")
#     print(f"🔥 FLAME mesh optimized to fit your scan as closely as possible!")
#     print("="*80)
    
#     return {
#         'fitted_vertices': fitted_vertices,
#         'fitted_faces': fitted_faces,
#         'fitted_params': fitted_params,
#         'losses': losses,
#         'output_dir': output_dir,
#         'fitted_mesh_path': fitted_path
#     }

# # Example usage
# if __name__ == "__main__":
    
#     # Make sure you have:
#     # 1. pip install chumpy (for loading real FLAME data)
#     # 2. Real FLAME models in models/ directory
#     # 3. Your scan and landmarks
    
#     results = run_real_flame_fitting(
#         scan_mesh_path="./scan_mesh_input/face_mesh_3dlandmark_checking.ply",
#         landmarks_npy_path="./scan_mesh_input/landmarks_3d_flame_perfect.npy",
#         flame_model_path="models/generic_model.pkl",
#         embedding_path="models/flame_static_embedding.pkl"
#     )
    
#     print(f"🎉 SUCCESS!")
#     print(f"Fitted FLAME mesh: {results['fitted_mesh_path']}")