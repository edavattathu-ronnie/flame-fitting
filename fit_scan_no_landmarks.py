# import numpy as np
# import chumpy as ch
# from os.path import join

# from psbody.mesh import Mesh
# from smpl_webuser.serialization import load_model
# from sbody.mesh_distance import ScanToMesh
# from sbody.robustifiers import GMOf
# from sbody.alignment.objectives import sample_from_mesh
# from fitting.util import write_simple_obj, safe_mkdir, get_unit_factor

# def fit_scan_surface_only(scan, model, weights, gmo_sigma, 
#                          shape_num=300, expr_num=100, opt_options=None):
#     """
#     Modified FLAME fitting that skips landmark alignment and goes directly to surface fitting
#     Use this when you've already manually aligned your scan and FLAME template
    
#     Args:
#         scan: Pre-aligned scan mesh (psbody.mesh.Mesh)
#         model: FLAME model (loaded with load_model)
#         weights: Dictionary of weights for different objectives
#         gmo_sigma: Robustifier parameter
#         shape_num: Number of shape components to use (0-299)
#         expr_num: Number of expression components to use (0-99)
#         opt_options: Optimization options
    
#     Returns:
#         model.r: Fitted vertices
#         model.f: Fitted faces
#         params: Fitted parameters
#     """
    
#     print("=== FLAME FITTING (SURFACE-ONLY) ===")
#     print("Skipping landmark alignment - using pre-aligned meshes")
    
#     # Variables for optimization
#     shape_idx = np.arange(0, min(300, shape_num))
#     expr_idx = np.arange(300, 300 + min(100, expr_num))
#     used_idx = np.union1d(shape_idx, expr_idx)
    
#     # Initialize parameters
#     model.betas[:] = np.zeros(model.betas.size)  # Start with neutral shape/expression
#     model.pose[:] = np.zeros(model.pose.size)    # Start with neutral pose
    
#     # Variables to optimize (excluding landmarks since we skip rigid alignment)
#     free_variables = [model.betas[used_idx]]  # Only optimize shape/expression initially
    
#     print("fit_scan_surface_only(): using the following weights:")
#     for key, value in weights.items():
#         print(f"  weights['{key}'] = {value}")
    
#     # === OBJECTIVES ===
    
#     # 1. Scan-to-mesh distance (main fitting objective)
#     sampler = sample_from_mesh(scan, sample_type='vertices')
#     s2m = ScanToMesh(scan, model, model.f, scan_sampler=sampler, 
#                      rho=lambda x: GMOf(x, sigma=gmo_sigma))
    
#     # 2. Regularizers
#     shape_err = weights.get('shape', 1e-4) * model.betas[shape_idx]
#     expr_err = weights.get('expr', 1e-4) * model.betas[expr_idx]
#     pose_err = weights.get('pose', 1e-3) * model.pose[3:]  # Exclude global rotation
    
#     # Combine objectives (no landmark term)
#     objectives = {
#         's2m': weights.get('s2m', 2.0) * s2m,
#         'shape': shape_err,
#         'expr': expr_err,
#         'pose': pose_err
#     }
    
#     # Optimization options
#     if opt_options is None:
#         print("Using default optimization options")
#         import scipy.sparse as sp
#         opt_options = {
#             'disp': 1,
#             'delta_0': 0.1,
#             'e_3': 1e-4,
#             'maxiter': 2000,
#             'sparse_solver': lambda A, x: sp.linalg.cg(A, x, maxiter=2000)[0]
#         }
    
#     def on_step(_):
#         pass
    
#     # === OPTIMIZATION STAGES ===
    
#     # Stage 1: Shape and expression only (keep pose fixed)
#     print("\nStage 1: Optimizing shape and expression (pose fixed)...")
#     from time import time
#     timer_start = time()
    
#     shape_expr_objectives = {
#         's2m': objectives['s2m'],
#         'shape': objectives['shape'], 
#         'expr': objectives['expr']
#     }
    
#     ch.minimize(
#         fun=shape_expr_objectives,
#         x0=[model.betas[used_idx]],
#         method='dogleg',
#         callback=on_step,
#         options=opt_options
#     )
    
#     timer_end = time()
#     print(f"Stage 1 completed in {timer_end - timer_start:.2f} seconds")
    
#     # Stage 2: Add pose optimization (if translation/rotation might need refinement)
#     print("\nStage 2: Adding pose optimization...")
#     timer_start = time()
    
#     # Now include pose in optimization
#     free_variables_full = [model.trans, model.pose, model.betas[used_idx]]
    
#     ch.minimize(
#         fun=objectives,
#         x0=free_variables_full,
#         method='dogleg', 
#         callback=on_step,
#         options=opt_options
#     )
    
#     timer_end = time()
#     print(f"Stage 2 completed in {timer_end - timer_start:.2f} seconds")
    
#     # Extract final parameters
#     fitted_params = {
#         'trans': model.trans.r,
#         'pose': model.pose.r,
#         'betas': model.betas.r
#     }
    
#     # Print fitting summary
#     print(f"\n=== FITTING SUMMARY ===")
#     print(f"Shape components used: {np.count_nonzero(fitted_params['betas'][:300])}/300")
#     print(f"Expression components used: {np.count_nonzero(fitted_params['betas'][300:400])}/100")
#     print(f"Final translation: {fitted_params['trans']}")
#     print(f"Pose parameters: {len(fitted_params['pose'])} values")
    
#     return model.r, model.f, fitted_params


# def run_surface_only_fitting():
#     """
#     Main function to run FLAME fitting without landmarks
#     """
    
#     # === INPUT FILES ===
#     scan_path = './face_realistically_scaled_new.ply'  # Your pre-aligned scan
#     model_path = './models/generic_model.pkl'
#     output_dir = './output_surface_only'
    
#     # Load pre-aligned scan
#     scan = Mesh(filename=scan_path)
#     print(f"Loaded pre-aligned scan from: {scan_path}")
#     print(f"Scan vertices: {len(scan.v)}, faces: {len(scan.f)}")
    
#     # Load FLAME model
#     model = load_model(model_path)
#     print(f"Loaded FLAME model from: {model_path}")
    
#     # Create output directory
#     safe_mkdir(output_dir)
    
#     # === FITTING WEIGHTS ===
#     # Since we're not using landmarks, increase surface-to-mesh weight
#     weights = {
#         's2m': 5.0,      # Higher weight for surface fitting (was 2.0 with landmarks)
#         'shape': 1e-3,   # Shape regularization 
#         'expr': 1e-3,    # Expression regularization
#         'pose': 1e-2     # Pose regularization (reduce if alignment is very good)
#     }
    
#     # Robustifier parameter
#     gmo_sigma = 1e-4
    
#     # Optimization options
#     opt_options = {
#         'disp': 1,
#         'delta_0': 0.1,
#         'e_3': 1e-4,
#         'maxiter': 3000,  # More iterations since we're relying only on surface fitting
#     }
    
#     import scipy.sparse as sp
#     opt_options['sparse_solver'] = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    
#     # === RUN FITTING ===
#     print(f"\nStarting surface-only FLAME fitting...")
#     print(f"Weights: {weights}")
    
#     mesh_vertices, mesh_faces, fitted_params = fit_scan_surface_only(
#         scan=scan,
#         model=model,
#         weights=weights,
#         gmo_sigma=gmo_sigma,
#         shape_num=300,
#         expr_num=100,
#         opt_options=opt_options
#     )
    
#     # === SAVE RESULTS ===
    
#     # Save fitted mesh
#     output_mesh_path = join(output_dir, 'fitted_flame_surface_only.obj')
#     write_simple_obj(mesh_v=mesh_vertices, mesh_f=mesh_faces, 
#                      filepath=output_mesh_path, verbose=False)
#     print(f'Fitted FLAME model saved to: {output_mesh_path}')
    
#     # Save original aligned scan for comparison
#     scan_copy_path = join(output_dir, 'original_aligned_scan.obj')
#     write_simple_obj(mesh_v=scan.v, mesh_f=scan.f, 
#                      filepath=scan_copy_path, verbose=False)
#     print(f'Original scan saved to: {scan_copy_path}')
    
#     # Save parameters as numpy file
#     params_path = join(output_dir, 'fitted_parameters.npz')
#     np.savez(params_path, **fitted_params)
#     print(f'Parameters saved to: {params_path}')
    
#     print(f"\n=== FITTING COMPLETED ===")
#     print(f"Check {output_dir} for results")
#     print("Load both the fitted model and original scan in MeshLab to compare")
    
#     return mesh_vertices, mesh_faces, fitted_params


# def quick_surface_fit(scan_path, model_path, output_path, weights=None):
#     """
#     Quick function for surface-only fitting with minimal setup
    
#     Args:
#         scan_path: Path to pre-aligned scan
#         model_path: Path to FLAME model 
#         output_path: Where to save fitted result
#         weights: Optional weight dictionary
#     """
    
#     if weights is None:
#         weights = {'s2m': 5.0, 'shape': 1e-3, 'expr': 1e-3, 'pose': 1e-2}
    
#     # Load data
#     scan = Mesh(filename=scan_path)
#     model = load_model(model_path)
    
#     # Fit
#     vertices, faces, params = fit_scan_surface_only(
#         scan=scan,
#         model=model, 
#         weights=weights,
#         gmo_sigma=1e-4
#     )
    
#     # Save
#     write_simple_obj(mesh_v=vertices, mesh_f=faces, filepath=output_path)
#     print(f"Quick fit saved to: {output_path}")
    
#     return vertices, faces, params


# if __name__ == '__main__':
#     run_surface_only_fitting()




import numpy as np
import chumpy as ch
import trimesh
from pathlib import Path

# Modern replacement for psbody.mesh.Mesh
class ModernMesh:
    """Drop-in replacement for psbody.mesh.Mesh using trimesh"""
    
    def __init__(self, filename=None, v=None, f=None):
        if filename is not None:
            self.mesh = trimesh.load(filename)
            self.v = self.mesh.vertices
            self.f = self.mesh.faces
        elif v is not None and f is not None:
            self.v = v
            self.f = f
            self.mesh = trimesh.Trimesh(vertices=v, faces=f)
        else:
            raise ValueError("Must provide either filename or vertices/faces")
    
    def write_obj(self, filename):
        """Save mesh as OBJ file"""
        self.mesh.export(filename)

# Modern replacement for write_simple_obj
def write_simple_obj(mesh_v, mesh_f, filepath, verbose=True):
    """Write mesh to OBJ file using trimesh"""
    mesh = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f)
    mesh.export(filepath)
    if verbose:
        print(f"Saved mesh to: {filepath}")

# Safe mkdir function
def safe_mkdir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

# Modern mesh distance calculation
class ModernScanToMesh:
    """Modern replacement for ScanToMesh using trimesh"""
    
    def __init__(self, scan_mesh, model_mesh, model_faces, scan_sampler=None, rho=None):
        # Convert scan mesh to numpy if needed
        if hasattr(scan_mesh, 'v'):
            self.scan_vertices = scan_mesh.v
        else:
            self.scan_vertices = scan_mesh.vertices
            
        # Convert model mesh to numpy if needed
        if hasattr(model_mesh, 'r'):
            self.model_vertices = model_mesh.r
        elif hasattr(model_mesh, 'vertices'):
            self.model_vertices = model_mesh.vertices
        else:
            self.model_vertices = model_mesh
            
        self.model_faces = model_faces
        self.rho = rho
        
        # Create trimesh object for distance calculations
        self.model_trimesh = trimesh.Trimesh(vertices=self.model_vertices, faces=self.model_faces)
    
    def __call__(self):
        """Compute scan-to-mesh distance"""
        # Compute closest points on mesh surface
        closest_points, distances, triangle_ids = self.model_trimesh.nearest.on_surface(self.scan_vertices)
        
        # Apply robustifier if provided
        if self.rho is not None:
            distances = self.rho(distances)
        
        return ch.array(distances)

# Function to sample from mesh (replacement for sbody function)
def sample_from_mesh(mesh, sample_type='vertices'):
    """Simple mesh sampling - just return vertices for now"""
    if hasattr(mesh, 'v'):
        return mesh.v
    else:
        return mesh.vertices

# Modified FLAME fitting function
def fit_scan_modern(scan, model, weights, gmo_sigma, 
                   shape_num=300, expr_num=100, opt_options=None):
    """
    FLAME fitting using modern mesh libraries (no psbody dependency)
    """
    
    print("=== MODERN FLAME FITTING (NO LANDMARKS) ===")
    
    # Variables for optimization
    shape_idx = np.arange(0, min(300, shape_num))
    expr_idx = np.arange(300, 300 + min(100, expr_num))
    used_idx = np.union1d(shape_idx, expr_idx)
    
    # Initialize parameters
    model.betas[:] = np.zeros(model.betas.size)
    model.pose[:] = np.zeros(model.pose.size)
    
    print("Using weights:", weights)
    
    # === OBJECTIVES ===
    
    # Scan-to-mesh distance using modern implementation
    s2m = ModernScanToMesh(scan, model, model.f, 
                          rho=lambda x: x**2 * gmo_sigma)  # Simple quadratic robustifier
    
    # Regularizers
    shape_err = weights.get('shape', 1e-4) * model.betas[shape_idx]
    expr_err = weights.get('expr', 1e-4) * model.betas[expr_idx]
    pose_err = weights.get('pose', 1e-3) * model.pose[3:]
    
    # Combine objectives
    objectives = {
        's2m': weights.get('s2m', 5.0) * s2m,
        'shape': shape_err,
        'expr': expr_err,
        'pose': pose_err
    }
    
    # Optimization options
    if opt_options is None:
        import scipy.sparse as sp
        opt_options = {
            'disp': 1,
            'delta_0': 0.1,
            'e_3': 1e-4,
            'maxiter': 2000,
        }
        opt_options['sparse_solver'] = lambda A, x: sp.linalg.cg(A, x, maxiter=2000)[0]
    
    def on_step(_):
        pass
    
    # === OPTIMIZATION ===
    
    print("Stage 1: Optimizing shape and expression...")
    from time import time
    timer_start = time()
    
    # Stage 1: Shape and expression only
    shape_expr_objectives = {
        's2m': objectives['s2m'],
        'shape': objectives['shape'],
        'expr': objectives['expr']
    }
    
    ch.minimize(
        fun=shape_expr_objectives,
        x0=[model.betas[used_idx]],
        method='dogleg',
        callback=on_step,
        options=opt_options
    )
    
    timer_end = time()
    print(f"Stage 1 completed in {timer_end - timer_start:.2f} seconds")
    
    print("Stage 2: Adding pose optimization...")
    timer_start = time()
    
    # Stage 2: Add pose
    free_variables = [model.trans, model.pose, model.betas[used_idx]]
    
    ch.minimize(
        fun=objectives,
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        options=opt_options
    )
    
    timer_end = time()
    print(f"Stage 2 completed in {timer_end - timer_start:.2f} seconds")
    
    # Extract results
    fitted_params = {
        'trans': model.trans.r,
        'pose': model.pose.r,
        'betas': model.betas.r
    }
    
    print(f"Shape components used: {np.count_nonzero(fitted_params['betas'][:300])}/300")
    print(f"Expression components used: {np.count_nonzero(fitted_params['betas'][300:400])}/100")
    
    return model.r, model.f, fitted_params

# Main function
def run_modern_flame_fitting():
    """Run FLAME fitting with modern dependencies"""
    
    # Paths
    scan_path = './face_realistically_scaled_new.ply'  # Your aligned scan
    model_path = './models/flame_static_embedding.pkl'
    output_dir = './output_modern'
    
    # Load scan using modern mesh loader
    scan = ModernMesh(filename=scan_path)
    print(f"Loaded scan: {len(scan.v)} vertices")
    
    # Load FLAME model (this still needs the original loader)
    try:
        from smpl_webuser.serialization import load_model
        model = load_model(model_path)
        print(f"Loaded FLAME model from: {model_path}")
    except ImportError:
        print("Error: FLAME model loader not available")
        print("You may need to install chumpy and download FLAME models")
        return
    
    # Create output directory
    safe_mkdir(output_dir)
    
    # Fitting weights
    weights = {
        's2m': 5.0,      # Surface fitting weight
        'shape': 1e-3,   # Shape regularization
        'expr': 1e-3,    # Expression regularization  
        'pose': 1e-2     # Pose regularization
    }
    
    # Run fitting
    try:
        mesh_v, mesh_f, params = fit_scan_modern(
            scan=scan,
            model=model,
            weights=weights,
            gmo_sigma=1e-4,
            shape_num=300,
            expr_num=100
        )
        
        # Save results
        output_path = Path(output_dir) / 'fitted_flame_modern.obj'
        write_simple_obj(mesh_v, mesh_f, str(output_path))
        
        # Save parameters
        params_path = Path(output_dir) / 'fitted_params.npz'
        np.savez(str(params_path), **params)
        
        print(f"FLAME fitting completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_modern_flame_fitting()