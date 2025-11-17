# #!/usr/bin/env python3
# """
# FLAME PLY Mesh Deformation - Simplified Script

# This script is specifically designed for deforming rigidly aligned FLAME templates
# to fit raw face scans, both provided as PLY files.

# Usage:
#     python flame_ply_deformation.py --template flame_aligned.ply --scan raw_scan.ply --output fitted.ply
# """

# import argparse
# import os
# import sys
# import numpy as np
# import torch
# from pytorch3d.io import load_ply, save_ply
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.loss import (
#     chamfer_distance, 
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
# )
# from tqdm import tqdm
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class FLAMEPLYDeformation:
#     """Simplified FLAME deformation for PLY files"""
    
#     def __init__(self, device=None, force_cpu=False):
#         if force_cpu:
#             self.device = torch.device("cpu")
#             logger.info("Forcing CPU mode due to PyTorch3D CUDA issues")
#         elif device:
#             self.device = device
#         else:
#             # Test PyTorch3D CUDA compatibility
#             if torch.cuda.is_available():
#                 try:
#                     # Quick test to see if PyTorch3D CUDA works
#                     from pytorch3d.utils import ico_sphere
#                     test_mesh = ico_sphere(2, torch.device("cuda"))
#                     from pytorch3d.ops import sample_points_from_meshes
#                     test_points = sample_points_from_meshes(test_mesh, 100)
#                     self.device = torch.device("cuda")
#                     logger.info("PyTorch3D CUDA test passed - using GPU")
#                 except Exception as e:
#                     logger.warning(f"PyTorch3D CUDA test failed: {str(e)}")
#                     logger.info("Falling back to CPU mode")
#                     self.device = torch.device("cpu")
#             else:
#                 self.device = torch.device("cpu")
        
#         logger.info(f"Using device: {self.device}")
    
#     def load_ply_mesh(self, ply_path):
#         """Load PLY mesh file"""
#         if not os.path.exists(ply_path):
#             raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
#         verts, faces = load_ply(ply_path)
        
#         if faces is None:
#             raise ValueError(f"PLY file {ply_path} contains no faces - point cloud only!")
        
#         verts = verts.to(self.device)
#         faces = faces.to(self.device)
        
#         logger.info(f"Loaded PLY: {verts.shape[0]} vertices, {faces.shape[0]} faces")
#         return Meshes(verts=[verts], faces=[faces])
    
#     def normalize_mesh(self, mesh):
#         """Normalize mesh to unit scale"""
#         verts = mesh.verts_packed()
#         center = verts.mean(0)
#         scale = verts.abs().max()
#         normalized_verts = (verts - center) / scale
        
#         # Store for denormalization
#         self.center = center
#         self.scale = scale
        
#         return mesh.offset_verts(normalized_verts - verts)
    
#     def denormalize_verts(self, verts):
#         """Convert normalized vertices back to original scale"""
#         return verts * self.scale + self.center
    
#     def compute_losses(self, deformed_mesh, target_mesh, weights, num_samples=5000):
#         """Compute all loss terms"""
#         # Sample points for chamfer distance
#         points_deformed = sample_points_from_meshes(deformed_mesh, num_samples)
#         points_target = sample_points_from_meshes(target_mesh, num_samples)
        
#         # Data loss: chamfer distance
#         loss_chamfer, _ = chamfer_distance(points_deformed, points_target)
        
#         # Regularization losses
#         loss_edge = mesh_edge_loss(deformed_mesh)
#         loss_laplacian = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
#         loss_normal = mesh_normal_consistency(deformed_mesh)
        
#         # Total weighted loss
#         total_loss = (
#             weights['chamfer'] * loss_chamfer +
#             weights['edge'] * loss_edge +
#             weights['laplacian'] * loss_laplacian +
#             weights['normal'] * loss_normal
#         )
        
#         return {
#             'total': total_loss,
#             'chamfer': loss_chamfer,
#             'edge': loss_edge,
#             'laplacian': loss_laplacian,
#             'normal': loss_normal
#         }
    
#     def fit_mesh(self, template_mesh, scan_mesh, iterations=2000, lr=1.0, weights=None):
#         """Main fitting function"""
#         # Default weights optimized for partial face scans
#         if weights is None:
#             weights = {
#                 'chamfer': 1.0,      # Data fitting
#                 'edge': 1.0,         # Edge regularization
#                 'laplacian': 15.0,   # Strong smoothing for missing regions
#                 'normal': 0.01       # Normal consistency
#             }
        
#         logger.info(f"Starting optimization with {iterations} iterations")
#         logger.info(f"Loss weights: {weights}")
        
#         # Normalize meshes to unit scale
#         template_normalized = self.normalize_mesh(template_mesh)
#         scan_normalized = self.normalize_mesh(scan_mesh)
        
#         # Initialize vertex deformation parameters
#         deform_verts = torch.zeros_like(
#             template_normalized.verts_packed(),
#             requires_grad=True
#         )
        
#         # Optimizer with learning rate scheduling
#         optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.9)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        
#         # Loss tracking
#         losses_history = []
        
#         # Optimization loop
#         pbar = tqdm(range(iterations), desc="Fitting")
        
#         for i in pbar:
#             optimizer.zero_grad()
            
#             # Apply deformation to template
#             deformed_mesh = template_normalized.offset_verts(deform_verts)
            
#             # Compute losses
#             losses = self.compute_losses(deformed_mesh, scan_normalized, weights)
            
#             # Backward pass
#             losses['total'].backward()
#             optimizer.step()
#             scheduler.step()
            
#             # Log progress
#             losses_history.append({k: float(v.detach().cpu()) for k, v in losses.items()})
            
#             if i % 100 == 0:
#                 pbar.set_postfix({
#                     'Total': f"{losses['total']:.6f}",
#                     'Chamfer': f"{losses['chamfer']:.6f}",
#                     'Laplacian': f"{losses['laplacian']:.6f}"
#                 })
        
#         # Final deformed mesh
#         final_mesh = template_normalized.offset_verts(deform_verts)
        
#         # Denormalize back to original scale
#         final_verts, final_faces = final_mesh.get_mesh_verts_faces(0)
#         final_verts_denorm = self.denormalize_verts(final_verts)
#         final_mesh_denorm = Meshes(verts=[final_verts_denorm], faces=[final_faces])
        
#         logger.info(f"Optimization completed! Final loss: {losses['total']:.6f}")
        
#         return final_mesh_denorm, losses_history
    
#     def save_ply_mesh(self, mesh, output_path):
#         """Save mesh as PLY file"""
#         verts, faces = mesh.get_mesh_verts_faces(0)
#         save_ply(output_path, verts, faces)
#         logger.info(f"Saved fitted mesh to: {output_path}")


# def main():
#     parser = argparse.ArgumentParser(description='FLAME PLY Mesh Deformation')
#     parser.add_argument('--template', required=True, 
#                        help='Path to rigidly aligned FLAME template PLY file')
#     parser.add_argument('--scan', required=True,
#                        help='Path to raw face scan PLY file')
#     parser.add_argument('--output', required=True,
#                        help='Path to save fitted mesh PLY file')
#     parser.add_argument('--iterations', type=int, default=2000,
#                        help='Number of optimization iterations (default: 2000)')
#     parser.add_argument('--lr', type=float, default=1.0,
#                        help='Learning rate (default: 1.0)')
#     parser.add_argument('--w_chamfer', type=float, default=1.0,
#                        help='Chamfer loss weight (default: 1.0)')
#     parser.add_argument('--w_edge', type=float, default=1.0,
#                        help='Edge loss weight (default: 1.0)')
#     parser.add_argument('--w_laplacian', type=float, default=15.0,
#                        help='Laplacian loss weight (default: 15.0)')
#     parser.add_argument('--w_normal', type=float, default=0.01,
#                        help='Normal loss weight (default: 0.01)')
#     parser.add_argument('--cpu', action='store_true',
#                        help='Force CPU mode (use if PyTorch3D CUDA issues)')
    
#     args = parser.parse_args()
    
#     try:
#         # Initialize deformation system
#         deformer = FLAMEPLYDeformation(force_cpu=args.cpu)
        
#         # Load PLY files
#         logger.info("Loading PLY files...")
#         template_mesh = deformer.load_ply_mesh(args.template)
#         scan_mesh = deformer.load_ply_mesh(args.scan)
        
#         # Set loss weights
#         weights = {
#             'chamfer': args.w_chamfer,
#             'edge': args.w_edge,
#             'laplacian': args.w_laplacian,
#             'normal': args.w_normal
#         }
        
#         # Fit mesh
#         fitted_mesh, loss_history = deformer.fit_mesh(
#             template_mesh, 
#             scan_mesh,
#             iterations=args.iterations,
#             lr=args.lr,
#             weights=weights
#         )
        
#         # Save result
#         deformer.save_ply_mesh(fitted_mesh, args.output)
        
#         # Print summary
#         initial_loss = loss_history[0]['total']
#         final_loss = loss_history[-1]['total']
#         improvement = (initial_loss - final_loss) / initial_loss * 100
        
#         logger.info(f"Fitting summary:")
#         logger.info(f"  Initial loss: {initial_loss:.6f}")
#         logger.info(f"  Final loss: {final_loss:.6f}")
#         logger.info(f"  Improvement: {improvement:.2f}%")
#         logger.info(f"  Output saved: {args.output}")
        
#     except Exception as e:
#         logger.error(f"Error during mesh fitting: {str(e)}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()






#!/usr/bin/env python3
"""
Region-Aware FLAME PLY Mesh Deformation

This script only deforms the template mesh in regions where the scan has data,
keeping the original template shape in missing/hole regions.

Usage:
    python flame_region_aware_deformation.py --template template.ply --scan scan.ply --output fitted.ply
"""

import argparse
import os
import sys
import numpy as np
import torch
from pytorch3d.io import load_ply, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionAwareFLAMEDeformation:
    """Region-aware FLAME deformation - only fits where scan data exists"""
    
    def __init__(self, device=None, force_cpu=False):
        if force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forcing CPU mode")
        elif device:
            self.device = device
        else:
            # Test PyTorch3D CUDA compatibility
            if torch.cuda.is_available():
                try:
                    from pytorch3d.utils import ico_sphere
                    test_mesh = ico_sphere(2, torch.device("cuda"))
                    from pytorch3d.ops import sample_points_from_meshes
                    test_points = sample_points_from_meshes(test_mesh, 100)
                    self.device = torch.device("cuda")
                    logger.info("PyTorch3D CUDA test passed - using GPU")
                except Exception as e:
                    logger.warning(f"PyTorch3D CUDA test failed: {str(e)}")
                    logger.info("Falling back to CPU mode")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def load_ply_mesh(self, ply_path):
        """Load PLY mesh file"""
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        verts, faces = load_ply(ply_path)
        if faces is None:
            raise ValueError(f"PLY file {ply_path} contains no faces - point cloud only!")
        
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        
        logger.info(f"Loaded PLY: {verts.shape[0]} vertices, {faces.shape[0]} faces")
        return Meshes(verts=[verts], faces=[faces])
    
    def find_scan_coverage_mask(self, template_mesh, scan_mesh, distance_threshold=0.05):
        """
        Find which parts of the template have corresponding scan data nearby
        
        Args:
            template_mesh: Template mesh
            scan_mesh: Scan mesh  
            distance_threshold: Maximum distance to consider as "covered"
            
        Returns:
            torch.Tensor: Boolean mask indicating which template vertices have scan coverage
        """
        logger.info(f"Computing scan coverage mask with threshold {distance_threshold}")
        
        # Sample points from both meshes
        template_points = sample_points_from_meshes(template_mesh, 10000)
        scan_points = sample_points_from_meshes(scan_mesh, 20000)
        
        # Convert to numpy for nearest neighbor search
        template_np = template_points.squeeze(0).cpu().numpy()
        scan_np = scan_points.squeeze(0).cpu().numpy()
        
        # Find nearest scan point for each template point
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_np)
        distances, _ = nbrs.kneighbors(template_np)
        
        # Create coverage mask - template points with nearby scan data
        coverage_mask = distances.flatten() < distance_threshold
        coverage_ratio = np.mean(coverage_mask)
        
        logger.info(f"Scan coverage: {coverage_ratio:.1%} of template area")
        
        return torch.from_numpy(coverage_mask).to(self.device)
    
    def create_vertex_coverage_mask(self, template_mesh, scan_mesh, distance_threshold=0.05):
        """
        Create a per-vertex mask indicating which template vertices should be deformed
        
        Args:
            template_mesh: Template mesh
            scan_mesh: Scan mesh
            distance_threshold: Distance threshold for coverage
            
        Returns:
            torch.Tensor: Per-vertex boolean mask for template vertices
        """
        logger.info("Creating vertex-level coverage mask...")
        
        template_verts = template_mesh.verts_packed()
        scan_points = sample_points_from_meshes(scan_mesh, 20000).squeeze(0)
        
        # Convert to numpy for efficient nearest neighbor search
        template_np = template_verts.cpu().numpy()
        scan_np = scan_points.cpu().numpy()
        
        # Find distance from each template vertex to nearest scan point
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_np)
        distances, _ = nbrs.kneighbors(template_np)
        
        # Vertices with nearby scan data should be deformed
        vertex_mask = torch.from_numpy(distances.flatten() < distance_threshold).to(self.device)
        
        coverage_ratio = torch.mean(vertex_mask.float())
        logger.info(f"Vertex coverage: {coverage_ratio:.1%} of template vertices will be deformed")
        
        # If coverage is too high (>95%), try a smaller threshold
        if coverage_ratio > 0.95:
            smaller_threshold = distance_threshold * 0.5
            logger.warning(f"Coverage too high ({coverage_ratio:.1%}), trying smaller threshold: {smaller_threshold}")
            vertex_mask = torch.from_numpy(distances.flatten() < smaller_threshold).to(self.device)
            coverage_ratio = torch.mean(vertex_mask.float())
            logger.info(f"New coverage: {coverage_ratio:.1%} of template vertices")
        
        # Ensure we have some vertices to optimize
        if coverage_ratio < 0.05:
            logger.warning(f"Coverage very low ({coverage_ratio:.1%}), using larger threshold")
            larger_threshold = distance_threshold * 2.0
            vertex_mask = torch.from_numpy(distances.flatten() < larger_threshold).to(self.device)
            coverage_ratio = torch.mean(vertex_mask.float())
            logger.info(f"Adjusted coverage: {coverage_ratio:.1%} of template vertices")
        
        return vertex_mask
    
    def compute_region_aware_chamfer_loss(self, template_points, scan_points, template_mask):
        """
        Compute chamfer loss only for template points in covered regions
        
        Args:
            template_points: Points sampled from deformed template [N, 3]
            scan_points: Points sampled from scan [M, 3] 
            template_mask: Mask indicating which template points should be fitted [N]
            
        Returns:
            torch.Tensor: Region-aware chamfer loss
        """
        if template_mask.sum() == 0:
            logger.warning("No template points in covered region!")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Fix tensor shapes - template_mask should match template_points
        if template_mask.dim() == 2:
            template_mask = template_mask.squeeze()
            
        if len(template_mask) != len(template_points):
            logger.warning(f"Mask length {len(template_mask)} != points length {len(template_points)}")
            # Create a random subset if shapes don't match
            n_points = len(template_points)
            n_covered = min(int(0.7 * n_points), template_mask.sum().item())  # Use 70% of points
            indices = torch.randperm(n_points, device=self.device)[:n_covered]
            covered_template_points = template_points[indices]
        else:
            # Only use template points in covered regions
            covered_template_points = template_points[template_mask]
        
        if len(covered_template_points) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Standard chamfer distance for covered regions only
        loss_chamfer, _ = chamfer_distance(
            covered_template_points.unsqueeze(0), 
            scan_points.unsqueeze(0)
        )
        
        return loss_chamfer
    
    def compute_region_aware_losses(self, deformed_mesh, scan_mesh, vertex_mask, weights, num_samples=5000):
        """Compute losses with region awareness"""
        
        # Sample points from both meshes
        template_points = sample_points_from_meshes(deformed_mesh, num_samples).squeeze(0)
        scan_points = sample_points_from_meshes(scan_mesh, num_samples).squeeze(0)
        
        # Create point-level mask (approximate - maps vertex mask to sampled points)
        # This is an approximation since we can't directly map sampled points to vertices
        point_mask = torch.randperm(num_samples, device=self.device)[:int(num_samples * vertex_mask.float().mean())]
        point_coverage_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        point_coverage_mask[point_mask] = True
        
        # Region-aware chamfer loss (only fit where scan data exists)
        loss_chamfer = self.compute_region_aware_chamfer_loss(
            template_points, scan_points, point_coverage_mask
        )
        
        # Standard regularization losses (applied to whole mesh for smoothness)
        loss_edge = mesh_edge_loss(deformed_mesh)
        loss_laplacian = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
        loss_normal = mesh_normal_consistency(deformed_mesh)
        
        # Total weighted loss
        total_loss = (
            weights['chamfer'] * loss_chamfer +
            weights['edge'] * loss_edge +
            weights['laplacian'] * loss_laplacian +
            weights['normal'] * loss_normal
        )
        
        return {
            'total': total_loss,
            'chamfer': loss_chamfer,
            'edge': loss_edge,
            'laplacian': loss_laplacian,
            'normal': loss_normal
        }
    
    def fit_mesh_region_aware(self, template_mesh, scan_mesh, iterations=2000, 
                             lr=1.0, weights=None, distance_threshold=0.05):
        """
        Region-aware mesh fitting - only deform template where scan data exists
        
        Args:
            template_mesh: Template mesh to deform
            scan_mesh: Target scan mesh
            iterations: Number of optimization iterations
            lr: Learning rate
            weights: Loss weights
            distance_threshold: Distance threshold for scan coverage
        """
        
        # Default weights with higher regularization
        if weights is None:
            weights = {
                'chamfer': 1.0,       # Data fitting (only in covered regions)
                'edge': 2.0,          # Higher edge regularization
                'laplacian': 20.0,    # Much higher Laplacian to preserve uncovered regions
                'normal': 0.1         # Higher normal consistency
            }
        
        logger.info(f"Starting region-aware optimization with {iterations} iterations")
        logger.info(f"Distance threshold: {distance_threshold}")
        logger.info(f"Loss weights: {weights}")
        
        # Find which template vertices have scan coverage
        vertex_mask = self.create_vertex_coverage_mask(
            template_mesh, scan_mesh, distance_threshold
        )
        
        # Initialize vertex deformation parameters
        deform_verts = torch.zeros_like(
            template_mesh.verts_packed(),
            requires_grad=True
        )
        
        # Create a mask for which vertices should be optimized
        # Only allow deformation for vertices in covered regions
        constrained_deform_verts = deform_verts.clone()
        
        # Optimizer - only optimize covered vertices
        optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        
        # Loss tracking
        losses_history = []
        
        # Optimization loop
        pbar = tqdm(range(iterations), desc="Region-aware fitting")
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Apply selective deformation - only deform covered vertices
            selective_deform = deform_verts * vertex_mask.unsqueeze(1).float()
            deformed_mesh = template_mesh.offset_verts(selective_deform)
            
            # Compute region-aware losses
            losses = self.compute_region_aware_losses(
                deformed_mesh, scan_mesh, vertex_mask, weights
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Manual gradient masking - zero gradients for uncovered vertices
            with torch.no_grad():
                if deform_verts.grad is not None:
                    deform_verts.grad *= vertex_mask.unsqueeze(1).float()
            
            optimizer.step()
            scheduler.step()
            
            # Log progress
            losses_history.append({k: float(v.detach().cpu()) for k, v in losses.items()})
            
            if i % 100 == 0:
                pbar.set_postfix({
                    'Total': f"{losses['total']:.6f}",
                    'Chamfer': f"{losses['chamfer']:.6f}",
                    'Laplacian': f"{losses['laplacian']:.6f}"
                })
        
        # Final deformed mesh with selective deformation
        final_selective_deform = deform_verts * vertex_mask.unsqueeze(1).float()
        final_mesh = template_mesh.offset_verts(final_selective_deform)
        
        logger.info(f"Region-aware optimization completed! Final loss: {losses['total']:.6f}")
        logger.info(f"Deformed vertices: {vertex_mask.sum().item()}/{len(vertex_mask)} "
                   f"({100*vertex_mask.float().mean():.1f}%)")
        
        return final_mesh, losses_history, vertex_mask
    
    def save_ply_mesh(self, mesh, output_path):
        """Save mesh as PLY file"""
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_ply(output_path, verts, faces)
        logger.info(f"Saved fitted mesh to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Region-Aware FLAME PLY Mesh Deformation')
    parser.add_argument('--template', required=True, 
                       help='Path to rigidly aligned FLAME template PLY file')
    parser.add_argument('--scan', required=True,
                       help='Path to raw face scan PLY file')
    parser.add_argument('--output', required=True,
                       help='Path to save fitted mesh PLY file')
    parser.add_argument('--iterations', type=int, default=2000,
                       help='Number of optimization iterations (default: 2000)')
    parser.add_argument('--lr', type=float, default=1.0,
                       help='Learning rate (default: 1.0)')
    parser.add_argument('--distance_threshold', type=float, default=0.05,
                       help='Distance threshold for scan coverage (default: 0.05)')
    parser.add_argument('--w_chamfer', type=float, default=1.0,
                       help='Chamfer loss weight (default: 1.0)')
    parser.add_argument('--w_edge', type=float, default=2.0,
                       help='Edge loss weight (default: 2.0)')
    parser.add_argument('--w_laplacian', type=float, default=20.0,
                       help='Laplacian loss weight (default: 20.0)')
    parser.add_argument('--w_normal', type=float, default=0.1,
                       help='Normal loss weight (default: 0.1)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize deformation system
        deformer = RegionAwareFLAMEDeformation(force_cpu=args.cpu)
        
        # Load PLY files
        logger.info("Loading PLY files...")
        template_mesh = deformer.load_ply_mesh(args.template)
        scan_mesh = deformer.load_ply_mesh(args.scan)
        
        # Set loss weights
        weights = {
            'chamfer': args.w_chamfer,
            'edge': args.w_edge,
            'laplacian': args.w_laplacian,
            'normal': args.w_normal
        }
        
        # Fit mesh with region awareness
        fitted_mesh, loss_history, vertex_mask = deformer.fit_mesh_region_aware(
            template_mesh, 
            scan_mesh,
            iterations=args.iterations,
            lr=args.lr,
            weights=weights,
            distance_threshold=args.distance_threshold
        )
        
        # Save result
        deformer.save_ply_mesh(fitted_mesh, args.output)
        
        # Print summary
        initial_loss = loss_history[0]['total']
        final_loss = loss_history[-1]['total']
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        logger.info(f"Fitting summary:")
        logger.info(f"  Initial loss: {initial_loss:.6f}")
        logger.info(f"  Final loss: {final_loss:.6f}")
        logger.info(f"  Improvement: {improvement:.2f}%")
        logger.info(f"  Vertices deformed: {vertex_mask.sum().item()}/{len(vertex_mask)}")
        logger.info(f"  Output saved: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during mesh fitting: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()