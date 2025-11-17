#!/usr/bin/env python3
"""
PLY File Diagnostic and Repair Script

This script diagnoses PLY file format issues and converts them to PyTorch3D-compatible format.
"""

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_ply_file(ply_path):
    """Diagnose PLY file format and structure"""
    logger.info(f"Diagnosing PLY file: {ply_path}")
    
    if not os.path.exists(ply_path):
        logger.error(f"File not found: {ply_path}")
        return None
    
    # Try different loading methods
    methods = []
    
    # Method 1: PyTorch3D
    try:
        from pytorch3d.io import load_ply
        verts, faces = load_ply(ply_path)
        methods.append(("PyTorch3D", verts.shape if verts is not None else None, 
                       faces.shape if faces is not None else None, "SUCCESS"))
    except Exception as e:
        methods.append(("PyTorch3D", None, None, f"FAILED: {str(e)}"))
    
    # Method 2: Trimesh
    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        methods.append(("Trimesh", mesh.vertices.shape, mesh.faces.shape, "SUCCESS"))
    except Exception as e:
        methods.append(("Trimesh", None, None, f"FAILED: {str(e)}"))
    
    # Method 3: Open3D (if available)
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(ply_path)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        methods.append(("Open3D", verts.shape, faces.shape, "SUCCESS"))
    except ImportError:
        methods.append(("Open3D", None, None, "NOT AVAILABLE"))
    except Exception as e:
        methods.append(("Open3D", None, None, f"FAILED: {str(e)}"))
    
    # Method 4: Manual PLY parsing
    try:
        verts, faces = parse_ply_manually(ply_path)
        methods.append(("Manual", verts.shape if verts is not None else None, 
                       faces.shape if faces is not None else None, "SUCCESS"))
    except Exception as e:
        methods.append(("Manual", None, None, f"FAILED: {str(e)}"))
    
    # Report results
    logger.info("Loading method results:")
    for method, v_shape, f_shape, status in methods:
        logger.info(f"  {method:<12}: Verts={v_shape}, Faces={f_shape}, Status={status}")
    
    return methods


def parse_ply_manually(ply_path):
    """Manually parse PLY file to understand format"""
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    vertex_count = 0
    face_count = 0
    in_header = True
    header_end_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('element face'):
            face_count = int(line.split()[-1])
        elif line == 'end_header':
            header_end_idx = i + 1
            break
    
    logger.info(f"PLY header info: {vertex_count} vertices, {face_count} faces")
    
    # Parse vertices
    vertices = []
    for i in range(header_end_idx, header_end_idx + vertex_count):
        parts = lines[i].strip().split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    vertices = np.array(vertices)
    
    # Parse faces
    faces = []
    face_start_idx = header_end_idx + vertex_count
    
    for i in range(face_start_idx, face_start_idx + face_count):
        parts = lines[i].strip().split()
        if len(parts) >= 4:  # Should have at least: count + 3 indices
            face_vertex_count = int(parts[0])
            if face_vertex_count == 3:  # Triangle
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
            else:
                logger.warning(f"Non-triangular face found with {face_vertex_count} vertices")
    
    faces = np.array(faces)
    
    return vertices, faces


def convert_ply_for_pytorch3d(input_path, output_path):
    """Convert PLY file to PyTorch3D compatible format"""
    logger.info(f"Converting {input_path} to PyTorch3D format...")
    
    try:
        # First try trimesh (most robust)
        import trimesh
        mesh = trimesh.load(input_path)
        
        logger.info(f"Loaded with Trimesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        # Check if faces are triangular (all faces should have 3 vertices)
        if hasattr(mesh, 'is_triangular'):
            is_triangular = mesh.is_triangular
        else:
            # Manual check for triangular faces
            is_triangular = mesh.faces.shape[1] == 3
        
        if not is_triangular:
            logger.info("Converting to triangular mesh...")
            # Force triangulation if needed
            mesh = mesh.triangulate()
        
        # Convert to PyTorch tensors
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))
        
        logger.info(f"Tensor shapes: vertices {vertices.shape}, faces {faces.shape}")
        
    except ImportError:
        logger.error("Trimesh not available and manual parsing failed (binary PLY)")
        logger.error("Please install trimesh: pip install trimesh")
        raise
    except Exception as e:
        logger.error(f"Failed to load with trimesh: {str(e)}")
        raise
    
    # Save in PyTorch3D compatible format
    from pytorch3d.io import save_ply
    save_ply(output_path, vertices, faces)
    
    # Verify the conversion worked
    try:
        from pytorch3d.io import load_ply
        test_verts, test_faces = load_ply(output_path)
        
        logger.info(f"✓ Conversion successful! Saved to {output_path}")
        logger.info(f"✓ Verification: {test_verts.shape[0]} vertices, {test_faces.shape[0]} faces")
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise
    
    return output_path


def fix_ply_files_for_flame(template_path, scan_path, output_dir="./fixed_ply"):
    """Fix both template and scan PLY files for FLAME fitting"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix template
    template_fixed = os.path.join(output_dir, "template_fixed.ply")
    logger.info("="*50)
    logger.info("FIXING TEMPLATE PLY")
    logger.info("="*50)
    diagnose_ply_file(template_path)
    convert_ply_for_pytorch3d(template_path, template_fixed)
    
    # Fix scan
    scan_fixed = os.path.join(output_dir, "scan_fixed.ply")
    logger.info("="*50) 
    logger.info("FIXING SCAN PLY")
    logger.info("="*50)
    diagnose_ply_file(scan_path)
    convert_ply_for_pytorch3d(scan_path, scan_fixed)
    
    return template_fixed, scan_fixed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PLY File Diagnostic and Repair")
    parser.add_argument('--diagnose', help='Diagnose a single PLY file')
    parser.add_argument('--convert', nargs=2, metavar=('input', 'output'), 
                       help='Convert PLY file to PyTorch3D format')
    parser.add_argument('--fix-flame', nargs=2, metavar=('template', 'scan'),
                       help='Fix both template and scan PLY files for FLAME')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_ply_file(args.diagnose)
        
    elif args.convert:
        convert_ply_for_pytorch3d(args.convert[0], args.convert[1])
        
    elif args.fix_flame:
        template_fixed, scan_fixed = fix_ply_files_for_flame(args.fix_flame[0], args.fix_flame[1])
        
        print("\n" + "="*60)
        print("PLY FILES FIXED FOR FLAME!")
        print("="*60)
        print(f"Fixed template: {template_fixed}")
        print(f"Fixed scan: {scan_fixed}")
        print("\nNow run:")
        print(f"python flame_ply_deformation.py --template {template_fixed} --scan {scan_fixed} --output fitted.ply")
        
    else:
        print("Please specify an action:")
        print("  --diagnose file.ply")
        print("  --convert input.ply output.ply") 
        print("  --fix-flame template.ply scan.ply")


if __name__ == "__main__":
    main()