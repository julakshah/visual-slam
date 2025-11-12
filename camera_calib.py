# THIS CODE IS 100% WRITTEN BY CLAUDE
# THIS CODE UNDISTORTS IMAGES FOR FOV MODELS

import cv2
import numpy as np
import os
from pathlib import Path


def read_camera_params(camera_file):
    """Read camera parameters from camera.txt file"""
    with open(camera_file, "r") as f:
        lines = f.readlines()

    # Line 1: normalized intrinsics + omega
    norm_params = list(map(float, lines[0].strip().split()))
    fx_norm, fy_norm, cx_norm, cy_norm, omega = norm_params[:5]

    # Line 2: original image dimensions
    orig_width, orig_height = map(int, lines[1].strip().split())

    # Line 3: distortion coefficients (not used with FOV model)
    dist_coeffs = list(map(float, lines[2].strip().split()))

    # Line 4: output image dimensions
    out_width, out_height = map(int, lines[3].strip().split())

    # Convert normalized parameters to pixel coordinates
    fx = fx_norm * orig_width
    fy = fy_norm * orig_height
    cx = cx_norm * orig_width
    cy = cy_norm * orig_height

    return fx, fy, cx, cy, omega, (orig_width, orig_height), (out_width, out_height)


def fov_undistort(img, fx, fy, cx, cy, omega, output_size):
    """
    Undistort image using FOV (Field of View) camera model

    FOV model: r_d = (1/omega) * atan(2 * r_u * tan(omega/2))
    where r_u is undistorted radius, r_d is distorted radius
    """
    height, width = img.shape[:2]
    out_width, out_height = output_size

    # Create output image
    undistorted = np.zeros((out_height, out_width, 3), dtype=img.dtype)

    # Precompute tan(omega/2) for efficiency
    tan_omega_half = np.tan(omega / 2.0)

    # For each pixel in the output (undistorted) image
    for v in range(out_height):
        for u in range(out_width):
            # Normalize coordinates (output image)
            x_u = (u - out_width / 2.0) / fx
            y_u = (v - out_height / 2.0) / fy

            # Compute undistorted radius
            r_u = np.sqrt(x_u**2 + y_u**2)

            if r_u < 1e-6:
                # Center pixel - no distortion
                u_d = int(cx)
                v_d = int(cy)
            else:
                # Apply FOV distortion model
                # r_d = (1/omega) * atan(2 * r_u * tan(omega/2))
                r_d = (1.0 / omega) * np.arctan(2.0 * r_u * tan_omega_half)

                # Scale distorted radius to get distorted coordinates
                scale = r_d / r_u
                x_d = x_u * scale
                y_d = y_u * scale

                # Convert back to pixel coordinates (in original image)
                u_d = x_d * fx + cx
                v_d = y_d * fy + cy

            # Check bounds and copy pixel
            if 0 <= u_d < width and 0 <= v_d < height:
                # Bilinear interpolation for sub-pixel accuracy
                u_d_int = int(u_d)
                v_d_int = int(v_d)

                # Get fractional parts
                du = u_d - u_d_int
                dv = v_d - v_d_int

                # Bilinear interpolation
                if u_d_int + 1 < width and v_d_int + 1 < height:
                    p00 = img[v_d_int, u_d_int]
                    p10 = img[v_d_int, u_d_int + 1]
                    p01 = img[v_d_int + 1, u_d_int]
                    p11 = img[v_d_int + 1, u_d_int + 1]

                    pixel = (
                        (1 - du) * (1 - dv) * p00
                        + du * (1 - dv) * p10
                        + (1 - du) * dv * p01
                        + du * dv * p11
                    )

                    undistorted[v, u] = pixel.astype(img.dtype)
                else:
                    undistorted[v, u] = img[v_d_int, u_d_int]

    return undistorted


def fov_undistort_optimized(img, fx, fy, cx, cy, omega, output_size):
    """
    Optimized version using numpy vectorization and cv2.remap
    Much faster than pixel-by-pixel processing
    """
    height, width = img.shape[:2]
    out_width, out_height = output_size

    # Create meshgrid for output image coordinates
    u_map = np.arange(out_width, dtype=np.float32)
    v_map = np.arange(out_height, dtype=np.float32)
    u_map, v_map = np.meshgrid(u_map, v_map)

    # Normalize coordinates (output image)
    x_u = (u_map - out_width / 2.0) / fx
    y_u = (v_map - out_height / 2.0) / fy

    # Compute undistorted radius
    r_u = np.sqrt(x_u**2 + y_u**2)

    # Precompute tan(omega/2)
    tan_omega_half = np.tan(omega / 2.0)

    # Apply FOV distortion model (vectorized)
    r_d = (1.0 / omega) * np.arctan(2.0 * r_u * tan_omega_half)

    # Handle center pixels (avoid division by zero)
    scale = np.where(r_u < 1e-6, 1.0, r_d / r_u)

    # Get distorted coordinates
    x_d = x_u * scale
    y_d = y_u * scale

    # Convert back to pixel coordinates in original image
    map_x = (x_d * fx + cx).astype(np.float32)
    map_y = (y_d * fy + cy).astype(np.float32)

    # Use cv2.remap for efficient interpolation
    undistorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return undistorted


def rectify_images(images_dir, output_dir, camera_file, use_optimized=True):
    """Rectify all images using FOV camera model"""

    # Read camera parameters
    fx, fy, cx, cy, omega, orig_size, out_size = read_camera_params(camera_file)

    print("=== FOV Camera Model Parameters ===")
    print(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"Omega (FOV parameter): {omega:.6f} rad ({np.degrees(omega):.2f}°)")
    print(f"Total FOV: ~{2 * np.degrees(omega):.2f}°")
    print(f"Original size: {orig_size}")
    print(f"Output size: {out_size}")
    print(f"Using {'optimized' if use_optimized else 'standard'} undistortion\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all jpg files and sort them
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

    if not image_files:
        print(f"No .jpg files found in {images_dir}")
        return

    print(f"Found {len(image_files)} images to process\n")

    # Select undistortion function
    undistort_func = fov_undistort_optimized if use_optimized else fov_undistort

    # Process each image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue

        # Undistort using FOV model
        rectified = undistort_func(img, fx, fy, cx, cy, omega, out_size)

        # Save rectified image
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, rectified)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")

    print(f"\nDone! Rectified images saved to {output_dir}")


if __name__ == "__main__":
    # Set paths
    images_dir = "v-slam-dataset/images"
    output_dir = "v-slam-dataset/images_rectified"
    camera_file = "v-slam-dataset/camera.txt"

    # Check if files exist
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} directory not found")
        exit(1)

    if not os.path.exists(camera_file):
        print(f"Error: {camera_file} not found")
        exit(1)

    # Rectify images using FOV model
    # Set use_optimized=False for slower but simpler implementation
    rectify_images(images_dir, output_dir, camera_file, use_optimized=True)
