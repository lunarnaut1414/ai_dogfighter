"""
Interpolation utilities for smooth terrain and field queries.
"""

import numpy as np
from typing import Tuple


def bilinear_interpolate(z00: float, z10: float, z01: float, z11: float,
                         fx: float, fy: float) -> float:
    """
    Perform bilinear interpolation.
    
    Args:
        z00: Value at (0, 0)
        z10: Value at (1, 0)
        z01: Value at (0, 1)
        z11: Value at (1, 1)
        fx: Fractional x position [0, 1]
        fy: Fractional y position [0, 1]
        
    Returns:
        Interpolated value
    """
    # Interpolate along x for both y levels
    z0 = z00 * (1 - fx) + z10 * fx
    z1 = z01 * (1 - fx) + z11 * fx
    
    # Interpolate along y
    return z0 * (1 - fy) + z1 * fy


def trilinear_interpolate(values: np.ndarray, fx: float, fy: float, fz: float) -> float:
    """
    Perform trilinear interpolation on a 2x2x2 cube.
    
    Args:
        values: 2x2x2 array of values
        fx: Fractional x position [0, 1]
        fy: Fractional y position [0, 1]
        fz: Fractional z position [0, 1]
        
    Returns:
        Interpolated value
    """
    # Interpolate along x
    c00 = values[0, 0, 0] * (1 - fx) + values[0, 0, 1] * fx
    c01 = values[0, 1, 0] * (1 - fx) + values[0, 1, 1] * fx
    c10 = values[1, 0, 0] * (1 - fx) + values[1, 0, 1] * fx
    c11 = values[1, 1, 0] * (1 - fx) + values[1, 1, 1] * fx
    
    # Interpolate along y
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    
    # Interpolate along z
    return c0 * (1 - fz) + c1 * fz


def cubic_interpolate(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """
    Perform cubic interpolation (Catmull-Rom spline).
    
    Args:
        p0, p1, p2, p3: Four consecutive points
        t: Position between p1 and p2 [0, 1]
        
    Returns:
        Interpolated value
    """
    t2 = t * t
    t3 = t2 * t
    
    return (
        -0.5 * t3 + t2 - 0.5 * t
    ) * p0 + (
        1.5 * t3 - 2.5 * t2 + 1.0
    ) * p1 + (
        -1.5 * t3 + 2.0 * t2 + 0.5 * t
    ) * p2 + (
        0.5 * t3 - 0.5 * t2
    ) * p3


def bicubic_interpolate(grid: np.ndarray, fx: float, fy: float) -> float:
    """
    Perform bicubic interpolation on a 4x4 grid.
    
    Args:
        grid: 4x4 array of values
        fx: Fractional x position [0, 1] within center cell
        fy: Fractional y position [0, 1] within center cell
        
    Returns:
        Interpolated value
    """
    # Interpolate along x for each row
    rows = []
    for y in range(4):
        rows.append(cubic_interpolate(
            grid[y, 0], grid[y, 1], grid[y, 2], grid[y, 3], fx
        ))
    
    # Interpolate along y
    return cubic_interpolate(rows[0], rows[1], rows[2], rows[3], fy)


def interpolate_grid_point(grid: np.ndarray, x: float, y: float, 
                           method: str = 'bilinear') -> float:
    """
    Interpolate value from 2D grid at arbitrary position.
    
    Args:
        grid: 2D array of values
        x: X position in grid coordinates
        y: Y position in grid coordinates
        method: Interpolation method ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        Interpolated value
    """
    height, width = grid.shape
    
    # Clamp to grid bounds
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    
    if method == 'nearest':
        ix = int(round(x))
        iy = int(round(y))
        return grid[iy, ix]
    
    elif method == 'bilinear':
        ix = int(x)
        iy = int(y)
        
        # Handle edge cases
        if ix >= width - 1:
            ix = width - 2
        if iy >= height - 1:
            iy = height - 2
        
        fx = x - ix
        fy = y - iy
        
        return bilinear_interpolate(
            grid[iy, ix], grid[iy, ix+1],
            grid[iy+1, ix], grid[iy+1, ix+1],
            fx, fy
        )
    
    elif method == 'bicubic':
        ix = int(x)
        iy = int(y)
        
        # Need 4x4 grid for bicubic
        if ix < 1 or ix >= width - 2 or iy < 1 or iy >= height - 2:
            # Fall back to bilinear at edges
            return interpolate_grid_point(grid, x, y, 'bilinear')
        
        fx = x - ix
        fy = y - iy
        
        # Extract 4x4 subgrid
        subgrid = grid[iy-1:iy+3, ix-1:ix+3]
        
        return bicubic_interpolate(subgrid, fx, fy)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def smooth_field_2d(field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to 2D field.
    
    Args:
        field: 2D array to smooth
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed field
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field, sigma=sigma, mode='reflect')


def smooth_field_3d(field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to 3D field.
    
    Args:
        field: 3D array to smooth
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed field
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field, sigma=sigma, mode='reflect')