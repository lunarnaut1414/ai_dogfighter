"""
Coordinate transformation and validation utilities.
"""

import numpy as np
from typing import Tuple, Optional


def world_to_grid(x: float, y: float, resolution: float) -> Tuple[int, int]:
    """
    Convert world coordinates to grid indices.
    
    Args:
        x: World x coordinate (meters)
        y: World y coordinate (meters)
        resolution: Grid cell size (meters)
        
    Returns:
        Grid indices (ix, iy)
    """
    ix = int(x / resolution)
    iy = int(y / resolution)
    return ix, iy


def grid_to_world(ix: int, iy: int, resolution: float) -> Tuple[float, float]:
    """
    Convert grid indices to world coordinates (center of cell).
    
    Args:
        ix: Grid x index
        iy: Grid y index
        resolution: Grid cell size (meters)
        
    Returns:
        World coordinates (x, y) at cell center
    """
    x = (ix + 0.5) * resolution
    y = (iy + 0.5) * resolution
    return x, y


def validate_position(position: np.ndarray, bounds: Optional[Tuple] = None) -> bool:
    """
    Validate that position is valid array with optional bounds checking.
    
    Args:
        position: Position array to validate
        bounds: Optional (min_x, min_y, min_z, max_x, max_y, max_z)
        
    Returns:
        True if position is valid
    """
    # Check array shape and type
    if not isinstance(position, np.ndarray):
        position = np.array(position)
    
    if position.shape != (3,):
        return False
    
    # Check for NaN or Inf
    if not np.all(np.isfinite(position)):
        return False
    
    # Check bounds if provided
    if bounds is not None:
        if len(bounds) != 6:
            raise ValueError("Bounds must be (min_x, min_y, min_z, max_x, max_y, max_z)")
        
        return (bounds[0] <= position[0] <= bounds[3] and
                bounds[1] <= position[1] <= bounds[4] and
                bounds[2] <= position[2] <= bounds[5])
    
    return True


def distance_2d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate 2D Euclidean distance (ignoring altitude).
    
    Args:
        pos1: First position [x, y, z]
        pos2: Second position [x, y, z]
        
    Returns:
        2D distance in meters
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx*dx + dy*dy)


def distance_3d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate 3D Euclidean distance.
    
    Args:
        pos1: First position [x, y, z]
        pos2: Second position [x, y, z]
        
    Returns:
        3D distance in meters
    """
    return np.linalg.norm(pos2 - pos1)


def bearing_2d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate bearing from pos1 to pos2.
    
    Args:
        pos1: From position [x, y, z]
        pos2: To position [x, y, z]
        
    Returns:
        Bearing in radians [0, 2π]
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    bearing = np.arctan2(dy, dx)
    
    # Normalize to [0, 2π]
    if bearing < 0:
        bearing += 2 * np.pi
    
    return bearing


def elevation_angle(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate elevation angle from pos1 to pos2.
    
    Args:
        pos1: Observer position [x, y, z]
        pos2: Target position [x, y, z]
        
    Returns:
        Elevation angle in radians [-π/2, π/2]
    """
    horizontal_dist = distance_2d(pos1, pos2)
    vertical_dist = pos2[2] - pos1[2]
    
    if horizontal_dist == 0:
        return np.pi/2 if vertical_dist > 0 else -np.pi/2
    
    return np.arctan(vertical_dist / horizontal_dist)


def rotate_2d(point: np.ndarray, angle: float, 
              center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rotate 2D point around center.
    
    Args:
        point: Point to rotate [x, y]
        angle: Rotation angle in radians
        center: Center of rotation (default origin)
        
    Returns:
        Rotated point [x, y]
    """
    if center is None:
        center = np.zeros(2)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Translate to origin
    p = point[:2] - center[:2]
    
    # Rotate
    rotated = np.array([
        p[0] * cos_a - p[1] * sin_a,
        p[0] * sin_a + p[1] * cos_a
    ])
    
    # Translate back
    return rotated + center[:2]


def project_point_on_line(point: np.ndarray, line_start: np.ndarray,
                         line_end: np.ndarray) -> np.ndarray:
    """
    Project point onto line segment.
    
    Args:
        point: Point to project
        line_start: Start of line segment
        line_end: End of line segment
        
    Returns:
        Closest point on line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq == 0:
        return line_start
    
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)
    
    return line_start + t * line_vec


def line_intersection_2d(p1: np.ndarray, p2: np.ndarray,
                        p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    """
    Find intersection point of two 2D line segments.
    
    Args:
        p1, p2: First line segment
        p3, p4: Second line segment
        
    Returns:
        Intersection point or None if no intersection
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]
    x4, y4 = p4[:2]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection exists
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])
    
    return None


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if point is inside polygon using ray casting.
    
    Args:
        point: Point to test [x, y]
        polygon: Array of polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point[:2]
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0][:2]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][:2]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside