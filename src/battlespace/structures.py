"""
Structure layer for battlespace environment.
Manages buildings, obstacles, and infrastructure.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from rtree import index


class StructureType(Enum):
    """Types of structures in the environment."""
    BUILDING = "building"
    TOWER = "tower"
    BRIDGE = "bridge"
    HANGAR = "hangar"
    RADAR = "radar"
    SAM_SITE = "sam_site"
    RUNWAY = "runway"
    WALL = "wall"
    TREE = "tree"  # For future forest areas


@dataclass
class Structure:
    """
    Base structure class representing an obstacle in the environment.
    """
    id: int
    position: np.ndarray  # [x, y, z_base]
    dimensions: np.ndarray  # [width, depth, height]
    type: StructureType
    rotation: float = 0.0  # Rotation around z-axis in radians
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get axis-aligned bounding box.
        
        Returns:
            (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        half_width = self.dimensions[0] / 2
        half_depth = self.dimensions[1] / 2
        
        # Simple AABB for now (ignoring rotation for collision)
        min_x = self.position[0] - half_width
        max_x = self.position[0] + half_width
        min_y = self.position[1] - half_depth
        max_y = self.position[1] + half_depth
        min_z = self.position[2]
        max_z = self.position[2] + self.dimensions[2]
        
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if point is inside structure.
        
        Args:
            point: [x, y, z] position
            
        Returns:
            True if point is inside structure
        """
        bbox = self.bounding_box
        return (bbox[0] <= point[0] <= bbox[3] and
                bbox[1] <= point[1] <= bbox[4] and
                bbox[2] <= point[2] <= bbox[5])
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """
        Calculate minimum distance from point to structure.
        
        Args:
            point: [x, y, z] position
            
        Returns:
            Minimum distance in meters
        """
        bbox = self.bounding_box
        
        # Calculate distance to closest point on bounding box
        dx = max(bbox[0] - point[0], 0, point[0] - bbox[3])
        dy = max(bbox[1] - point[1], 0, point[1] - bbox[4])
        dz = max(bbox[2] - point[2], 0, point[2] - bbox[5])
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)


class StructureLayer:
    """
    Manages all structures in the battlespace.
    """
    
    def __init__(self, width: float, height: float,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize structure layer.
        
        Args:
            width: Battlespace width in meters
            height: Battlespace height in meters
            config: Structure configuration
        """
        self.width = width
        self.height = height
        self.config = config or {}
        
        # Structure storage
        self.structures: List[Structure] = []
        self.next_id = 0
        
        # Spatial index for efficient queries
        self.spatial_index = None
        self._rebuild_spatial_index()
        
    def _rebuild_spatial_index(self):
        """Rebuild the spatial index from current structures."""
        # Create new R-tree index
        p = index.Property()
        p.dimension = 3
        self.spatial_index = index.Index(properties=p)
        
        # Add all structures to index
        for struct in self.structures:
            bbox = struct.bounding_box
            self.spatial_index.insert(struct.id, bbox)
    
    def add_structure(self, position: np.ndarray, dimensions: np.ndarray,
                     structure_type: StructureType, rotation: float = 0.0) -> Structure:
        """
        Add a structure to the environment.
        
        Args:
            position: [x, y, z_base] position
            dimensions: [width, depth, height]
            structure_type: Type of structure
            rotation: Rotation angle in radians
            
        Returns:
            Created structure
        """
        struct = Structure(
            id=self.next_id,
            position=position,
            dimensions=dimensions,
            type=structure_type,
            rotation=rotation
        )
        
        self.structures.append(struct)
        self.next_id += 1
        
        # Add to spatial index
        if self.spatial_index is not None:
            self.spatial_index.insert(struct.id, struct.bounding_box)
        
        return struct
    
    def remove_structure(self, struct_id: int):
        """
        Remove a structure by ID.
        
        Args:
            struct_id: Structure ID to remove
        """
        # Find and remove structure
        for i, struct in enumerate(self.structures):
            if struct.id == struct_id:
                # Remove from spatial index
                if self.spatial_index is not None:
                    self.spatial_index.delete(struct_id, struct.bounding_box)
                
                # Remove from list
                del self.structures[i]
                break
    
    def generate(self, terrain_layer=None):
        """
        Generate structures based on configuration.
        
        Args:
            terrain_layer: Terrain layer for elevation queries
        """
        if not self.config.get('enabled', False):
            return
        
        # Clear existing structures
        self.structures.clear()
        self._rebuild_spatial_index()
        
        # Generate different structure types
        if 'cities' in self.config:
            for city_config in self.config['cities']:
                self._generate_city(city_config, terrain_layer)
        
        if 'airbases' in self.config:
            for base_config in self.config['airbases']:
                self._generate_airbase(base_config, terrain_layer)
        
        if 'sam_sites' in self.config:
            for sam_config in self.config['sam_sites']:
                self._generate_sam_site(sam_config, terrain_layer)
    
    def _generate_city(self, config: Dict[str, Any], terrain_layer=None):
        """Generate an urban area."""
        center = np.array(config['center'] + [0])
        radius = config['radius']
        density = config.get('density', 'medium')
        
        # Determine number of buildings based on density
        num_buildings = {
            'low': 20,
            'medium': 50,
            'high': 100
        }.get(density, 50)
        
        # Generate buildings in circular area
        for _ in range(num_buildings):
            # Random position within radius
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, radius)
            
            x = center[0] + dist * np.cos(angle)
            y = center[1] + dist * np.sin(angle)
            
            # Get terrain elevation if available
            z = 0
            if terrain_layer is not None:
                z = terrain_layer.get_elevation_at(x, y)
            
            # Random building dimensions
            width = np.random.uniform(20, 100)
            depth = np.random.uniform(20, 100)
            height = np.random.uniform(30, 200)
            
            # Random rotation
            rotation = np.random.uniform(0, 2 * np.pi)
            
            self.add_structure(
                np.array([x, y, z]),
                np.array([width, depth, height]),
                StructureType.BUILDING,
                rotation
            )
    
    def _generate_airbase(self, config: Dict[str, Any], terrain_layer=None):
        """Generate an airbase with runway and buildings."""
        position = np.array(config['position'] + [0])
        heading = np.radians(config.get('runway_heading', 0))
        
        # Get terrain elevation
        if terrain_layer is not None:
            position[2] = terrain_layer.get_elevation_at(position[0], position[1])
        
        # Add runway (2000m x 50m)
        self.add_structure(
            position,
            np.array([2000, 50, 1]),
            StructureType.RUNWAY,
            heading
        )
        
        # Add control tower
        tower_offset = np.array([100, 100, 0])
        self.add_structure(
            position + tower_offset,
            np.array([20, 20, 50]),
            StructureType.TOWER
        )
        
        # Add hangars
        for i in range(3):
            hangar_offset = np.array([-200 - i*100, 0, 0])
            self.add_structure(
                position + hangar_offset,
                np.array([80, 60, 20]),
                StructureType.HANGAR,
                heading
            )
        
        # Add SAM site if configured
        if config.get('has_sam', False):
            sam_offset = np.array([500, 500, 0])
            self.add_structure(
                position + sam_offset,
                np.array([10, 10, 5]),
                StructureType.SAM_SITE
            )
    
    def _generate_sam_site(self, config: Dict[str, Any], terrain_layer=None):
        """Generate a SAM site."""
        position = np.array(config['position'] + [0])
        
        # Get terrain elevation
        if terrain_layer is not None:
            position[2] = terrain_layer.get_elevation_at(position[0], position[1])
        
        # Add SAM launcher
        self.add_structure(
            position,
            np.array([10, 10, 5]),
            StructureType.SAM_SITE
        )
        
        # Add radar
        radar_offset = np.array([50, 0, 0])
        self.add_structure(
            position + radar_offset,
            np.array([5, 5, 15]),
            StructureType.RADAR
        )
    
    def check_collision(self, position: np.ndarray, radius: float = 1.0) -> bool:
        """
        Check if a sphere collides with any structure.
        
        Args:
            position: [x, y, z] center of sphere
            radius: Sphere radius in meters
            
        Returns:
            True if collision detected
        """
        # Query spatial index for nearby structures
        query_bbox = (
            position[0] - radius, position[1] - radius, position[2] - radius,
            position[0] + radius, position[1] + radius, position[2] + radius
        )
        
        nearby_ids = list(self.spatial_index.intersection(query_bbox))
        
        # Check detailed collision with nearby structures
        for struct_id in nearby_ids:
            struct = self.structures[struct_id]
            if struct.distance_to_point(position) <= radius:
                return True
        
        return False
    
    def get_nearby_structures(self, position: np.ndarray, 
                            range: float) -> List[Structure]:
        """
        Get all structures within range of position.
        
        Args:
            position: [x, y, z] query position
            range: Search radius in meters
            
        Returns:
            List of structures within range
        """
        query_bbox = (
            position[0] - range, position[1] - range, position[2] - range,
            position[0] + range, position[1] + range, position[2] + range
        )
        
        nearby_ids = list(self.spatial_index.intersection(query_bbox))
        
        # Filter by actual distance
        nearby_structures = []
        for struct_id in nearby_ids:
            struct = self.structures[struct_id]
            if struct.distance_to_point(position) <= range:
                nearby_structures.append(struct)
        
        return nearby_structures
    
    def check_line_intersection(self, start: np.ndarray, end: np.ndarray,
                               point: np.ndarray) -> bool:
        """
        Check if a line segment intersects any structure.
        
        Args:
            start: Start position of line
            end: End position of line
            point: Current point on line (for optimization)
            
        Returns:
            True if line intersects a structure
        """
        # Get bounding box of line segment
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        min_y = min(start[1], end[1])
        max_y = max(start[1], end[1])
        min_z = min(start[2], end[2])
        max_z = max(start[2], end[2])
        
        query_bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
        nearby_ids = list(self.spatial_index.intersection(query_bbox))
        
        # Check intersection with each nearby structure
        for struct_id in nearby_ids:
            struct = self.structures[struct_id]
            # Simplified check - if point is inside structure
            if struct.contains_point(point):
                return True
        
        return False