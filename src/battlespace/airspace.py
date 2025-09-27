"""
Airspace management for battlespace environment.
Handles no-fly zones, altitude restrictions, and threat zones.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing


@dataclass
class NoFlyZone:
    """Represents a restricted airspace volume."""
    id: int
    polygon: Polygon  # 2D boundary
    min_altitude: float
    max_altitude: float
    name: str = "Restricted Area"
    
    def contains(self, position: np.ndarray) -> bool:
        """
        Check if position is inside no-fly zone.
        
        Args:
            position: [x, y, z] position
            
        Returns:
            True if position is inside zone
        """
        # Check altitude bounds
        if position[2] < self.min_altitude or position[2] > self.max_altitude:
            return False
        
        # Check horizontal bounds
        point = Point(position[0], position[1])
        return self.polygon.contains(point)


@dataclass
class ThreatZone:
    """Represents a threat coverage area (e.g., SAM site)."""
    id: int
    center: np.ndarray  # [x, y, z]
    max_range: float
    max_altitude: float
    min_altitude: float = 0.0
    threat_level: float = 1.0  # 0.0 to 1.0
    name: str = "Threat"
    
    def get_threat_level(self, position: np.ndarray) -> float:
        """
        Calculate threat level at position.
        
        Args:
            position: [x, y, z] position
            
        Returns:
            Threat level (0.0 = safe, 1.0 = maximum threat)
        """
        # Check altitude bounds
        if position[2] < self.min_altitude or position[2] > self.max_altitude:
            return 0.0
        
        # Calculate horizontal distance
        dx = position[0] - self.center[0]
        dy = position[1] - self.center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance >= self.max_range:
            return 0.0
        
        # Linear falloff with distance
        return self.threat_level * (1.0 - distance / self.max_range)


@dataclass
class ControlledAirspace:
    """Represents controlled airspace requiring clearance."""
    id: int
    polygon: Polygon
    floor: float
    ceiling: float
    clearance_level: int  # Required clearance level
    name: str = "Controlled Airspace"


class AirspaceLayer:
    """
    Manages airspace restrictions and zones.
    """
    
    def __init__(self, width: float, height: float, ceiling: float,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize airspace layer.
        
        Args:
            width: Battlespace width in meters
            height: Battlespace height in meters
            ceiling: Maximum altitude in meters
            config: Airspace configuration
        """
        self.width = width
        self.height = height
        self.altitude_ceiling = ceiling
        self.config = config or {}
        
        # Zone storage
        self.no_fly_zones: List[NoFlyZone] = []
        self.threat_zones: List[ThreatZone] = []
        self.controlled_airspace: List[ControlledAirspace] = []
        
        self.next_zone_id = 0
        
        # Initialize from config
        self._load_from_config()
    
    def _load_from_config(self):
        """Load airspace zones from configuration."""
        if 'no_fly_zones' in self.config:
            for zone_config in self.config['no_fly_zones']:
                self.add_no_fly_zone(
                    zone_config['polygon'],
                    zone_config.get('min_altitude', 0),
                    zone_config.get('max_altitude', self.altitude_ceiling),
                    zone_config.get('name', 'Restricted Area')
                )
        
        if 'threat_zones' in self.config:
            for threat_config in self.config['threat_zones']:
                self.add_threat_zone(
                    np.array(threat_config['center']),
                    threat_config['max_range'],
                    threat_config['max_altitude'],
                    threat_config.get('min_altitude', 0),
                    threat_config.get('threat_level', 1.0),
                    threat_config.get('name', 'Threat')
                )
    
    def add_no_fly_zone(self, polygon_points: List[Tuple[float, float]],
                       min_altitude: float, max_altitude: float,
                       name: str = "Restricted Area") -> NoFlyZone:
        """
        Add a no-fly zone.
        
        Args:
            polygon_points: List of (x, y) points defining the boundary
            min_altitude: Minimum altitude of restriction
            max_altitude: Maximum altitude of restriction
            name: Zone name
            
        Returns:
            Created no-fly zone
        """
        polygon = Polygon(polygon_points)
        
        zone = NoFlyZone(
            id=self.next_zone_id,
            polygon=polygon,
            min_altitude=min_altitude,
            max_altitude=max_altitude,
            name=name
        )
        
        self.no_fly_zones.append(zone)
        self.next_zone_id += 1
        
        return zone
    
    def add_threat_zone(self, center: np.ndarray, max_range: float,
                       max_altitude: float, min_altitude: float = 0.0,
                       threat_level: float = 1.0, name: str = "Threat") -> ThreatZone:
        """
        Add a threat zone (e.g., SAM coverage).
        
        Args:
            center: [x, y, z] center position
            max_range: Maximum horizontal range
            max_altitude: Maximum altitude coverage
            min_altitude: Minimum altitude coverage
            threat_level: Threat intensity (0-1)
            name: Zone name
            
        Returns:
            Created threat zone
        """
        zone = ThreatZone(
            id=self.next_zone_id,
            center=center,
            max_range=max_range,
            max_altitude=max_altitude,
            min_altitude=min_altitude,
            threat_level=threat_level,
            name=name
        )
        
        self.threat_zones.append(zone)
        self.next_zone_id += 1
        
        return zone
    
    def add_controlled_airspace(self, polygon_points: List[Tuple[float, float]],
                               floor: float, ceiling: float,
                               clearance_level: int = 1,
                               name: str = "Controlled Airspace") -> ControlledAirspace:
        """
        Add controlled airspace.
        
        Args:
            polygon_points: List of (x, y) points defining the boundary
            floor: Lower altitude limit
            ceiling: Upper altitude limit
            clearance_level: Required clearance level
            name: Airspace name
            
        Returns:
            Created controlled airspace
        """
        polygon = Polygon(polygon_points)
        
        airspace = ControlledAirspace(
            id=self.next_zone_id,
            polygon=polygon,
            floor=floor,
            ceiling=ceiling,
            clearance_level=clearance_level,
            name=name
        )
        
        self.controlled_airspace.append(airspace)
        self.next_zone_id += 1
        
        return airspace
    
    def is_position_valid(self, position: np.ndarray,
                         clearance_level: int = 0) -> bool:
        """
        Check if position violates airspace restrictions.
        
        Args:
            position: [x, y, z] position to check
            clearance_level: Aircraft clearance level
            
        Returns:
            True if position is valid
        """
        # Check altitude ceiling
        if position[2] > self.altitude_ceiling:
            return False
        
        # Check no-fly zones
        for zone in self.no_fly_zones:
            if zone.contains(position):
                return False
        
        # Check controlled airspace
        for airspace in self.controlled_airspace:
            if clearance_level < airspace.clearance_level:
                point = Point(position[0], position[1])
                if (airspace.polygon.contains(point) and
                    airspace.floor <= position[2] <= airspace.ceiling):
                    return False
        
        return True
    
    def get_threat_level(self, position: np.ndarray) -> float:
        """
        Get cumulative threat level at position.
        
        Args:
            position: [x, y, z] position
            
        Returns:
            Total threat level (can exceed 1.0 if multiple threats)
        """
        total_threat = 0.0
        
        for zone in self.threat_zones:
            total_threat += zone.get_threat_level(position)
        
        return total_threat
    
    def get_active_threats(self, position: np.ndarray) -> List[ThreatZone]:
        """
        Get list of threats affecting position.
        
        Args:
            position: [x, y, z] position
            
        Returns:
            List of active threat zones
        """
        active = []
        
        for zone in self.threat_zones:
            if zone.get_threat_level(position) > 0:
                active.append(zone)
        
        return active
    
    def get_safe_altitude_range(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get safe altitude range at horizontal position.
        
        Args:
            x: East coordinate
            y: North coordinate
            
        Returns:
            (min_safe_altitude, max_safe_altitude)
        """
        min_safe = 0.0
        max_safe = self.altitude_ceiling
        
        point = Point(x, y)
        
        # Check no-fly zones
        for zone in self.no_fly_zones:
            if zone.polygon.contains(point):
                # Adjust safe altitudes around no-fly zone
                if zone.min_altitude <= min_safe:
                    min_safe = max(min_safe, zone.max_altitude)
                if zone.max_altitude >= max_safe:
                    max_safe = min(max_safe, zone.min_altitude)
        
        # Check threat zones
        for threat in self.threat_zones:
            dx = x - threat.center[0]
            dy = y - threat.center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < threat.max_range:
                # Suggest flying above or below threat envelope
                if threat.max_altitude < max_safe:
                    min_safe = max(min_safe, threat.max_altitude)
                elif threat.min_altitude > min_safe:
                    max_safe = min(max_safe, threat.min_altitude)
        
        # Ensure valid range
        if min_safe >= max_safe:
            # No safe altitude available
            return (self.altitude_ceiling, self.altitude_ceiling)
        
        return (min_safe, max_safe)
    
    def find_safe_corridor(self, start: np.ndarray, end: np.ndarray,
                          num_samples: int = 10) -> Optional[List[float]]:
        """
        Find safe altitude corridor between two points.
        
        Args:
            start: Start position [x, y, z]
            end: End position [x, y, z]
            num_samples: Number of points to sample
            
        Returns:
            List of safe altitudes along path, or None if no safe path
        """
        safe_altitudes = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            min_alt, max_alt = self.get_safe_altitude_range(x, y)
            
            if min_alt >= max_alt:
                # No safe altitude at this point
                return None
            
            # Choose middle of safe range
            safe_alt = (min_alt + max_alt) / 2
            safe_altitudes.append(safe_alt)
        
        return safe_altitudes