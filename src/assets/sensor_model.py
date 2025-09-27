"""
Sensor model for aircraft detection and tracking.
Provides realistic sensor limitations and measurement noise.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.battlespace import Battlespace
from src.assets.aircraft_3dof import AircraftState


class SensorType(Enum):
    """Types of sensors"""
    RADAR = "radar"
    INFRARED = "infrared"
    VISUAL = "visual"
    RWR = "rwr"  # Radar Warning Receiver


@dataclass
class SensorConfig:
    """Sensor configuration parameters"""
    sensor_type: SensorType
    max_range: float  # meters
    fov_azimuth: float  # radians
    fov_elevation: float  # radians
    update_rate: float  # Hz
    position_error_std: float  # meters
    velocity_error_std: float  # m/s
    min_rcs_detection: float = 0.1  # Minimum RCS for detection (m²)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SensorConfig':
        """Create from configuration dictionary"""
        return cls(
            sensor_type=SensorType(config.get('type', 'radar')),
            max_range=config.get('max_range', 10000),
            fov_azimuth=np.radians(config.get('fov_azimuth', 120)),
            fov_elevation=np.radians(config.get('fov_elevation', 60)),
            update_rate=config.get('update_rate', 10),
            position_error_std=config.get('position_error', 50),
            velocity_error_std=config.get('velocity_error', 5),
            min_rcs_detection=config.get('min_rcs', 0.1)
        )


@dataclass
class TargetTrack:
    """Target track information from sensor"""
    track_id: str
    sensor_type: SensorType
    timestamp: float
    
    # Measured values
    position: np.ndarray  # [x, y, z]
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    
    # Measurement uncertainty
    position_covariance: Optional[np.ndarray] = None  # 3x3 matrix
    velocity_covariance: Optional[np.ndarray] = None  # 3x3 matrix
    
    # Derived values
    range: Optional[float] = None
    bearing: Optional[float] = None
    elevation: Optional[float] = None
    
    # Track quality
    detection_probability: float = 1.0
    snr: float = 10.0  # Signal-to-noise ratio
    track_quality: str = "good"  # good, degraded, poor
    
    # Classification
    classification: str = "unknown"  # unknown, friendly, hostile
    confidence: float = 0.5


class SensorModel:
    """
    Aircraft sensor system model with realistic limitations.
    """
    
    def __init__(self, sensor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize sensor model.
        
        Args:
            sensor_config: Sensor configuration dictionary
        """
        if sensor_config is None:
            # Default radar configuration
            self.config = SensorConfig(
                sensor_type=SensorType.RADAR,
                max_range=10000.0,
                fov_azimuth=np.radians(120),
                fov_elevation=np.radians(60),
                update_rate=10.0,
                position_error_std=50.0,
                velocity_error_std=5.0
            )
        else:
            self.config = SensorConfig.from_dict(sensor_config)
        
        # Track management
        self.tracks: Dict[str, TargetTrack] = {}
        self.next_track_id = 0
        self.last_update_time = 0.0
        
        # Detection parameters
        self.false_alarm_rate = 1e-6  # Per resolution cell
        self.min_snr_detection = 10.0  # dB
        
        # Atmospheric effects
        self.atmospheric_attenuation = 0.01  # dB/km
        
    def detect_targets(self, own_state: AircraftState, 
                      true_targets: Dict[str, AircraftState],
                      battlespace: Battlespace,
                      current_time: float) -> List[TargetTrack]:
        """
        Generate sensor detections with realistic effects.
        
        Args:
            own_state: Own aircraft state
            true_targets: Dictionary of true target states
            battlespace: Battlespace for LOS checks
            current_time: Current simulation time
            
        Returns:
            List of detected target tracks
        """
        detections = []
        
        # Check if it's time for sensor update
        if current_time - self.last_update_time < 1.0 / self.config.update_rate:
            return list(self.tracks.values())  # Return existing tracks
        
        self.last_update_time = current_time
        
        # Process each potential target
        for target_id, target_state in true_targets.items():
            track = self._process_target(
                own_state, target_state, target_id, battlespace, current_time
            )
            
            if track is not None:
                detections.append(track)
                self.tracks[target_id] = track
            elif target_id in self.tracks:
                # Lost track
                del self.tracks[target_id]
        
        # Add false alarms (optional)
        if np.random.random() < self.false_alarm_rate:
            false_track = self._generate_false_alarm(own_state, current_time)
            if false_track:
                detections.append(false_track)
        
        return detections
    
    def _process_target(self, own_state: AircraftState,
                       target_state: AircraftState,
                       target_id: str,
                       battlespace: Battlespace,
                       current_time: float) -> Optional[TargetTrack]:
        """
        Process a single target for detection.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            target_id: Target identifier
            battlespace: Battlespace for checks
            current_time: Current time
            
        Returns:
            Target track if detected, None otherwise
        """
        # Calculate relative geometry
        relative_pos = target_state.position - own_state.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Check range
        if range_to_target > self.config.max_range:
            return None
        
        # Calculate bearing and elevation
        bearing = np.arctan2(relative_pos[1], relative_pos[0])
        elevation = np.arctan2(relative_pos[2], 
                              np.linalg.norm(relative_pos[:2]))
        
        # Check field of view
        if not self._in_fov(own_state, bearing, elevation):
            return None
        
        # Check line of sight (terrain masking)
        if not battlespace.get_line_of_sight(own_state.position, 
                                            target_state.position):
            return None
        
        # Calculate detection probability
        p_detect = self._calculate_detection_probability(
            range_to_target, target_state
        )
        
        # Random detection based on probability
        if np.random.random() > p_detect:
            return None
        
        # Generate measurement with noise
        measured_position = self._add_measurement_noise(
            target_state.position, range_to_target
        )
        
        # Estimate velocity (with more noise)
        target_vel = target_state.get_velocity_vector()
        measured_velocity = target_vel + np.random.randn(3) * self.config.velocity_error_std
        
        # Create position covariance matrix
        pos_variance = (self.config.position_error_std * (1 + range_to_target/10000)) ** 2
        position_covariance = np.eye(3) * pos_variance
        
        # Create velocity covariance matrix
        vel_variance = self.config.velocity_error_std ** 2
        velocity_covariance = np.eye(3) * vel_variance
        
        # Determine track quality based on SNR
        snr = self._calculate_snr(range_to_target)
        if snr > 20:
            track_quality = "good"
        elif snr > 10:
            track_quality = "degraded"
        else:
            track_quality = "poor"
        
        # Create track
        track = TargetTrack(
            track_id=f"T{self.next_track_id:04d}",
            sensor_type=self.config.sensor_type,
            timestamp=current_time,
            position=measured_position,
            velocity=measured_velocity,
            position_covariance=position_covariance,
            velocity_covariance=velocity_covariance,
            range=range_to_target,
            bearing=bearing,
            elevation=elevation,
            detection_probability=p_detect,
            snr=snr,
            track_quality=track_quality,
            classification="unknown",
            confidence=0.5
        )
        
        self.next_track_id += 1
        
        return track
    
    def _in_fov(self, own_state: AircraftState, 
                bearing: float, elevation: float) -> bool:
        """
        Check if target is within field of view.
        
        Args:
            own_state: Own aircraft state
            bearing: Bearing to target
            elevation: Elevation to target
            
        Returns:
            True if in FOV
        """
        # Calculate relative bearing
        relative_bearing = bearing - own_state.heading
        while relative_bearing > np.pi:
            relative_bearing -= 2 * np.pi
        while relative_bearing < -np.pi:
            relative_bearing += 2 * np.pi
        
        # Check azimuth FOV
        if abs(relative_bearing) > self.config.fov_azimuth / 2:
            return False
        
        # Check elevation FOV
        if abs(elevation) > self.config.fov_elevation / 2:
            return False
        
        return True
    
    def _calculate_detection_probability(self, range_m: float,
                                        target_state: AircraftState) -> float:
        """
        Calculate probability of detection based on range and target.
        
        Args:
            range_m: Range to target in meters
            target_state: Target state
            
        Returns:
            Detection probability [0, 1]
        """
        # Simple model: probability decreases with range
        # P_d = P_0 * exp(-range/range_scale)
        
        # Base probability at zero range
        p_base = 0.95
        
        # Range scaling factor
        range_scale = self.config.max_range / 3  # 63% probability at 1/3 max range
        
        # Calculate base probability
        p_range = p_base * np.exp(-range_m / range_scale)
        
        # Aspect angle effects (head-on vs side-on)
        # Simplified: side aspect gives better detection
        
        # Altitude effects (ground clutter reduction)
        if target_state.position[2] < 100:  # Low altitude
            p_range *= 0.5  # Reduced detection due to ground clutter
        
        # Weather effects (if available)
        # Could integrate with battlespace weather system
        
        return np.clip(p_range, 0, 1)
    
    def _calculate_snr(self, range_m: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            range_m: Range to target
            
        Returns:
            SNR in dB
        """
        # Radar equation (simplified)
        # SNR = P_t * G² * σ * λ² / ((4π)³ * R⁴ * L)
        
        # Use simplified model
        snr_max = 40.0  # dB at minimum range
        range_km = range_m / 1000.0
        
        # R⁴ loss (40 log R)
        range_loss = 40 * np.log10(max(range_km, 0.1))
        
        # Atmospheric loss
        atm_loss = self.atmospheric_attenuation * range_km
        
        snr = snr_max - range_loss - atm_loss
        
        return max(snr, 0)
    
    def _add_measurement_noise(self, true_position: np.ndarray,
                              range_m: float) -> np.ndarray:
        """
        Add realistic measurement noise to position.
        
        Args:
            true_position: True position
            range_m: Range to target
            
        Returns:
            Measured position with noise
        """
        # Error increases with range
        range_factor = 1 + range_m / 10000  # Error doubles at 10km
        
        # Add Gaussian noise
        position_noise = np.random.randn(3) * self.config.position_error_std * range_factor
        
        # Range-dependent azimuth error
        azimuth_error = np.random.randn() * np.radians(1) * range_factor
        
        # Apply noise in sensor frame then transform back
        measured = true_position + position_noise
        
        return measured
    
    def _generate_false_alarm(self, own_state: AircraftState,
                            current_time: float) -> Optional[TargetTrack]:
        """
        Generate a false alarm track.
        
        Args:
            own_state: Own aircraft state
            current_time: Current time
            
        Returns:
            False alarm track or None
        """
        # Random position within sensor volume
        range_fa = np.random.uniform(1000, self.config.max_range)
        bearing_fa = own_state.heading + np.random.uniform(
            -self.config.fov_azimuth/2, self.config.fov_azimuth/2
        )
        elevation_fa = np.random.uniform(
            -self.config.fov_elevation/2, self.config.fov_elevation/2
        )
        
        # Convert to Cartesian
        x = own_state.position[0] + range_fa * np.cos(elevation_fa) * np.cos(bearing_fa)
        y = own_state.position[1] + range_fa * np.cos(elevation_fa) * np.sin(bearing_fa)
        z = own_state.position[2] + range_fa * np.sin(elevation_fa)
        
        position = np.array([x, y, z])
        
        # Random velocity
        velocity = np.random.randn(3) * 50  # Random velocity up to 50 m/s
        
        track = TargetTrack(
            track_id=f"F{self.next_track_id:04d}",  # F for false
            sensor_type=self.config.sensor_type,
            timestamp=current_time,
            position=position,
            velocity=velocity,
            range=range_fa,
            bearing=bearing_fa,
            elevation=elevation_fa,
            detection_probability=0.1,
            snr=5.0,
            track_quality="poor",
            classification="unknown",
            confidence=0.1
        )
        
        self.next_track_id += 1
        
        return track
    
    def get_sensor_coverage(self, own_state: AircraftState,
                           resolution: int = 20) -> np.ndarray:
        """
        Get sensor coverage volume for visualization.
        
        Args:
            own_state: Own aircraft state
            resolution: Number of points along each dimension
            
        Returns:
            Array of coverage boundary points
        """
        points = []
        
        # Generate points on the coverage boundary
        for az in np.linspace(-self.config.fov_azimuth/2, 
                             self.config.fov_azimuth/2, resolution):
            for el in np.linspace(-self.config.fov_elevation/2,
                                self.config.fov_elevation/2, resolution):
                # Point at max range
                bearing = own_state.heading + az
                
                x = own_state.position[0] + self.config.max_range * np.cos(el) * np.cos(bearing)
                y = own_state.position[1] + self.config.max_range * np.cos(el) * np.sin(bearing)
                z = own_state.position[2] + self.config.max_range * np.sin(el)
                
                points.append([x, y, z])
        
        return np.array(points)
    
    def reset(self):
        """Reset sensor model state"""
        self.tracks.clear()
        self.next_track_id = 0
        self.last_update_time = 0.0

    def detect(self, 
            sensor_position: np.ndarray,
            sensor_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray,
            battlespace=None,
            current_time: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Simple detection method for compatibility with scenario runner.
        
        Args:
            sensor_position: Sensor platform position [x, y, z]
            sensor_velocity: Sensor platform velocity vector
            target_position: Target position [x, y, z]
            target_velocity: Target velocity vector
            battlespace: Battlespace instance (optional)
            current_time: Current simulation time
            
        Returns:
            Detection dictionary or None if not detected
        """
        # Ensure inputs are numpy arrays
        sensor_position = np.array(sensor_position)
        target_position = np.array(target_position)
        sensor_velocity = np.array(sensor_velocity) if sensor_velocity is not None else np.zeros(3)
        target_velocity = np.array(target_velocity) if target_velocity is not None else np.zeros(3)
        
        # Calculate range
        range_vector = target_position - sensor_position
        range_to_target = np.linalg.norm(range_vector)
        
        # Check max range
        if range_to_target > self.config.max_range:
            return None
        
        # Calculate bearing and elevation
        bearing = np.arctan2(range_vector[1], range_vector[0])
        horizontal_range = np.sqrt(range_vector[0]**2 + range_vector[1]**2)
        elevation = np.arctan2(range_vector[2], horizontal_range) if horizontal_range > 0 else 0
        
        # Check field of view (simplified)
        # Assuming sensor is pointed forward along velocity vector or heading
        sensor_speed = np.linalg.norm(sensor_velocity[:2])
        if sensor_speed > 0.1:
            sensor_heading = np.arctan2(sensor_velocity[1], sensor_velocity[0])
        else:
            sensor_heading = 0  # Default forward
        
        relative_bearing = bearing - sensor_heading
        # Normalize to [-pi, pi]
        while relative_bearing > np.pi:
            relative_bearing -= 2 * np.pi
        while relative_bearing < -np.pi:
            relative_bearing += 2 * np.pi
        
        # Check FOV limits (fov_azimuth and fov_elevation are in degrees in config)
        fov_az_rad = np.radians(self.config.fov_azimuth) / 2
        fov_el_rad = np.radians(self.config.fov_elevation) / 2
        
        if abs(relative_bearing) > fov_az_rad:
            return None
        if abs(elevation) > fov_el_rad:
            return None
        
        # Check line of sight if battlespace provided
        if battlespace is not None:
            try:
                if hasattr(battlespace, 'get_line_of_sight'):
                    if not battlespace.get_line_of_sight(sensor_position, target_position):
                        return None
            except:
                pass  # Continue if LOS check fails
        
        # Simple detection probability based on range
        max_range = self.config.max_range
        detection_prob = max(0.0, 1.0 - (range_to_target / max_range) ** 2)
        
        # Random detection based on probability
        if np.random.random() > detection_prob:
            return None
        
        # Add measurement noise
        position_error = self.config.position_error_std * (1 + range_to_target / 10000)
        measured_position = target_position + np.random.randn(3) * position_error
        
        velocity_error = self.config.velocity_error_std
        measured_velocity = target_velocity + np.random.randn(3) * velocity_error
        
        # Return detection
        return {
            'detected': True,
            'position': measured_position.tolist(),  # Convert to list for JSON serialization
            'velocity': measured_velocity.tolist(),
            'range': float(range_to_target),
            'bearing': float(bearing),
            'elevation': float(elevation),
            'quality': float(detection_prob),
            'timestamp': float(current_time)
        }