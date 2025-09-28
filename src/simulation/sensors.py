"""
Sensor simulation module for interceptor guidance system.
Provides realistic sensor models with noise, occlusion, and detection probability.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy


class SensorType(Enum):
    """Types of sensors available"""
    RADAR = "radar"
    INFRARED = "infrared"
    CAMERA = "camera"
    LIDAR = "lidar"
    GPS = "gps"
    IMU = "imu"
    PERFECT = "perfect"  # Perfect sensor for testing


@dataclass
class Detection:
    """Single target detection from a sensor"""
    target_id: str
    timestamp: float
    sensor_type: SensorType
    
    # Measured values (with noise)
    measured_position: np.ndarray
    measured_velocity: Optional[np.ndarray] = None
    
    # Uncertainty estimates
    position_covariance: Optional[np.ndarray] = None
    velocity_covariance: Optional[np.ndarray] = None
    
    # Detection quality metrics
    signal_strength: float = 1.0
    confidence: float = 1.0
    
    # True values (for analysis only)
    true_position: Optional[np.ndarray] = None
    true_velocity: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary"""
        return {
            'target_id': self.target_id,
            'timestamp': self.timestamp,
            'sensor_type': self.sensor_type.value,
            'measured_position': self.measured_position.tolist(),
            'measured_velocity': self.measured_velocity.tolist() if self.measured_velocity is not None else None,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence
        }


@dataclass 
class SensorConfig:
    """Configuration for a sensor"""
    sensor_type: SensorType
    max_range: float = 10000.0  # meters
    min_range: float = 50.0      # meters
    fov_azimuth: float = 120.0   # degrees
    fov_elevation: float = 60.0  # degrees
    update_rate: float = 10.0    # Hz
    
    # Noise parameters
    position_noise_std: float = 10.0    # meters
    velocity_noise_std: float = 1.0     # m/s
    bearing_noise_std: float = 0.5      # degrees
    elevation_noise_std: float = 0.5    # degrees
    
    # Detection parameters
    detection_probability: float = 0.95
    false_alarm_rate: float = 0.001
    
    # Advanced parameters
    min_rcs: float = 0.1  # Minimum radar cross section for detection
    weather_degradation: bool = True
    terrain_occlusion: bool = True


class SensorBase:
    """Base class for all sensors"""
    
    def __init__(self, config: SensorConfig):
        """
        Initialize sensor.
        
        Args:
            config: Sensor configuration
        """
        self.config = config
        self.last_update_time = 0.0
        self.update_period = 1.0 / config.update_rate
        self.detections_history = []
        
    def can_detect(self, 
                   own_position: np.ndarray,
                   own_velocity: np.ndarray,
                   target_position: np.ndarray,
                   target_velocity: np.ndarray,
                   environment: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
        """
        Check if target can be detected.
        
        Returns:
            Tuple of (can_detect, detection_probability)
        """
        # Range check
        range_to_target = np.linalg.norm(target_position - own_position)
        if range_to_target < self.config.min_range or range_to_target > self.config.max_range:
            return False, 0.0
            
        # FOV check
        if not self._is_in_fov(own_position, target_position):
            return False, 0.0
            
        # Calculate detection probability based on range
        range_factor = 1.0 - (range_to_target / self.config.max_range) ** 2
        base_probability = self.config.detection_probability * range_factor
        
        # Environmental degradation
        if environment and self.config.weather_degradation:
            weather_factor = environment.get('visibility_factor', 1.0)
            base_probability *= weather_factor
            
        # Terrain occlusion check
        if environment and self.config.terrain_occlusion:
            if self._is_terrain_blocked(own_position, target_position, environment):
                return False, 0.0
                
        return True, base_probability
        
    def _is_in_fov(self, own_position: np.ndarray, target_position: np.ndarray) -> bool:
        """Check if target is within field of view"""
        # Vector to target
        to_target = target_position - own_position
        range_2d = np.linalg.norm(to_target[:2])
        
        if range_2d < 0.1:
            return True  # Target directly above/below
            
        # Azimuth angle (assume sensor points north/forward)
        bearing = np.arctan2(to_target[1], to_target[0])
        
        # Elevation angle
        elevation = np.arctan2(to_target[2], range_2d)
        
        # Check FOV limits (simplified - assumes forward-looking sensor)
        max_azimuth = np.radians(self.config.fov_azimuth / 2)
        max_elevation = np.radians(self.config.fov_elevation / 2)
        
        # For now, use simple FOV cone
        return (abs(bearing) <= max_azimuth and 
                abs(elevation) <= max_elevation)
                
    def _is_terrain_blocked(self, 
                           own_position: np.ndarray,
                           target_position: np.ndarray,
                           environment: Dict[str, Any]) -> bool:
        """Check if terrain blocks line of sight"""
        # Simplified terrain occlusion check
        if 'terrain_height' in environment:
            # Sample points along line of sight
            num_samples = 10
            for i in range(1, num_samples):
                t = i / num_samples
                sample_pos = own_position + t * (target_position - own_position)
                
                # Get terrain height at sample position
                terrain_height = environment.get('terrain_height', 0)
                if isinstance(terrain_height, callable):
                    terrain_height = terrain_height(sample_pos[0], sample_pos[1])
                    
                if sample_pos[2] < terrain_height:
                    return True
                    
        return False
        
    def add_measurement_noise(self, 
                            true_position: np.ndarray,
                            true_velocity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Add measurement noise to true values"""
        # Position noise
        position_noise = np.random.randn(3) * self.config.position_noise_std
        measured_position = true_position + position_noise
        
        # Velocity noise (if velocity is measured)
        measured_velocity = None
        if true_velocity is not None:
            velocity_noise = np.random.randn(3) * self.config.velocity_noise_std
            measured_velocity = true_velocity + velocity_noise
            
        return measured_position, measured_velocity


class Radar(SensorBase):
    """Radar sensor model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize radar sensor"""
        if config is None:
            config = {}
            
        # Create SensorConfig from dict
        sensor_config = SensorConfig(
            sensor_type=SensorType.RADAR,
            max_range=config.get('max_range', 15000.0),
            min_range=config.get('min_range', 100.0),
            fov_azimuth=config.get('fov_azimuth', 120.0),
            fov_elevation=config.get('fov_elevation', 60.0),
            update_rate=config.get('update_rate', 10.0),
            position_noise_std=config.get('position_noise_std', 20.0),
            velocity_noise_std=config.get('velocity_noise_std', 2.0),
            detection_probability=config.get('detection_probability', 0.9)
        )
        super().__init__(sensor_config)
        
        # Radar-specific parameters
        self.can_measure_velocity = True
        self.min_doppler_velocity = config.get('min_doppler_velocity', 5.0)  # m/s
        
    def get_detection(self,
                     own_state: Dict[str, Any],
                     target_state: Dict[str, Any],
                     environment: Optional[Dict[str, Any]] = None) -> Optional[Detection]:
        """Get radar detection of a target"""
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state.get('velocity', [0, 0, 0]))
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Check if can detect
        can_detect, prob = self.can_detect(own_pos, own_vel, target_pos, target_vel, environment)
        
        if not can_detect:
            return None
            
        # Random detection based on probability
        if np.random.random() > prob:
            return None
            
        # Add measurement noise
        measured_pos, measured_vel = self.add_measurement_noise(target_pos, target_vel)
        
        # Calculate signal strength based on range and RCS
        range_to_target = np.linalg.norm(target_pos - own_pos)
        rcs = target_state.get('rcs', 1.0)  # Radar cross section
        signal_strength = (rcs / (range_to_target ** 4)) * 1e12  # Simplified radar equation
        
        return Detection(
            target_id=target_state.get('id', 'unknown'),
            timestamp=own_state.get('time', 0.0),
            sensor_type=SensorType.RADAR,
            measured_position=measured_pos,
            measured_velocity=measured_vel,
            position_covariance=np.eye(3) * (self.config.position_noise_std ** 2),
            velocity_covariance=np.eye(3) * (self.config.velocity_noise_std ** 2),
            signal_strength=min(1.0, signal_strength),
            confidence=prob,
            true_position=target_pos,
            true_velocity=target_vel
        )


class InfraredSensor(SensorBase):
    """Infrared/thermal sensor model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IR sensor"""
        if config is None:
            config = {}
            
        sensor_config = SensorConfig(
            sensor_type=SensorType.INFRARED,
            max_range=config.get('max_range', 8000.0),
            min_range=config.get('min_range', 50.0),
            fov_azimuth=config.get('fov_azimuth', 60.0),
            fov_elevation=config.get('fov_elevation', 45.0),
            update_rate=config.get('update_rate', 30.0),
            position_noise_std=config.get('position_noise_std', 15.0),
            detection_probability=config.get('detection_probability', 0.85)
        )
        super().__init__(sensor_config)
        
        # IR-specific parameters
        self.can_measure_velocity = False
        self.min_heat_signature = config.get('min_heat_signature', 0.1)
        
    def get_detection(self,
                     own_state: Dict[str, Any],
                     target_state: Dict[str, Any],
                     environment: Optional[Dict[str, Any]] = None) -> Optional[Detection]:
        """Get IR detection of a target"""
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state.get('velocity', [0, 0, 0]))
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Check heat signature
        heat_signature = target_state.get('heat_signature', 1.0)
        if heat_signature < self.min_heat_signature:
            return None
            
        # Check if can detect
        can_detect, prob = self.can_detect(own_pos, own_vel, target_pos, target_vel, environment)
        
        if not can_detect:
            return None
            
        # Modify probability based on heat signature
        prob *= min(1.0, heat_signature)
        
        # Random detection
        if np.random.random() > prob:
            return None
            
        # IR typically doesn't measure velocity directly
        measured_pos, _ = self.add_measurement_noise(target_pos, None)
        
        range_to_target = np.linalg.norm(target_pos - own_pos)
        signal_strength = heat_signature / (range_to_target / 1000.0) ** 2
        
        return Detection(
            target_id=target_state.get('id', 'unknown'),
            timestamp=own_state.get('time', 0.0),
            sensor_type=SensorType.INFRARED,
            measured_position=measured_pos,
            measured_velocity=None,
            position_covariance=np.eye(3) * (self.config.position_noise_std ** 2),
            signal_strength=min(1.0, signal_strength),
            confidence=prob,
            true_position=target_pos
        )


class PerfectSensor(SensorBase):
    """Perfect sensor for testing (no noise, perfect detection)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize perfect sensor"""
        if config is None:
            config = {}
            
        sensor_config = SensorConfig(
            sensor_type=SensorType.PERFECT,
            max_range=config.get('max_range', 50000.0),
            min_range=0.0,
            fov_azimuth=360.0,
            fov_elevation=180.0,
            update_rate=config.get('update_rate', 50.0),
            position_noise_std=0.0,
            velocity_noise_std=0.0,
            detection_probability=1.0,
            false_alarm_rate=0.0
        )
        super().__init__(sensor_config)
        
    def get_detection(self,
                     own_state: Dict[str, Any],
                     target_state: Dict[str, Any],
                     environment: Optional[Dict[str, Any]] = None) -> Optional[Detection]:
        """Get perfect detection of a target"""
        own_pos = np.array(own_state['position'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Check range only
        range_to_target = np.linalg.norm(target_pos - own_pos)
        if range_to_target > self.config.max_range:
            return None
            
        return Detection(
            target_id=target_state.get('id', 'unknown'),
            timestamp=own_state.get('time', 0.0),
            sensor_type=SensorType.PERFECT,
            measured_position=target_pos.copy(),
            measured_velocity=target_vel.copy(),
            position_covariance=np.zeros((3, 3)),
            velocity_covariance=np.zeros((3, 3)),
            signal_strength=1.0,
            confidence=1.0,
            true_position=target_pos,
            true_velocity=target_vel
        )


class SensorSuite:
    """Collection of multiple sensors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sensor suite.
        
        Args:
            config: Configuration dictionary with sensor specifications
        """
        self.sensors = {}
        
        if config is None:
            # Default sensor suite
            self.sensors['radar'] = Radar()
            self.sensors['infrared'] = InfraredSensor()
        else:
            # Create sensors from config
            for sensor_name, sensor_config in config.items():
                sensor_type = sensor_config.get('type', 'radar')
                
                if sensor_type == 'radar':
                    self.sensors[sensor_name] = Radar(sensor_config)
                elif sensor_type == 'infrared':
                    self.sensors[sensor_name] = InfraredSensor(sensor_config)
                elif sensor_type == 'perfect':
                    self.sensors[sensor_name] = PerfectSensor(sensor_config)
                else:
                    print(f"Warning: Unknown sensor type '{sensor_type}'")
                    
    def get_all_detections(self,
                          own_state: Dict[str, Any],
                          targets: Dict[str, Dict[str, Any]],
                          environment: Optional[Dict[str, Any]] = None) -> Dict[str, List[Detection]]:
        """
        Get detections from all sensors.
        
        Returns:
            Dictionary mapping sensor_name to list of detections
        """
        all_detections = {}
        
        for sensor_name, sensor in self.sensors.items():
            detections = []
            
            for target_id, target_state in targets.items():
                detection = sensor.get_detection(own_state, target_state, environment)
                if detection:
                    detections.append(detection)
                    
            all_detections[sensor_name] = detections
            
        return all_detections
        
    def get_fused_detections(self,
                            own_state: Dict[str, Any],
                            targets: Dict[str, Dict[str, Any]],
                            environment: Optional[Dict[str, Any]] = None) -> List[Detection]:
        """
        Get fused detections from all sensors.
        Simple fusion: uses best detection per target.
        """
        all_detections = self.get_all_detections(own_state, targets, environment)
        
        # Group detections by target
        target_detections = {}
        for sensor_name, detections in all_detections.items():
            for detection in detections:
                if detection.target_id not in target_detections:
                    target_detections[detection.target_id] = []
                target_detections[detection.target_id].append(detection)
                
        # Fuse detections (simple: choose highest confidence)
        fused_detections = []
        for target_id, detections in target_detections.items():
            best_detection = max(detections, key=lambda d: d.confidence)
            
            # If multiple sensors detect, improve confidence
            if len(detections) > 1:
                best_detection.confidence = min(1.0, best_detection.confidence * 1.2)
                
            fused_detections.append(best_detection)
            
        return fused_detections


# Simplified sensor interface for backward compatibility
class SimpleSensor:
    """Simplified sensor interface"""
    
    def __init__(self, max_range=10000, fov=120):
        self.max_range = max_range
        self.fov = fov
        
    def detect(self, own_pos, target_pos):
        """Simple detection check"""
        range_to_target = np.linalg.norm(np.array(target_pos) - np.array(own_pos))
        return range_to_target <= self.max_range