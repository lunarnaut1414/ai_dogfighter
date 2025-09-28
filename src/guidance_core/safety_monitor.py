# src/guidance_core/safety_monitor.py
"""
Safety monitoring and constraint enforcement system.
Ensures safe operation within defined boundaries and limits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class SafetyLevel(Enum):
    """Safety alert levels"""
    NOMINAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class ConstraintType(Enum):
    """Types of safety constraints"""
    ALTITUDE_MIN = "altitude_min"
    ALTITUDE_MAX = "altitude_max"
    SPEED_MIN = "speed_min"
    SPEED_MAX = "speed_max"
    ACCELERATION_MAX = "acceleration_max"
    BANK_ANGLE_MAX = "bank_angle_max"
    FUEL_MIN = "fuel_min"
    BOUNDARY = "boundary"
    NO_FLY_ZONE = "no_fly_zone"
    TERRAIN_CLEARANCE = "terrain_clearance"
    COLLISION_AVOIDANCE = "collision_avoidance"


@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    constraint_type: ConstraintType
    severity: SafetyLevel
    value: float
    limit: float
    message: str
    timestamp: float
    position: Optional[np.ndarray] = None


@dataclass
class SafetyEnvelope:
    """Defines safe operating envelope"""
    # Altitude limits (meters)
    altitude_min: float = 100.0
    altitude_max: float = 10000.0
    altitude_warning_margin: float = 50.0
    
    # Speed limits (m/s)
    speed_min: float = 20.0
    speed_max: float = 80.0
    speed_warning_margin: float = 5.0
    
    # Acceleration limits (m/s²)
    acceleration_max: float = 30.0  # 3g
    acceleration_warning: float = 25.0
    
    # Bank angle limits (degrees)
    bank_angle_max: float = 60.0
    bank_angle_warning: float = 50.0
    
    # Fuel limits (fraction)
    fuel_emergency: float = 0.1  # 10%
    fuel_warning: float = 0.2  # 20%
    fuel_rtb: float = 0.3  # 30%
    
    # Boundary limits (meters from origin)
    boundary_radius: float = 25000.0
    boundary_warning_margin: float = 1000.0
    
    # Terrain clearance (meters)
    terrain_clearance_min: float = 100.0
    terrain_clearance_warning: float = 200.0
    
    # Collision avoidance (meters)
    collision_range_min: float = 50.0
    collision_range_warning: float = 200.0
    collision_time_warning: float = 5.0  # seconds


class SafetyMonitor:
    """
    Monitors aircraft state for safety violations and provides corrective actions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize safety monitor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.envelope = SafetyEnvelope()
        
        # Override envelope with config
        if config:
            for key, value in config.items():
                if hasattr(self.envelope, key):
                    setattr(self.envelope, key, value)
                    
        # Violation history
        self.violations: List[SafetyViolation] = []
        self.max_violation_history = 100
        
        # No-fly zones
        self.no_fly_zones: List[Dict] = []
        
        # Current safety state
        self.current_safety_level = SafetyLevel.NOMINAL
        self.active_violations: List[SafetyViolation] = []
        
        # Statistics
        self.total_violations = 0
        self.violation_counts = {level: 0 for level in SafetyLevel}
        
    def check_all_constraints(self, state: Dict, environment: Optional[Dict] = None) -> Tuple[SafetyLevel, List[SafetyViolation]]:
        """
        Check all safety constraints.
        
        Args:
            state: Aircraft state dictionary
            environment: Environmental conditions
            
        Returns:
            Tuple of (overall safety level, list of violations)
        """
        violations = []
        current_time = time.time()
        
        # Check altitude constraints
        altitude_violations = self._check_altitude(state, current_time)
        violations.extend(altitude_violations)
        
        # Check speed constraints
        speed_violations = self._check_speed(state, current_time)
        violations.extend(speed_violations)
        
        # Check acceleration constraints
        accel_violations = self._check_acceleration(state, current_time)
        violations.extend(accel_violations)
        
        # Check bank angle constraints
        bank_violations = self._check_bank_angle(state, current_time)
        violations.extend(bank_violations)
        
        # Check fuel constraints
        fuel_violations = self._check_fuel(state, current_time)
        violations.extend(fuel_violations)
        
        # Check boundary constraints
        boundary_violations = self._check_boundary(state, current_time)
        violations.extend(boundary_violations)
        
        # Check no-fly zones
        nfz_violations = self._check_no_fly_zones(state, current_time)
        violations.extend(nfz_violations)
        
        # Check terrain clearance if environment provided
        if environment and 'terrain_height' in environment:
            terrain_violations = self._check_terrain_clearance(state, environment, current_time)
            violations.extend(terrain_violations)
            
        # Check collision avoidance if other aircraft provided
        if environment and 'other_aircraft' in environment:
            collision_violations = self._check_collision_avoidance(state, environment, current_time)
            violations.extend(collision_violations)
            
        # Update violation history
        self._update_violation_history(violations)
        
        # Determine overall safety level
        if violations:
            max_severity = max(v.severity for v in violations)
            self.current_safety_level = max_severity
        else:
            self.current_safety_level = SafetyLevel.NOMINAL
            
        self.active_violations = violations
        return self.current_safety_level, violations
        
    def _check_altitude(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check altitude constraints"""
        violations = []
        altitude = state['position'][2]
        
        # Check minimum altitude
        if altitude < self.envelope.altitude_min:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.ALTITUDE_MIN,
                severity=SafetyLevel.CRITICAL,
                value=altitude,
                limit=self.envelope.altitude_min,
                message=f"CRITICAL: Altitude {altitude:.0f}m below minimum {self.envelope.altitude_min:.0f}m",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
        elif altitude < self.envelope.altitude_min + self.envelope.altitude_warning_margin:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.ALTITUDE_MIN,
                severity=SafetyLevel.WARNING,
                value=altitude,
                limit=self.envelope.altitude_min,
                message=f"WARNING: Approaching minimum altitude",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
            
        # Check maximum altitude
        if altitude > self.envelope.altitude_max:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.ALTITUDE_MAX,
                severity=SafetyLevel.WARNING,
                value=altitude,
                limit=self.envelope.altitude_max,
                message=f"WARNING: Altitude {altitude:.0f}m exceeds maximum {self.envelope.altitude_max:.0f}m",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
        elif altitude > self.envelope.altitude_max - self.envelope.altitude_warning_margin:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.ALTITUDE_MAX,
                severity=SafetyLevel.CAUTION,
                value=altitude,
                limit=self.envelope.altitude_max,
                message=f"CAUTION: Approaching maximum altitude",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
            
        return violations
        
    def _check_speed(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check speed constraints"""
        violations = []
        velocity = np.array(state['velocity'])
        speed = np.linalg.norm(velocity)
        
        # Check minimum speed (stall)
        if speed < self.envelope.speed_min:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.SPEED_MIN,
                severity=SafetyLevel.CRITICAL,
                value=speed,
                limit=self.envelope.speed_min,
                message=f"CRITICAL: Speed {speed:.1f}m/s below stall speed {self.envelope.speed_min:.1f}m/s",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
        elif speed < self.envelope.speed_min + self.envelope.speed_warning_margin:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.SPEED_MIN,
                severity=SafetyLevel.WARNING,
                value=speed,
                limit=self.envelope.speed_min,
                message=f"WARNING: Approaching stall speed",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
            
        # Check maximum speed
        if speed > self.envelope.speed_max:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.SPEED_MAX,
                severity=SafetyLevel.WARNING,
                value=speed,
                limit=self.envelope.speed_max,
                message=f"WARNING: Speed {speed:.1f}m/s exceeds Vmax {self.envelope.speed_max:.1f}m/s",
                timestamp=timestamp,
                position=np.array(state['position'])
            ))
            
        return violations
        
    def _check_acceleration(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check acceleration constraints"""
        violations = []
        
        if 'acceleration' in state:
            accel = np.array(state['acceleration'])
            accel_magnitude = np.linalg.norm(accel)
            
            if accel_magnitude > self.envelope.acceleration_max:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.ACCELERATION_MAX,
                    severity=SafetyLevel.CRITICAL,
                    value=accel_magnitude,
                    limit=self.envelope.acceleration_max,
                    message=f"CRITICAL: Acceleration {accel_magnitude/9.81:.1f}g exceeds limit {self.envelope.acceleration_max/9.81:.1f}g",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
            elif accel_magnitude > self.envelope.acceleration_warning:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.ACCELERATION_MAX,
                    severity=SafetyLevel.WARNING,
                    value=accel_magnitude,
                    limit=self.envelope.acceleration_max,
                    message=f"WARNING: High acceleration {accel_magnitude/9.81:.1f}g",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
                
        return violations
        
    def _check_bank_angle(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check bank angle constraints"""
        violations = []
        
        if 'bank_angle' in state:
            bank_angle = abs(state['bank_angle']) * 180/np.pi  # Convert to degrees
            
            if bank_angle > self.envelope.bank_angle_max:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.BANK_ANGLE_MAX,
                    severity=SafetyLevel.WARNING,
                    value=bank_angle,
                    limit=self.envelope.bank_angle_max,
                    message=f"WARNING: Bank angle {bank_angle:.0f}° exceeds limit {self.envelope.bank_angle_max:.0f}°",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
            elif bank_angle > self.envelope.bank_angle_warning:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.BANK_ANGLE_MAX,
                    severity=SafetyLevel.CAUTION,
                    value=bank_angle,
                    limit=self.envelope.bank_angle_max,
                    message=f"CAUTION: High bank angle {bank_angle:.0f}°",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
                
        return violations
        
    def _check_fuel(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check fuel constraints"""
        violations = []
        
        if 'fuel_fraction' in state:
            fuel = state['fuel_fraction']
            
            if fuel < self.envelope.fuel_emergency:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.FUEL_MIN,
                    severity=SafetyLevel.EMERGENCY,
                    value=fuel,
                    limit=self.envelope.fuel_emergency,
                    message=f"EMERGENCY: Fuel critical {fuel:.1%}",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
            elif fuel < self.envelope.fuel_warning:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.FUEL_MIN,
                    severity=SafetyLevel.WARNING,
                    value=fuel,
                    limit=self.envelope.fuel_warning,
                    message=f"WARNING: Low fuel {fuel:.1%}",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
            elif fuel < self.envelope.fuel_rtb:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.FUEL_MIN,
                    severity=SafetyLevel.CAUTION,
                    value=fuel,
                    limit=self.envelope.fuel_rtb,
                    message=f"CAUTION: Fuel at RTB threshold {fuel:.1%}",
                    timestamp=timestamp,
                    position=np.array(state['position'])
                ))
                
        return violations
        
    def _check_boundary(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check boundary constraints"""
        violations = []
        position = np.array(state['position'])
        distance_from_origin = np.linalg.norm(position[:2])  # Horizontal distance
        
        if distance_from_origin > self.envelope.boundary_radius:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.BOUNDARY,
                severity=SafetyLevel.CRITICAL,
                value=distance_from_origin,
                limit=self.envelope.boundary_radius,
                message=f"CRITICAL: Outside operational boundary",
                timestamp=timestamp,
                position=position
            ))
        elif distance_from_origin > self.envelope.boundary_radius - self.envelope.boundary_warning_margin:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.BOUNDARY,
                severity=SafetyLevel.WARNING,
                value=distance_from_origin,
                limit=self.envelope.boundary_radius,
                message=f"WARNING: Approaching boundary",
                timestamp=timestamp,
                position=position
            ))
            
        return violations
        
    def _check_no_fly_zones(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check no-fly zone violations"""
        violations = []
        position = np.array(state['position'])
        
        for nfz in self.no_fly_zones:
            nfz_center = np.array(nfz['center'])
            nfz_radius = nfz['radius']
            
            distance = np.linalg.norm(position - nfz_center)
            
            if distance < nfz_radius:
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.NO_FLY_ZONE,
                    severity=SafetyLevel.CRITICAL,
                    value=distance,
                    limit=nfz_radius,
                    message=f"CRITICAL: Inside no-fly zone {nfz.get('name', 'unnamed')}",
                    timestamp=timestamp,
                    position=position
                ))
            elif distance < nfz_radius + 500:  # 500m warning margin
                violations.append(SafetyViolation(
                    constraint_type=ConstraintType.NO_FLY_ZONE,
                    severity=SafetyLevel.WARNING,
                    value=distance,
                    limit=nfz_radius,
                    message=f"WARNING: Approaching no-fly zone {nfz.get('name', 'unnamed')}",
                    timestamp=timestamp,
                    position=position
                ))
                
        return violations
        
    def _check_terrain_clearance(self, state: Dict, environment: Dict, 
                                timestamp: float) -> List[SafetyViolation]:
        """Check terrain clearance"""
        violations = []
        position = np.array(state['position'])
        altitude = position[2]
        terrain_height = environment.get('terrain_height', 0)
        
        clearance = altitude - terrain_height
        
        if clearance < self.envelope.terrain_clearance_min:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.TERRAIN_CLEARANCE,
                severity=SafetyLevel.CRITICAL,
                value=clearance,
                limit=self.envelope.terrain_clearance_min,
                message=f"CRITICAL: Terrain clearance {clearance:.0f}m below minimum",
                timestamp=timestamp,
                position=position
            ))
        elif clearance < self.envelope.terrain_clearance_warning:
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.TERRAIN_CLEARANCE,
                severity=SafetyLevel.WARNING,
                value=clearance,
                limit=self.envelope.terrain_clearance_warning,
                message=f"WARNING: Low terrain clearance {clearance:.0f}m",
                timestamp=timestamp,
                position=position
            ))
            
        return violations
        
    def _check_collision_avoidance(self, state: Dict, environment: Dict, 
                                  timestamp: float) -> List[SafetyViolation]:
        """Check for potential collisions"""
        violations = []
        own_pos = np.array(state['position'])
        own_vel = np.array(state['velocity'])
        
        for other in environment.get('other_aircraft', []):
            other_pos = np.array(other['position'])
            other_vel = np.array(other.get('velocity', [0, 0, 0]))
            
            # Current separation
            separation = np.linalg.norm(other_pos - own_pos)
            
            # Time to closest point of approach
            rel_pos = other_pos - own_pos
            rel_vel = other_vel - own_vel
            
            if np.linalg.norm(rel_vel) > 0:
                tcpa = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
                
                if 0 < tcpa < self.envelope.collision_time_warning:
                    # Predicted minimum separation
                    min_separation = np.linalg.norm(rel_pos + rel_vel * tcpa)
                    
                    if min_separation < self.envelope.collision_range_min:
                        violations.append(SafetyViolation(
                            constraint_type=ConstraintType.COLLISION_AVOIDANCE,
                            severity=SafetyLevel.EMERGENCY,
                            value=min_separation,
                            limit=self.envelope.collision_range_min,
                            message=f"EMERGENCY: Collision alert! CPA {min_separation:.0f}m in {tcpa:.1f}s",
                            timestamp=timestamp,
                            position=own_pos
                        ))
                    elif min_separation < self.envelope.collision_range_warning:
                        violations.append(SafetyViolation(
                            constraint_type=ConstraintType.COLLISION_AVOIDANCE,
                            severity=SafetyLevel.WARNING,
                            value=min_separation,
                            limit=self.envelope.collision_range_warning,
                            message=f"WARNING: Close approach {min_separation:.0f}m in {tcpa:.1f}s",
                            timestamp=timestamp,
                            position=own_pos
                        ))
                        
        return violations
        
    def _update_violation_history(self, violations: List[SafetyViolation]):
        """Update violation history and statistics"""
        for violation in violations:
            self.violations.append(violation)
            self.total_violations += 1
            self.violation_counts[violation.severity] += 1
            
        # Trim history if too long
        if len(self.violations) > self.max_violation_history:
            self.violations = self.violations[-self.max_violation_history:]
            
    def get_corrective_action(self, violations: List[SafetyViolation]) -> Dict[str, Any]:
        """
        Recommend corrective action based on violations.
        
        Args:
            violations: List of current violations
            
        Returns:
            Corrective action recommendation
        """
        if not violations:
            return {'action': 'none', 'message': 'All systems nominal'}
            
        # Find most severe violation
        most_severe = max(violations, key=lambda v: v.severity.value)
        
        action = {
            'action': 'correct',
            'severity': most_severe.severity.name,
            'violations': len(violations)
        }
        
        # Recommend specific actions based on violation type
        if most_severe.constraint_type == ConstraintType.ALTITUDE_MIN:
            action['command'] = 'climb'
            action['target_altitude'] = self.envelope.altitude_min + 200
            action['message'] = "CLIMB immediately"
            
        elif most_severe.constraint_type == ConstraintType.ALTITUDE_MAX:
            action['command'] = 'descend'
            action['target_altitude'] = self.envelope.altitude_max - 200
            action['message'] = "DESCEND to safe altitude"
            
        elif most_severe.constraint_type == ConstraintType.SPEED_MIN:
            action['command'] = 'increase_speed'
            action['target_speed'] = self.envelope.speed_min + 10
            action['message'] = "INCREASE speed to avoid stall"
            
        elif most_severe.constraint_type == ConstraintType.SPEED_MAX:
            action['command'] = 'reduce_speed'
            action['target_speed'] = self.envelope.speed_max - 10
            action['message'] = "REDUCE speed to safe level"
            
        elif most_severe.constraint_type == ConstraintType.FUEL_MIN:
            action['command'] = 'rtb'
            action['message'] = "RTB immediately - fuel critical"
            
        elif most_severe.constraint_type == ConstraintType.COLLISION_AVOIDANCE:
            action['command'] = 'evade'
            action['message'] = "EVADE - collision threat"
            
        elif most_severe.constraint_type == ConstraintType.NO_FLY_ZONE:
            action['command'] = 'exit_nfz'
            action['message'] = "EXIT no-fly zone immediately"
            
        elif most_severe.constraint_type == ConstraintType.BOUNDARY:
            action['command'] = 'turn_back'
            action['message'] = "TURN back to operational area"
            
        elif most_severe.constraint_type == ConstraintType.TERRAIN_CLEARANCE:
            action['command'] = 'climb'
            action['target_altitude'] = most_severe.value + self.envelope.terrain_clearance_warning + 100
            action['message'] = "CLIMB - terrain proximity"
            
        return action
        
    def add_no_fly_zone(self, center: List[float], radius: float, name: str = ""):
        """Add a no-fly zone"""
        self.no_fly_zones.append({
            'center': center,
            'radius': radius,
            'name': name
        })
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics"""
        return {
            'current_level': self.current_safety_level.name,
            'active_violations': len(self.active_violations),
            'total_violations': self.total_violations,
            'violation_breakdown': self.violation_counts,
            'history_size': len(self.violations)
        }