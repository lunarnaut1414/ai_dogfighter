"""
Flight controller for autonomous aircraft behaviors.
Provides different control modes for target aircraft.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass

from src.assets.aircraft_3dof import Aircraft3DOF, AircraftState


class BehaviorMode(Enum):
    """Flight behavior modes"""
    MANUAL = "manual"
    WAYPOINT = "waypoint"
    ORBIT = "orbit"
    PURSUIT = "pursuit"
    EVADE = "evade"
    FORMATION = "formation"
    PATROL = "patrol"
    RTB = "rtb"  # Return to base


@dataclass
class ControlCommand:
    """Control command output"""
    bank_angle: float  # radians
    throttle: float  # [0, 1]
    mode: str = ""
    
    def limit(self, max_bank: float, max_throttle: float = 1.0):
        """Apply control limits"""
        self.bank_angle = np.clip(self.bank_angle, -max_bank, max_bank)
        self.throttle = np.clip(self.throttle, 0.0, max_throttle)
        return self


class FlightController:
    """
    Autonomous flight controller for aircraft behaviors.
    Provides various control modes for different flight patterns.
    """
    
    def __init__(self, aircraft_config: Dict[str, Any]):
        """
        Initialize flight controller.
        
        Args:
            aircraft_config: Aircraft configuration dictionary
        """
        self.config = aircraft_config
        
        # Control gains (tunable)
        self.k_heading = 1.0  # Heading control gain
        self.k_altitude = 0.1  # Altitude control gain  
        self.k_speed = 0.05  # Speed control gain
        
        # Behavior state
        self.mode = BehaviorMode.MANUAL
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0
        self.orbit_center: Optional[np.ndarray] = None
        self.orbit_radius = 1000.0  # meters
        self.formation_lead: Optional[str] = None
        self.formation_offset: np.ndarray = np.zeros(3)
        
        # Control limits from config
        perf = aircraft_config.get('performance', {})
        ctrl = aircraft_config.get('control', {})
        self.max_bank = np.radians(ctrl.get('bank_angle_max', 60))
        self.max_climb_rate = perf.get('climb_rate_max', 10)
        self.cruise_speed = perf.get('v_cruise', 50)
        
        # Evasion parameters
        self.evasion_distance = 2000.0  # Start evasion at this range
        self.evasion_aggressiveness = 1.0  # 0-1, higher = more aggressive
        
    def compute_commands(self, state: AircraftState, 
                        target: Optional[np.ndarray] = None,
                        threat: Optional[np.ndarray] = None) -> ControlCommand:
        """
        Compute control commands based on current mode.
        
        Args:
            state: Current aircraft state
            target: Target position for some modes
            threat: Threat position for evasion
            
        Returns:
            Control commands
        """
        if self.mode == BehaviorMode.MANUAL:
            # No automatic control
            return ControlCommand(0.0, state.throttle, "manual")
            
        elif self.mode == BehaviorMode.WAYPOINT:
            if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
                return self.waypoint_guidance(state, self.waypoints[self.current_waypoint_idx])
            return ControlCommand(0.0, state.throttle, "waypoint_idle")
            
        elif self.mode == BehaviorMode.ORBIT:
            if self.orbit_center is not None:
                return self.orbit_guidance(state, self.orbit_center)
            elif target is not None:
                return self.orbit_guidance(state, target)
            return ControlCommand(0.0, state.throttle, "orbit_idle")
            
        elif self.mode == BehaviorMode.PURSUIT:
            if target is not None:
                return self.pursuit_guidance(state, target)
            return ControlCommand(0.0, state.throttle, "pursuit_idle")
            
        elif self.mode == BehaviorMode.EVADE:
            if threat is not None:
                return self.evasive_guidance(state, threat)
            return ControlCommand(0.0, state.throttle, "evade_idle")
            
        elif self.mode == BehaviorMode.PATROL:
            return self.patrol_guidance(state)
            
        elif self.mode == BehaviorMode.RTB:
            # Return to first waypoint (assumed to be base)
            if self.waypoints:
                return self.waypoint_guidance(state, self.waypoints[0])
            return ControlCommand(0.0, state.throttle, "rtb_idle")
            
        return ControlCommand(0.0, state.throttle, "unknown")
        
    def waypoint_guidance(self, state: AircraftState, 
                         waypoint: np.ndarray) -> ControlCommand:
        """
        Navigate to a waypoint.
        
        Args:
            state: Current aircraft state
            waypoint: Target waypoint [x, y, z]
            
        Returns:
            Control commands
        """
        # Vector to waypoint
        delta = waypoint - state.position
        horizontal_dist = np.linalg.norm(delta[:2])
        
        # Check if waypoint reached
        if horizontal_dist < 50.0:  # Within 50m
            self.next_waypoint()
            return ControlCommand(0.0, state.throttle, "waypoint_reached")
            
        # Desired heading to waypoint
        desired_heading = np.arctan2(delta[1], delta[0])
        
        # Heading error with wrapping
        heading_error = self._wrap_angle(desired_heading - state.heading)
        
        # Bank angle command (proportional control)
        bank_cmd = self.k_heading * heading_error
        
        # Altitude control
        altitude_error = waypoint[2] - state.position[2]
        desired_climb_rate = np.clip(
            altitude_error * self.k_altitude,
            -self.max_climb_rate,
            self.max_climb_rate
        )
        
        # Adjust bank for climb/descent (reduce bank when climbing)
        if abs(desired_climb_rate) > 2.0:
            bank_cmd *= 0.7  # Reduce bank angle during climbs
            
        # Speed control
        speed_error = self.cruise_speed - state.velocity
        throttle_cmd = state.throttle + speed_error * self.k_speed
        
        # Add throttle for climb
        if desired_climb_rate > 0:
            throttle_cmd += 0.1 * (desired_climb_rate / self.max_climb_rate)
            
        return ControlCommand(bank_cmd, throttle_cmd, "waypoint").limit(self.max_bank)
        
    def orbit_guidance(self, state: AircraftState, 
                      center: np.ndarray) -> ControlCommand:
        """
        Orbit around a point.
        
        Args:
            state: Current aircraft state
            center: Center point to orbit [x, y, z]
            
        Returns:
            Control commands
        """
        # Vector from center to aircraft
        radial = state.position - center
        radial[2] = 0  # Ignore vertical component for orbit
        current_radius = np.linalg.norm(radial[:2])
        
        if current_radius < 0.1:
            # Too close to center, move away first
            desired_heading = state.heading  # Continue current heading
        else:
            # Tangent direction (perpendicular to radial)
            tangent = np.array([-radial[1], radial[0], 0])
            tangent = tangent / np.linalg.norm(tangent[:2])
            
            # Desired heading (tangent direction with radius correction)
            radius_error = self.orbit_radius - current_radius
            
            # Mix radial and tangent based on radius error
            radial_weight = np.clip(radius_error / 100.0, -0.5, 0.5)
            desired_dir = tangent + radial_weight * (radial / current_radius)
            
            desired_heading = np.arctan2(desired_dir[1], desired_dir[0])
            
        # Heading control
        heading_error = self._wrap_angle(desired_heading - state.heading)
        bank_cmd = self.k_heading * heading_error
        
        # Maintain altitude
        altitude_error = center[2] - state.position[2]
        if abs(altitude_error) > 50:
            bank_cmd *= 0.8  # Reduce bank when correcting altitude
            
        # Speed control
        throttle_cmd = 0.5 + (self.cruise_speed - state.velocity) * self.k_speed
        
        return ControlCommand(bank_cmd, throttle_cmd, "orbit").limit(self.max_bank)
        
    def pursuit_guidance(self, state: AircraftState,
                        target: np.ndarray) -> ControlCommand:
        """
        Pure pursuit guidance to intercept target.
        
        Args:
            state: Current aircraft state
            target: Target position [x, y, z]
            
        Returns:
            Control commands
        """
        # Lead pursuit - aim ahead of target if it's moving
        # For now, simple pure pursuit
        delta = target - state.position
        range_to_target = np.linalg.norm(delta)
        
        # Desired heading toward target
        desired_heading = np.arctan2(delta[1], delta[0])
        heading_error = self._wrap_angle(desired_heading - state.heading)
        
        # More aggressive bank when close
        urgency = np.clip(2000.0 / (range_to_target + 100), 0.5, 2.0)
        bank_cmd = self.k_heading * heading_error * urgency
        
        # Speed up when far, slow down when close for better turn
        if range_to_target > 1000:
            desired_speed = self.cruise_speed * 1.2  # Speed up
        elif range_to_target < 200:
            desired_speed = self.cruise_speed * 0.8  # Slow down for turn
        else:
            desired_speed = self.cruise_speed
            
        speed_error = desired_speed - state.velocity
        throttle_cmd = state.throttle + speed_error * self.k_speed * 2  # More aggressive speed control
        
        return ControlCommand(bank_cmd, throttle_cmd, "pursuit").limit(self.max_bank)
        
    def evasive_guidance(self, state: AircraftState,
                        threat: np.ndarray) -> ControlCommand:
        """
        Evasive maneuver away from threat.
        
        Args:
            state: Current aircraft state
            threat: Threat position [x, y, z]
            
        Returns:
            Control commands
        """
        # Vector from threat to aircraft
        escape_vector = state.position - threat
        range_to_threat = np.linalg.norm(escape_vector[:2])
        
        if range_to_threat < 0.1:
            # Too close, random direction
            escape_heading = state.heading + np.pi/2
        else:
            # Head away from threat
            escape_heading = np.arctan2(escape_vector[1], escape_vector[0])
            
            # Add weaving based on aggressiveness
            if self.evasion_aggressiveness > 0.5:
                # Sinusoidal weaving
                weave = np.sin(state.time * 2) * np.pi/4 * self.evasion_aggressiveness
                escape_heading += weave
                
        # Heading control
        heading_error = self._wrap_angle(escape_heading - state.heading)
        
        # Aggressive bank for evasion
        bank_cmd = self.k_heading * heading_error * (1 + self.evasion_aggressiveness)
        
        # Full throttle when close, normal when far
        threat_urgency = np.clip(1000.0 / (range_to_threat + 100), 0.5, 1.0)
        throttle_cmd = 0.5 + 0.5 * threat_urgency
        
        # Altitude changes for 3D evasion
        if self.evasion_aggressiveness > 0.7:
            # Climb or dive based on relative altitude
            if threat[2] > state.position[2]:
                # Threat is above, dive
                bank_cmd *= 0.8  # Reduce bank to allow energy for dive
            else:
                # Threat is below, climb
                throttle_cmd = 1.0  # Max power for climb
                
        return ControlCommand(bank_cmd, throttle_cmd, "evade").limit(self.max_bank)
        
    def patrol_guidance(self, state: AircraftState) -> ControlCommand:
        """
        Patrol pattern (figure-8 or racetrack).
        
        Args:
            state: Current aircraft state
            
        Returns:
            Control commands
        """
        # Simple figure-8 pattern using waypoints
        if not self.waypoints:
            # Create default patrol pattern around current position
            center = state.position.copy()
            self.waypoints = [
                center + np.array([1000, 0, 0]),
                center + np.array([1000, 1000, 0]),
                center + np.array([-1000, 1000, 0]),
                center + np.array([-1000, 0, 0]),
                center + np.array([-1000, -1000, 0]),
                center + np.array([1000, -1000, 0]),
            ]
            self.current_waypoint_idx = 0
            
        return self.waypoint_guidance(state, self.waypoints[self.current_waypoint_idx])
        
    def set_mode(self, mode: BehaviorMode):
        """
        Set behavior mode.
        
        Args:
            mode: New behavior mode
        """
        self.mode = mode
        
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """
        Set waypoint list.
        
        Args:
            waypoints: List of waypoint positions
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
    def next_waypoint(self):
        """Advance to next waypoint"""
        if self.waypoints:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.current_waypoint_idx = 0  # Loop back to start
                
    def set_orbit(self, center: np.ndarray, radius: float):
        """
        Set orbit parameters.
        
        Args:
            center: Orbit center position
            radius: Orbit radius in meters
        """
        self.orbit_center = center
        self.orbit_radius = radius
        
    def set_evasion_parameters(self, distance: float, aggressiveness: float):
        """
        Set evasion parameters.
        
        Args:
            distance: Start evasion at this range
            aggressiveness: 0-1, higher = more aggressive
        """
        self.evasion_distance = distance
        self.evasion_aggressiveness = np.clip(aggressiveness, 0.0, 1.0)
        
    def _wrap_angle(self, angle: float) -> float:
        """
        Wrap angle to [-π, π].
        
        Args:
            angle: Angle in radians
            
        Returns:
            Wrapped angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get controller status.
        
        Returns:
            Status dictionary
        """
        status = {
            'mode': self.mode.value,
            'waypoint_count': len(self.waypoints),
            'current_waypoint': self.current_waypoint_idx,
        }
        
        if self.orbit_center is not None:
            status['orbit_center'] = self.orbit_center.tolist()
            status['orbit_radius'] = self.orbit_radius
            
        return status