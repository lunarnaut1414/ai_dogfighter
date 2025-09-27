"""
Guidance laws for autonomous interceptor.
Implements various pursuit and intercept algorithms.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass


class GuidanceMode(Enum):
    """Guidance algorithm modes"""
    PURE_PURSUIT = "pure_pursuit"
    LEAD_PURSUIT = "lead_pursuit"
    PROPORTIONAL_NAVIGATION = "proportional_navigation"
    AUGMENTED_PN = "augmented_pn"
    OPTIMAL_GUIDANCE = "optimal_guidance"
    PREDICTIVE_GUIDANCE = "predictive_guidance"


@dataclass
class GuidanceCommand:
    """Guidance command output"""
    commanded_heading: float  # radians
    commanded_altitude: float  # meters
    commanded_velocity: float  # m/s
    commanded_throttle: float  # 0-1
    target_id: Optional[str] = None
    confidence: float = 1.0
    mode: Optional[str] = None


class GuidanceLaw:
    """Base class for guidance laws"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize guidance law.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.last_los_rate = 0.0
        self.last_time = None
        
    def compute(self, 
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute guidance commands.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset internal state"""
        self.last_los_rate = 0.0
        self.last_time = None


class PurePursuit(GuidanceLaw):
    """
    Pure pursuit guidance - always point at target.
    Simplest guidance law, good for slow targets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.look_ahead_time = config.get('look_ahead_time', 0.0) if config else 0.0
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute pure pursuit guidance.
        
        Args:
            interceptor_state: Current interceptor state
            target_state: Current target state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Extract positions
        int_pos = np.array(interceptor_state['position'])
        tgt_pos = np.array(target_state['position'])
        
        # Look ahead if configured
        if self.look_ahead_time > 0 and 'velocity_vector' in target_state:
            tgt_vel = np.array(target_state['velocity_vector'])
            tgt_pos = tgt_pos + tgt_vel * self.look_ahead_time
            
        # Calculate line of sight vector
        los_vector = tgt_pos - int_pos
        range_to_target = np.linalg.norm(los_vector)
        
        if range_to_target < 0.1:  # Avoid division by zero
            return GuidanceCommand(
                commanded_heading=interceptor_state.get('heading', 0),
                commanded_altitude=int_pos[2],
                commanded_velocity=interceptor_state.get('velocity', 50),
                commanded_throttle=0.5
            )
            
        # Calculate desired heading (2D)
        desired_heading = np.arctan2(los_vector[1], los_vector[0])
        
        # Calculate desired altitude
        desired_altitude = tgt_pos[2]
        
        # Speed control based on range
        if range_to_target > 5000:
            commanded_throttle = 1.0
        elif range_to_target > 1000:
            commanded_throttle = 0.8
        else:
            commanded_throttle = 0.6
            
        desired_velocity = interceptor_state.get('max_velocity', 80) * commanded_throttle
        
        return GuidanceCommand(
            commanded_heading=desired_heading,
            commanded_altitude=desired_altitude,
            commanded_velocity=desired_velocity,
            commanded_throttle=commanded_throttle,
            mode="pure_pursuit"
        )


class LeadPursuit(GuidanceLaw):
    """
    Lead pursuit guidance - aim ahead of target.
    Better for moving targets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lead_constant = config.get('lead_constant', 1.0) if config else 1.0
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute lead pursuit guidance.
        """
        # Extract states
        int_pos = np.array(interceptor_state['position'])
        tgt_pos = np.array(target_state['position'])
        int_vel = np.array(interceptor_state.get('velocity_vector', [0, 0, 0]))
        tgt_vel = np.array(target_state.get('velocity_vector', [0, 0, 0]))
        
        # Calculate range and closing velocity
        los_vector = tgt_pos - int_pos
        range_to_target = np.linalg.norm(los_vector)
        
        if range_to_target < 0.1:
            return GuidanceCommand(
                commanded_heading=interceptor_state.get('heading', 0),
                commanded_altitude=int_pos[2],
                commanded_velocity=interceptor_state.get('velocity', 50),
                commanded_throttle=0.5
            )
            
        los_unit = los_vector / range_to_target
        closing_velocity = -np.dot(los_unit, int_vel - tgt_vel)
        
        # Estimate time to intercept
        if closing_velocity > 1.0:
            time_to_go = range_to_target / closing_velocity
        else:
            time_to_go = 100.0  # Large value if not closing
            
        # Limit lead time
        lead_time = min(time_to_go * self.lead_constant, 10.0)
        
        # Predict target position
        predicted_pos = tgt_pos + tgt_vel * lead_time
        
        # Calculate heading to predicted position
        lead_vector = predicted_pos - int_pos
        desired_heading = np.arctan2(lead_vector[1], lead_vector[0])
        
        # Altitude and speed control
        desired_altitude = predicted_pos[2]
        
        if range_to_target > 3000:
            commanded_throttle = 1.0
        elif range_to_target > 500:
            commanded_throttle = 0.85
        else:
            commanded_throttle = 0.7
            
        desired_velocity = interceptor_state.get('max_velocity', 80) * commanded_throttle
        
        return GuidanceCommand(
            commanded_heading=desired_heading,
            commanded_altitude=desired_altitude,
            commanded_velocity=desired_velocity,
            commanded_throttle=commanded_throttle,
            mode="lead_pursuit"
        )


class ProportionalNavigation(GuidanceLaw):
    """
    Proportional Navigation (PN) guidance law.
    Commands acceleration proportional to line-of-sight rate.
    Most common missile guidance law.
    """
    
    def __init__(self, navigation_gain: float = 3.0, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.N = navigation_gain  # Navigation constant (typically 3-5)
        self.last_los_angle = None
        self.los_rate_filter = None
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute proportional navigation guidance.
        """
        # Extract states
        int_pos = np.array(interceptor_state['position'])
        tgt_pos = np.array(target_state['position'])
        int_vel = interceptor_state.get('velocity', 50)
        
        # Calculate line of sight
        los_vector = tgt_pos - int_pos
        range_to_target = np.linalg.norm(los_vector)
        
        if range_to_target < 0.1:
            return GuidanceCommand(
                commanded_heading=interceptor_state.get('heading', 0),
                commanded_altitude=int_pos[2],
                commanded_velocity=int_vel,
                commanded_throttle=0.5
            )
            
        # Calculate LOS angles
        los_azimuth = np.arctan2(los_vector[1], los_vector[0])
        los_elevation = np.arctan2(los_vector[2], 
                                   np.sqrt(los_vector[0]**2 + los_vector[1]**2))
        
        # Calculate LOS rates
        if self.last_los_angle is not None:
            los_rate_az = (los_azimuth - self.last_los_angle[0]) / dt
            los_rate_el = (los_elevation - self.last_los_angle[1]) / dt
            
            # Simple low-pass filter for noise reduction
            if self.los_rate_filter is not None:
                alpha = 0.7  # Filter constant
                los_rate_az = alpha * los_rate_az + (1-alpha) * self.los_rate_filter[0]
                los_rate_el = alpha * los_rate_el + (1-alpha) * self.los_rate_filter[1]
                
            self.los_rate_filter = (los_rate_az, los_rate_el)
        else:
            los_rate_az = 0.0
            los_rate_el = 0.0
            self.los_rate_filter = (0.0, 0.0)
            
        self.last_los_angle = (los_azimuth, los_elevation)
        
        # Calculate closing velocity
        if 'velocity_vector' in interceptor_state and 'velocity_vector' in target_state:
            int_vel_vec = np.array(interceptor_state['velocity_vector'])
            tgt_vel_vec = np.array(target_state['velocity_vector'])
            closing_velocity = -np.dot(los_vector/range_to_target, 
                                      int_vel_vec - tgt_vel_vec)
        else:
            closing_velocity = int_vel
            
        # PN guidance law: a_cmd = N * Vc * λ_dot
        lateral_accel_cmd = self.N * closing_velocity * los_rate_az
        vertical_accel_cmd = self.N * closing_velocity * los_rate_el
        
        # Convert acceleration commands to heading and altitude
        # Limit accelerations to aircraft capabilities
        max_lateral_accel = 9.81 * 3  # 3g lateral
        max_vertical_accel = 9.81 * 2  # 2g vertical
        
        lateral_accel_cmd = np.clip(lateral_accel_cmd, -max_lateral_accel, max_lateral_accel)
        vertical_accel_cmd = np.clip(vertical_accel_cmd, -max_vertical_accel, max_vertical_accel)
        
        # Convert to heading change
        if int_vel > 0:
            heading_change = lateral_accel_cmd / int_vel * dt
        else:
            heading_change = 0
            
        current_heading = interceptor_state.get('heading', 0)
        commanded_heading = current_heading + heading_change
        
        # Altitude command based on vertical acceleration
        altitude_change = vertical_accel_cmd * dt * dt / 2  # Simple integration
        commanded_altitude = int_pos[2] + altitude_change + vertical_accel_cmd * 5  # Lead compensation
        
        # Speed control
        if range_to_target > 2000:
            commanded_throttle = 1.0
        elif range_to_target > 500:
            commanded_throttle = 0.85
        else:
            commanded_throttle = 0.7
            
        return GuidanceCommand(
            commanded_heading=commanded_heading,
            commanded_altitude=commanded_altitude,
            commanded_velocity=int_vel,
            commanded_throttle=commanded_throttle,
            mode="proportional_navigation"
        )


class AugmentedProportionalNavigation(ProportionalNavigation):
    """
    Augmented Proportional Navigation (APN).
    Adds target acceleration compensation to basic PN.
    """
    
    def __init__(self, navigation_gain: float = 4.0, config: Optional[Dict[str, Any]] = None):
        super().__init__(navigation_gain, config)
        self.last_target_vel = None
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute augmented proportional navigation.
        """
        # Get basic PN command
        base_command = super().compute(interceptor_state, target_state, dt)
        
        # Estimate target acceleration
        if 'velocity_vector' in target_state:
            tgt_vel = np.array(target_state['velocity_vector'])
            
            if self.last_target_vel is not None:
                tgt_accel = (tgt_vel - self.last_target_vel) / dt
                
                # Add compensation term
                # APN: a_cmd = N * Vc * λ_dot + (N/2) * a_target_perpendicular
                
                # Project target acceleration perpendicular to LOS
                int_pos = np.array(interceptor_state['position'])
                tgt_pos = np.array(target_state['position'])
                los_vector = tgt_pos - int_pos
                range_to_target = np.linalg.norm(los_vector)
                
                if range_to_target > 0.1:
                    los_unit = los_vector / range_to_target
                    tgt_accel_perp = tgt_accel - np.dot(tgt_accel, los_unit) * los_unit
                    
                    # Add half the perpendicular target acceleration
                    compensation = np.linalg.norm(tgt_accel_perp) * self.N / 2
                    
                    # Apply compensation to altitude command
                    base_command.commanded_altitude += compensation * dt * 5
                    
            self.last_target_vel = tgt_vel
            
        return base_command


class OptimalGuidance(GuidanceLaw):
    """
    Optimal guidance law minimizing control effort.
    Based on optimal control theory.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.time_constant = config.get('time_constant', 3.0) if config else 3.0
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute optimal guidance commands.
        """
        # Extract states
        int_pos = np.array(interceptor_state['position'])
        tgt_pos = np.array(target_state['position'])
        int_vel_vec = np.array(interceptor_state.get('velocity_vector', [0, 0, 0]))
        tgt_vel_vec = np.array(target_state.get('velocity_vector', [0, 0, 0]))
        
        # Calculate zero effort miss (ZEM)
        rel_pos = tgt_pos - int_pos
        rel_vel = tgt_vel_vec - int_vel_vec
        range_to_target = np.linalg.norm(rel_pos)
        
        if range_to_target < 0.1:
            return GuidanceCommand(
                commanded_heading=interceptor_state.get('heading', 0),
                commanded_altitude=int_pos[2],
                commanded_velocity=interceptor_state.get('velocity', 50),
                commanded_throttle=0.5
            )
            
        # Estimate time-to-go
        closing_speed = -np.dot(rel_pos, rel_vel) / range_to_target
        if closing_speed > 1.0:
            t_go = range_to_target / closing_speed
        else:
            t_go = 100.0
            
        # Calculate ZEM
        zem = rel_pos + rel_vel * t_go
        
        # Optimal acceleration command
        if t_go > 0.1:
            n_opt = 3 * self.time_constant / (t_go * t_go)
            accel_cmd = n_opt * zem / t_go
        else:
            accel_cmd = np.zeros(3)
            
        # Limit acceleration
        max_accel = 9.81 * 4  # 4g limit
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > max_accel:
            accel_cmd = accel_cmd * (max_accel / accel_mag)
            
        # Convert to heading and altitude commands
        int_vel = np.linalg.norm(int_vel_vec)
        if int_vel > 0:
            # Lateral acceleration to heading rate
            heading_rate = accel_cmd[1] / int_vel
            commanded_heading = interceptor_state.get('heading', 0) + heading_rate * dt
            
            # Vertical acceleration to altitude rate
            altitude_rate = accel_cmd[2]
            commanded_altitude = int_pos[2] + altitude_rate * dt * 5
        else:
            commanded_heading = interceptor_state.get('heading', 0)
            commanded_altitude = int_pos[2]
            
        # Speed control based on time-to-go
        if t_go > 20:
            commanded_throttle = 1.0
        elif t_go > 5:
            commanded_throttle = 0.85
        else:
            commanded_throttle = 0.7
            
        return GuidanceCommand(
            commanded_heading=commanded_heading,
            commanded_altitude=commanded_altitude,
            commanded_velocity=int_vel,
            commanded_throttle=commanded_throttle,
            mode="optimal_guidance"
        )


class PredictiveGuidance(GuidanceLaw):
    """
    Predictive guidance using target motion prediction.
    Good for maneuvering targets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.prediction_horizon = config.get('prediction_horizon', 5.0) if config else 5.0
        self.target_history = []
        self.max_history = 10
        
    def _predict_target_position(self, current_state: Dict, prediction_time: float) -> np.ndarray:
        """
        Predict target position using motion history.
        """
        current_pos = np.array(current_state['position'])
        
        if len(self.target_history) < 2:
            # Simple linear prediction
            if 'velocity_vector' in current_state:
                vel = np.array(current_state['velocity_vector'])
                return current_pos + vel * prediction_time
            else:
                return current_pos
                
        # Estimate acceleration from history
        if len(self.target_history) >= 3:
            # Use last 3 points for quadratic prediction
            t = [h['time'] for h in self.target_history[-3:]]
            pos = [h['position'] for h in self.target_history[-3:]]
            
            dt1 = t[-1] - t[-2]
            dt2 = t[-2] - t[-3]
            
            if dt1 > 0 and dt2 > 0:
                vel1 = (pos[-1] - pos[-2]) / dt1
                vel2 = (pos[-2] - pos[-3]) / dt2
                accel = (vel1 - vel2) / ((dt1 + dt2) / 2)
                
                # Quadratic prediction
                vel = np.array(current_state.get('velocity_vector', vel1))
                predicted_pos = current_pos + vel * prediction_time + 0.5 * accel * prediction_time**2
                return predicted_pos
                
        # Fallback to linear prediction
        vel = np.array(current_state.get('velocity_vector', [0, 0, 0]))
        return current_pos + vel * prediction_time
        
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute predictive guidance.
        """
        # Update target history
        self.target_history.append({
            'time': interceptor_state.get('time', 0),
            'position': np.array(target_state['position']),
            'velocity': np.array(target_state.get('velocity_vector', [0, 0, 0]))
        })
        
        if len(self.target_history) > self.max_history:
            self.target_history.pop(0)
            
        # Extract states
        int_pos = np.array(interceptor_state['position'])
        int_vel = interceptor_state.get('velocity', 50)
        
        # Estimate intercept time
        current_range = np.linalg.norm(np.array(target_state['position']) - int_pos)
        estimated_intercept_time = min(current_range / int_vel, self.prediction_horizon)
        
        # Predict target position
        predicted_target_pos = self._predict_target_position(target_state, estimated_intercept_time)
        
        # Calculate intercept trajectory
        intercept_vector = predicted_target_pos - int_pos
        intercept_range = np.linalg.norm(intercept_vector)
        
        if intercept_range < 0.1:
            return GuidanceCommand(
                commanded_heading=interceptor_state.get('heading', 0),
                commanded_altitude=int_pos[2],
                commanded_velocity=int_vel,
                commanded_throttle=0.5
            )
            
        # Command heading toward predicted intercept point
        commanded_heading = np.arctan2(intercept_vector[1], intercept_vector[0])
        commanded_altitude = predicted_target_pos[2]
        
        # Adjust speed based on prediction confidence
        if len(self.target_history) >= 3:
            # High confidence - aggressive pursuit
            if current_range > 2000:
                commanded_throttle = 1.0
            elif current_range > 500:
                commanded_throttle = 0.9
            else:
                commanded_throttle = 0.75
        else:
            # Low confidence - conservative approach
            commanded_throttle = 0.8
            
        return GuidanceCommand(
            commanded_heading=commanded_heading,
            commanded_altitude=commanded_altitude,
            commanded_velocity=int_vel,
            commanded_throttle=commanded_throttle,
            mode="predictive_guidance"
        )


class HybridGuidance:
    """
    Hybrid guidance system that switches between different guidance laws
    based on engagement geometry and phase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid guidance system.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize individual guidance laws
        self.pure_pursuit = PurePursuit()
        self.lead_pursuit = LeadPursuit()
        self.pn_guidance = ProportionalNavigation(navigation_gain=3.0)
        self.apn_guidance = AugmentedProportionalNavigation(navigation_gain=4.0)
        self.optimal_guidance = OptimalGuidance()
        self.predictive_guidance = PredictiveGuidance()
        
        # Mode selection parameters
        self.long_range_threshold = config.get('long_range', 5000) if config else 5000
        self.medium_range_threshold = config.get('medium_range', 1000) if config else 1000
        self.close_range_threshold = config.get('close_range', 200) if config else 200
        
        self.current_mode = None
        
    def select_mode(self, 
                   range_to_target: float,
                   closing_velocity: float,
                   target_maneuvering: bool = False) -> GuidanceMode:
        """
        Select appropriate guidance mode based on engagement conditions.
        
        Args:
            range_to_target: Distance to target in meters
            closing_velocity: Closing velocity in m/s (positive = closing)
            target_maneuvering: Whether target is maneuvering
            
        Returns:
            Selected guidance mode
        """
        if range_to_target > self.long_range_threshold:
            # Long range - use lead pursuit or predictive
            if target_maneuvering:
                return GuidanceMode.PREDICTIVE_GUIDANCE
            else:
                return GuidanceMode.LEAD_PURSUIT
                
        elif range_to_target > self.medium_range_threshold:
            # Medium range - use PN or APN
            if target_maneuvering:
                return GuidanceMode.AUGMENTED_PN
            else:
                return GuidanceMode.PROPORTIONAL_NAVIGATION
                
        elif range_to_target > self.close_range_threshold:
            # Short range - use optimal guidance
            return GuidanceMode.OPTIMAL_GUIDANCE
            
        else:
            # Terminal phase - pure pursuit for simplicity
            return GuidanceMode.PURE_PURSUIT
            
    def compute(self,
                interceptor_state: Dict[str, Any],
                target_state: Dict[str, Any],
                dt: float = 0.02) -> GuidanceCommand:
        """
        Compute hybrid guidance commands.
        
        Args:
            interceptor_state: Current interceptor state
            target_state: Current target state  
            dt: Time step
            
        Returns:
            Guidance command with selected mode
        """
        # Calculate engagement parameters
        int_pos = np.array(interceptor_state['position'])
        tgt_pos = np.array(target_state['position'])
        range_to_target = np.linalg.norm(tgt_pos - int_pos)
        
        # Calculate closing velocity
        if 'velocity_vector' in interceptor_state and 'velocity_vector' in target_state:
            int_vel = np.array(interceptor_state['velocity_vector'])
            tgt_vel = np.array(target_state['velocity_vector'])
            los_unit = (tgt_pos - int_pos) / max(range_to_target, 0.1)
            closing_velocity = -np.dot(los_unit, int_vel - tgt_vel)
        else:
            closing_velocity = interceptor_state.get('velocity', 50)
            
        # Detect if target is maneuvering (simplified)
        target_maneuvering = False
        if 'acceleration' in target_state:
            target_accel = np.linalg.norm(target_state['acceleration'])
            target_maneuvering = target_accel > 5.0  # More than 0.5g
            
        # Select guidance mode
        mode = self.select_mode(range_to_target, closing_velocity, target_maneuvering)
        
        # Execute selected guidance law
        if mode == GuidanceMode.PURE_PURSUIT:
            command = self.pure_pursuit.compute(interceptor_state, target_state, dt)
        elif mode == GuidanceMode.LEAD_PURSUIT:
            command = self.lead_pursuit.compute(interceptor_state, target_state, dt)
        elif mode == GuidanceMode.PROPORTIONAL_NAVIGATION:
            command = self.pn_guidance.compute(interceptor_state, target_state, dt)
        elif mode == GuidanceMode.AUGMENTED_PN:
            command = self.apn_guidance.compute(interceptor_state, target_state, dt)
        elif mode == GuidanceMode.OPTIMAL_GUIDANCE:
            command = self.optimal_guidance.compute(interceptor_state, target_state, dt)
        elif mode == GuidanceMode.PREDICTIVE_GUIDANCE:
            command = self.predictive_guidance.compute(interceptor_state, target_state, dt)
        else:
            # Fallback to pure pursuit
            command = self.pure_pursuit.compute(interceptor_state, target_state, dt)
            
        # Update mode in command
        command.mode = mode.value
        self.current_mode = mode
        
        return command
        
    def reset(self):
        """Reset all guidance laws"""
        self.pure_pursuit.reset()
        self.lead_pursuit.reset()
        self.pn_guidance.reset()
        self.apn_guidance.reset()
        self.optimal_guidance.reset()
        self.predictive_guidance.reset()
        self.current_mode = None