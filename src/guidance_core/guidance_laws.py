# src/guidance_core/guidance_laws.py
"""
Core guidance law implementations for interceptor.
Includes Pure Pursuit, Proportional Navigation variants, Optimal Guidance, and MPC.
All algorithms are platform-agnostic (pure math, no hardware dependencies).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings


@dataclass
class GuidanceCommand:
    """Output command from guidance law"""
    acceleration_command: np.ndarray  # [ax, ay, az] in m/s²
    heading_rate: float  # rad/s
    flight_path_rate: float  # rad/s
    throttle_command: float  # 0-1
    guidance_mode: str
    time_to_go: Optional[float] = None
    predicted_miss_distance: Optional[float] = None
    confidence: float = 1.0


class GuidanceLaw:
    """Base class for all guidance laws"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize guidance law with configuration.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.name = "base"
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute guidance command.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class PurePursuit(GuidanceLaw):
    """
    Pure Pursuit guidance law.
    Simple but robust, good for testing and fallback.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "pure_pursuit"
        self.look_ahead_time = config.get('look_ahead_time', 2.0) if config else 2.0
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute pure pursuit guidance.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Extract states
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Predict target position
        target_pred = target_pos + target_vel * self.look_ahead_time
        
        # Compute pursuit vector
        pursuit_vector = target_pred - own_pos
        range_to_target = np.linalg.norm(pursuit_vector)
        
        if range_to_target < 1.0:  # Very close
            return GuidanceCommand(
                acceleration_command=np.zeros(3),
                heading_rate=0.0,
                flight_path_rate=0.0,
                throttle_command=0.5,
                guidance_mode=self.name,
                time_to_go=0.0,
                predicted_miss_distance=range_to_target
            )
            
        # Normalize pursuit direction
        pursuit_dir = pursuit_vector / range_to_target
        
        # Compute desired velocity
        own_speed = np.linalg.norm(own_vel)
        desired_vel = pursuit_dir * own_speed
        
        # Compute acceleration command
        accel_cmd = (desired_vel - own_vel) / dt
        accel_cmd = np.clip(accel_cmd, -30, 30)  # Limit to 3g
        
        # Convert to heading and flight path angle rates
        heading_rate = self._compute_heading_rate(own_vel, desired_vel)
        flight_path_rate = self._compute_flight_path_rate(own_vel, desired_vel)
        
        # Simple throttle control
        closing_rate = np.dot(own_vel - target_vel, pursuit_dir)
        if closing_rate < 0:  # Opening
            throttle = 1.0
        else:
            throttle = 0.7
            
        return GuidanceCommand(
            acceleration_command=accel_cmd,
            heading_rate=heading_rate,
            flight_path_rate=flight_path_rate,
            throttle_command=throttle,
            guidance_mode=self.name,
            time_to_go=range_to_target / max(closing_rate, 1.0),
            predicted_miss_distance=range_to_target * 0.1  # Simple estimate
        )
        
    def _compute_heading_rate(self, current_vel: np.ndarray, desired_vel: np.ndarray) -> float:
        """Compute heading rate command"""
        # Project onto horizontal plane
        current_horizontal = current_vel[:2]
        desired_horizontal = desired_vel[:2]
        
        if np.linalg.norm(current_horizontal) < 1.0:
            return 0.0
            
        # Compute angle difference
        current_heading = np.arctan2(current_horizontal[1], current_horizontal[0])
        desired_heading = np.arctan2(desired_horizontal[1], desired_horizontal[0])
        
        # Wrap angle difference
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # P controller for heading rate
        return np.clip(heading_error * 2.0, -1.0, 1.0)  # rad/s
        
    def _compute_flight_path_rate(self, current_vel: np.ndarray, desired_vel: np.ndarray) -> float:
        """Compute flight path angle rate command"""
        current_speed = np.linalg.norm(current_vel)
        desired_speed = np.linalg.norm(desired_vel)
        
        if current_speed < 1.0 or desired_speed < 1.0:
            return 0.0
            
        current_fpa = np.arcsin(-current_vel[2] / current_speed)
        desired_fpa = np.arcsin(-desired_vel[2] / desired_speed)
        
        fpa_error = desired_fpa - current_fpa
        return np.clip(fpa_error * 2.0, -0.5, 0.5)  # rad/s


class ProportionalNavigation(GuidanceLaw):
    """
    Proportional Navigation (PN) guidance law.
    Classic missile guidance algorithm.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "proportional_navigation"
        self.N = config.get('navigation_gain', 3.0) if config else 3.0
        self.previous_los = None
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute PN guidance command.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Extract states
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Relative kinematics
        rel_pos = target_pos - own_pos
        rel_vel = target_vel - own_vel
        range_to_target = np.linalg.norm(rel_pos)
        
        if range_to_target < 1.0:
            return GuidanceCommand(
                acceleration_command=np.zeros(3),
                heading_rate=0.0,
                flight_path_rate=0.0,
                throttle_command=0.5,
                guidance_mode=self.name,
                time_to_go=0.0,
                predicted_miss_distance=range_to_target
            )
            
        # Line of sight vector
        los = rel_pos / range_to_target
        
        # Closing velocity
        closing_vel = -np.dot(rel_vel, los)
        
        # LOS rate calculation
        if self.previous_los is not None:
            # Numerical differentiation
            los_rate = (los - self.previous_los) / dt
            
            # Remove radial component
            los_rate = los_rate - np.dot(los_rate, los) * los
            
            # PN acceleration command: a = N * Vc * omega
            accel_cmd = self.N * closing_vel * los_rate
            
            # Limit acceleration
            accel_magnitude = np.linalg.norm(accel_cmd)
            if accel_magnitude > 30:  # 3g limit
                accel_cmd = accel_cmd * (30 / accel_magnitude)
        else:
            accel_cmd = np.zeros(3)
            
        self.previous_los = los.copy()
        
        # Convert to control rates
        own_speed = np.linalg.norm(own_vel)
        if own_speed > 1.0:
            # Desired acceleration perpendicular to velocity
            lateral_accel = accel_cmd - np.dot(accel_cmd, own_vel/own_speed) * (own_vel/own_speed)
            
            # Heading rate from horizontal acceleration
            heading_rate = lateral_accel[1] * np.cos(np.arctan2(own_vel[1], own_vel[0])) - \
                          lateral_accel[0] * np.sin(np.arctan2(own_vel[1], own_vel[0]))
            heading_rate = heading_rate / (own_speed * np.cos(np.arcsin(-own_vel[2]/own_speed)))
            
            # Flight path rate from vertical acceleration
            flight_path_rate = -lateral_accel[2] / own_speed
            
            heading_rate = np.clip(heading_rate, -1.0, 1.0)
            flight_path_rate = np.clip(flight_path_rate, -0.5, 0.5)
        else:
            heading_rate = 0.0
            flight_path_rate = 0.0
            
        # Time-to-go estimation
        if closing_vel > 0:
            time_to_go = range_to_target / closing_vel
        else:
            time_to_go = np.inf
            
        # Miss distance prediction (simplified)
        miss_distance = np.linalg.norm(rel_pos + rel_vel * time_to_go) if time_to_go < 100 else range_to_target
        
        # Throttle control
        if closing_vel < 10:  # Slow closing
            throttle = 1.0
        elif range_to_target < 500:  # Close range
            throttle = 0.8
        else:
            throttle = 0.7
            
        return GuidanceCommand(
            acceleration_command=accel_cmd,
            heading_rate=heading_rate,
            flight_path_rate=flight_path_rate,
            throttle_command=throttle,
            guidance_mode=self.name,
            time_to_go=time_to_go,
            predicted_miss_distance=miss_distance
        )


class AugmentedProportionalNavigation(ProportionalNavigation):
    """
    Augmented Proportional Navigation (APN) guidance law.
    PN with target acceleration compensation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "augmented_pn"
        self.previous_target_vel = None
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute APN guidance command.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Get base PN command
        pn_command = super().compute(own_state, target_state, dt)
        
        # Add target acceleration compensation
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        if self.previous_target_vel is not None:
            # Estimate target acceleration
            target_accel = (target_vel - self.previous_target_vel) / dt
            
            # Add compensation term (N/2 * at)
            compensation = (self.N / 2) * target_accel
            
            # Limit compensation
            comp_magnitude = np.linalg.norm(compensation)
            if comp_magnitude > 10:  # 1g limit for compensation
                compensation = compensation * (10 / comp_magnitude)
                
            pn_command.acceleration_command += compensation
            
            # Recompute control rates with augmented acceleration
            accel_total = pn_command.acceleration_command
            accel_magnitude = np.linalg.norm(accel_total)
            if accel_magnitude > 30:  # 3g total limit
                accel_total = accel_total * (30 / accel_magnitude)
                pn_command.acceleration_command = accel_total
                
        self.previous_target_vel = target_vel.copy()
        
        return pn_command


class OptimalGuidanceLaw(GuidanceLaw):
    """
    Optimal Guidance Law (OGL) for energy-optimal interception.
    Based on optimal control theory.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "optimal_guidance"
        self.gravity = 9.81
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute optimal guidance command.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Extract states
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Relative kinematics
        rel_pos = target_pos - own_pos
        rel_vel = target_vel - own_vel
        range_to_target = np.linalg.norm(rel_pos)
        
        # Estimate time-to-go
        closing_speed = -np.dot(rel_vel, rel_pos) / range_to_target
        if closing_speed > 0:
            tgo = range_to_target / closing_speed
        else:
            tgo = 100.0  # Large value
            
        if tgo < 0.1:  # Very close
            return GuidanceCommand(
                acceleration_command=np.zeros(3),
                heading_rate=0.0,
                flight_path_rate=0.0,
                throttle_command=0.5,
                guidance_mode=self.name,
                time_to_go=tgo,
                predicted_miss_distance=range_to_target
            )
            
        # Zero-effort miss (ZEM) calculation
        zem = rel_pos + rel_vel * tgo
        zem_magnitude = np.linalg.norm(zem)
        
        # Optimal acceleration command
        # a = -6 * ZEM / tgo^2 (for minimum energy)
        if tgo < 100:
            accel_cmd = -6 * zem / (tgo * tgo)
            
            # Add gravity compensation
            accel_cmd[2] -= self.gravity
            
            # Limit acceleration
            accel_magnitude = np.linalg.norm(accel_cmd)
            if accel_magnitude > 30:  # 3g limit
                accel_cmd = accel_cmd * (30 / accel_magnitude)
        else:
            # Fall back to pursuit if tgo is large
            pursuit = PurePursuit()
            return pursuit.compute(own_state, target_state, dt)
            
        # Convert to control rates
        own_speed = np.linalg.norm(own_vel)
        if own_speed > 1.0:
            # Project acceleration perpendicular to velocity
            lateral_accel = accel_cmd - np.dot(accel_cmd, own_vel/own_speed) * (own_vel/own_speed)
            
            # Compute rates
            heading_rate = (lateral_accel[0] * own_vel[1] - lateral_accel[1] * own_vel[0]) / (own_speed * own_speed)
            flight_path_rate = -lateral_accel[2] / own_speed
            
            heading_rate = np.clip(heading_rate, -1.0, 1.0)
            flight_path_rate = np.clip(flight_path_rate, -0.5, 0.5)
        else:
            heading_rate = 0.0
            flight_path_rate = 0.0
            
        # Optimal throttle based on energy management
        energy_to_target = 0.5 * own_speed * own_speed + self.gravity * (target_pos[2] - own_pos[2])
        if energy_to_target > 0:
            throttle = 0.7  # Maintain energy
        else:
            throttle = 1.0  # Need more energy
            
        return GuidanceCommand(
            acceleration_command=accel_cmd,
            heading_rate=heading_rate,
            flight_path_rate=flight_path_rate,
            throttle_command=throttle,
            guidance_mode=self.name,
            time_to_go=tgo,
            predicted_miss_distance=zem_magnitude
        )


class ModelPredictiveControl(GuidanceLaw):
    """
    Model Predictive Control (MPC) guidance.
    Optimizes trajectory over prediction horizon.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "model_predictive_control"
        self.horizon = config.get('horizon', 10) if config else 10
        self.dt_horizon = config.get('dt_horizon', 0.5) if config else 0.5
        self.Q = np.diag([1.0, 1.0, 1.0])  # Position error weight
        self.R = np.diag([0.1, 0.1, 0.1])  # Control effort weight
        
    def compute(self, own_state: Dict, target_state: Dict, dt: float) -> GuidanceCommand:
        """
        Compute MPC guidance command.
        Simplified version - full MPC would use optimization solver.
        
        Args:
            own_state: Own aircraft state
            target_state: Target aircraft state
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Extract states
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Predict target trajectory
        target_trajectory = []
        for i in range(self.horizon):
            t_future = i * self.dt_horizon
            # Simple constant velocity prediction
            target_future = target_pos + target_vel * t_future
            target_trajectory.append(target_future)
            
        # Simplified MPC: compute average desired acceleration
        accel_cmd = np.zeros(3)
        total_weight = 0.0
        
        for i, target_future in enumerate(target_trajectory):
            t_future = (i + 1) * self.dt_horizon
            
            # Where we want to be
            desired_pos = target_future
            
            # Where we will be with current velocity
            predicted_pos = own_pos + own_vel * t_future + 0.5 * accel_cmd * t_future * t_future
            
            # Position error
            error = desired_pos - predicted_pos
            
            # Weight (closer time steps more important)
            weight = 1.0 / (i + 1)
            
            # Desired acceleration to correct error
            if t_future > 0:
                desired_accel = 2 * error / (t_future * t_future)
                accel_cmd += weight * desired_accel
                total_weight += weight
                
        if total_weight > 0:
            accel_cmd /= total_weight
            
        # Limit acceleration
        accel_magnitude = np.linalg.norm(accel_cmd)
        if accel_magnitude > 30:  # 3g limit
            accel_cmd = accel_cmd * (30 / accel_magnitude)
            
        # Convert to control rates
        own_speed = np.linalg.norm(own_vel)
        if own_speed > 1.0:
            lateral_accel = accel_cmd - np.dot(accel_cmd, own_vel/own_speed) * (own_vel/own_speed)
            heading_rate = (lateral_accel[0] * own_vel[1] - lateral_accel[1] * own_vel[0]) / (own_speed * own_speed)
            flight_path_rate = -lateral_accel[2] / own_speed
            
            heading_rate = np.clip(heading_rate, -1.0, 1.0)
            flight_path_rate = np.clip(flight_path_rate, -0.5, 0.5)
        else:
            heading_rate = 0.0
            flight_path_rate = 0.0
            
        # Time-to-go and miss distance
        rel_pos = target_pos - own_pos
        range_to_target = np.linalg.norm(rel_pos)
        closing_speed = -np.dot(target_vel - own_vel, rel_pos) / range_to_target
        
        if closing_speed > 0:
            time_to_go = range_to_target / closing_speed
            predicted_miss = np.linalg.norm(rel_pos + (target_vel - own_vel) * time_to_go)
        else:
            time_to_go = np.inf
            predicted_miss = range_to_target
            
        # MPC typically optimizes throttle too
        if range_to_target > 1000:
            throttle = 0.8
        elif range_to_target > 500:
            throttle = 0.7
        else:
            throttle = 0.6
            
        return GuidanceCommand(
            acceleration_command=accel_cmd,
            heading_rate=heading_rate,
            flight_path_rate=flight_path_rate,
            throttle_command=throttle,
            guidance_mode=self.name,
            time_to_go=time_to_go,
            predicted_miss_distance=predicted_miss
        )


class GuidanceLawSelector:
    """
    Selects appropriate guidance law based on engagement conditions.
    """
    
    def __init__(self):
        """Initialize guidance law selector with all available laws"""
        self.laws = {
            'pure_pursuit': PurePursuit(),
            'proportional_navigation': ProportionalNavigation(),
            'augmented_pn': AugmentedProportionalNavigation(),
            'optimal_guidance': OptimalGuidanceLaw(),
            'model_predictive_control': ModelPredictiveControl()
        }
        
        self.current_law = 'proportional_navigation'
        self.switch_hysteresis_time = 2.0  # seconds
        self.last_switch_time = 0.0
        
    def select_guidance_law(self, own_state: Dict, target_state: Dict, 
                           mission_phase: str, current_time: float) -> str:
        """
        Select appropriate guidance law based on conditions.
        
        Args:
            own_state: Own aircraft state
            target_state: Target state
            mission_phase: Current mission phase
            current_time: Current simulation time
            
        Returns:
            Selected guidance law name
        """
        # Prevent rapid switching
        if current_time - self.last_switch_time < self.switch_hysteresis_time:
            return self.current_law
            
        # Extract key parameters
        rel_pos = np.array(target_state['position']) - np.array(own_state['position'])
        range_to_target = np.linalg.norm(rel_pos)
        
        rel_vel = np.array(target_state.get('velocity', [0,0,0])) - np.array(own_state['velocity'])
        closing_speed = -np.dot(rel_vel, rel_pos) / max(range_to_target, 1.0)
        
        # Selection logic
        selected = self.current_law
        
        if mission_phase == 'search' or mission_phase == 'track':
            # Use simple pursuit for search/track
            selected = 'pure_pursuit'
            
        elif mission_phase == 'intercept':
            if range_to_target > 2000:
                # Long range - use PN
                selected = 'proportional_navigation'
            elif range_to_target > 500:
                # Medium range - use APN for maneuvering targets
                target_accel = target_state.get('acceleration', np.zeros(3))
                if np.linalg.norm(target_accel) > 5:  # Target maneuvering
                    selected = 'augmented_pn'
                else:
                    selected = 'proportional_navigation'
            elif range_to_target > 100:
                # Short range - use optimal guidance
                selected = 'optimal_guidance'
            else:
                # Terminal phase - use MPC for precise control
                selected = 'model_predictive_control'
                
        elif mission_phase == 'evade':
            # Use MPC for evasive maneuvers
            selected = 'model_predictive_control'
            
        # Update if changed
        if selected != self.current_law:
            self.current_law = selected
            self.last_switch_time = current_time
            print(f"[GUIDANCE] Switched to {selected} at range {range_to_target:.0f}m")
            
        return self.current_law
        
    def compute(self, own_state: Dict, target_state: Dict, 
               mission_phase: str, current_time: float, dt: float) -> GuidanceCommand:
        """
        Compute guidance command using selected law.
        
        Args:
            own_state: Own aircraft state
            target_state: Target state
            mission_phase: Current mission phase
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Guidance command
        """
        # Select appropriate law
        law_name = self.select_guidance_law(own_state, target_state, mission_phase, current_time)
        
        # Compute command
        law = self.laws[law_name]
        command = law.compute(own_state, target_state, dt)
        
        return command