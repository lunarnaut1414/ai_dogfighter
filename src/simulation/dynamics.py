"""
Dynamics model for 3DOF fixed-wing aircraft simulation.
Provides interface between simulation environment and aircraft dynamics.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.assets.aircraft_3dof import Aircraft3DOF, AircraftState, FlightMode


class DynamicsType(Enum):
    """Types of dynamics models available"""
    POINT_MASS = "point_mass"          # Simple point mass
    THREE_DOF = "three_dof"             # 3DOF fixed-wing
    SIX_DOF = "six_dof"                 # Full 6DOF (future)
    QUADROTOR = "quadrotor"             # Quadrotor dynamics (future)


@dataclass
class DynamicsState:
    """Complete state for dynamics model"""
    position: np.ndarray           # [x, y, z] in meters
    velocity: np.ndarray           # [vx, vy, vz] in m/s
    orientation: np.ndarray        # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    
    # Additional states
    airspeed: float = 0.0
    angle_of_attack: float = 0.0
    sideslip_angle: float = 0.0
    
    # Control states
    throttle: float = 0.0
    control_surfaces: np.ndarray = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'airspeed': self.airspeed,
            'angle_of_attack': self.angle_of_attack,
            'sideslip_angle': self.sideslip_angle,
            'throttle': self.throttle
        }


class DynamicsModel:
    """
    Base dynamics model interface for aircraft simulation.
    Wraps the Aircraft3DOF model for use in the simulation environment.
    """
    
    def __init__(self, 
                 model_type: str = "three_dof",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dynamics model.
        
        Args:
            model_type: Type of dynamics model ("point_mass", "three_dof", etc.)
            config: Configuration dictionary for the model
        """
        self.model_type = DynamicsType(model_type) if isinstance(model_type, str) else model_type
        self.config = config or {}
        
        # Initialize underlying model based on type
        if self.model_type == DynamicsType.THREE_DOF:
            self._init_three_dof()
        elif self.model_type == DynamicsType.POINT_MASS:
            self._init_point_mass()
        else:
            raise ValueError(f"Unsupported dynamics model type: {model_type}")
            
        # State tracking
        self.time = 0.0
        self.dt = self.config.get('dt', 0.02)
        
    def _init_three_dof(self):
        """Initialize 3DOF fixed-wing model"""
        # Create Aircraft3DOF instance
        aircraft_config = self.config.get('aircraft', {})
        self.aircraft_model = Aircraft3DOF(config_dict=aircraft_config)
        
        # Initialize state
        initial_state = self.config.get('initial_state', {})
        position = np.array(initial_state.get('position', [0.0, 0.0, 1000.0]))
        velocity = initial_state.get('velocity', 50.0)
        heading = np.radians(initial_state.get('heading', 0.0))
        climb_angle = np.radians(initial_state.get('climb_angle', 0.0))
        
        self.aircraft_model.initialize_state(
            position=position,
            velocity=velocity,
            heading=heading,
            flight_path_angle=climb_angle
        )
        
    def _init_point_mass(self):
        """Initialize simple point mass model"""
        # Simple point mass dynamics (no aerodynamics)
        self.mass = self.config.get('mass', 100.0)  # kg
        self.max_thrust = self.config.get('max_thrust', 1000.0)  # N
        self.drag_coefficient = self.config.get('drag_coefficient', 0.1)
        
        # State vectors
        initial_state = self.config.get('initial_state', {})
        self.position = np.array(initial_state.get('position', [0.0, 0.0, 1000.0]))
        self.velocity = np.array(initial_state.get('velocity', [50.0, 0.0, 0.0]))
        self.throttle = 0.5
        
    def step(self, 
             control_input: Dict[str, Any],
             environmental_effects: Optional[Dict[str, Any]] = None) -> DynamicsState:
        """
        Step the dynamics model forward in time.
        
        Args:
            control_input: Control commands (throttle, bank_angle, etc.)
            environmental_effects: Environmental effects (wind, turbulence, etc.)
            
        Returns:
            Updated dynamics state
        """
        if self.model_type == DynamicsType.THREE_DOF:
            return self._step_three_dof(control_input, environmental_effects)
        elif self.model_type == DynamicsType.POINT_MASS:
            return self._step_point_mass(control_input, environmental_effects)
        else:
            raise ValueError(f"Step not implemented for {self.model_type}")
            
    def _step_three_dof(self, 
                       control_input: Dict[str, Any],
                       environmental_effects: Optional[Dict[str, Any]] = None) -> DynamicsState:
        """Step the 3DOF aircraft model"""
        # Apply control inputs
        if 'throttle' in control_input:
            self.aircraft_model.set_controls(throttle=control_input['throttle'])
        if 'bank_angle' in control_input:
            self.aircraft_model.set_controls(bank_angle=control_input['bank_angle'])
            
        # Apply environmental effects if provided
        if environmental_effects:
            self.aircraft_model.apply_environmental_effects(environmental_effects)
            
        # Step the model
        self.aircraft_model.update(self.dt)
        
        # Extract state
        state = self.aircraft_model.state
        
        # Convert to DynamicsState
        velocity_vec = state.get_velocity_vector()
        
        # Compute orientation from velocity vector
        heading = state.heading
        climb_angle = state.flight_path_angle
        bank_angle = state.bank_angle
        
        # Create orientation vector [roll, pitch, yaw]
        orientation = np.array([bank_angle, climb_angle, heading])
        
        # Angular velocity (simplified for 3DOF)
        angular_velocity = np.array([0.0, 0.0, state.heading_rate])
        
        return DynamicsState(
            position=state.position.copy(),
            velocity=velocity_vec,
            orientation=orientation,
            angular_velocity=angular_velocity,
            airspeed=state.velocity,
            angle_of_attack=0.0,  # Simplified in 3DOF
            sideslip_angle=0.0,   # Simplified in 3DOF
            throttle=state.throttle
        )
        
    def _step_point_mass(self,
                        control_input: Dict[str, Any],
                        environmental_effects: Optional[Dict[str, Any]] = None) -> DynamicsState:
        """Step the point mass model"""
        # Get control inputs
        self.throttle = control_input.get('throttle', self.throttle)
        desired_heading = control_input.get('heading', None)
        desired_altitude = control_input.get('altitude', None)
        
        # Compute thrust vector
        if desired_heading is not None:
            # Point thrust towards desired heading
            thrust_direction = np.array([
                np.cos(desired_heading),
                np.sin(desired_heading),
                0.0
            ])
        else:
            # Thrust in velocity direction
            speed = np.linalg.norm(self.velocity)
            if speed > 0.1:
                thrust_direction = self.velocity / speed
            else:
                thrust_direction = np.array([1.0, 0.0, 0.0])
                
        thrust_magnitude = self.throttle * self.max_thrust
        thrust = thrust_magnitude * thrust_direction
        
        # Add altitude control
        if desired_altitude is not None:
            altitude_error = desired_altitude - self.position[2]
            thrust[2] += np.clip(altitude_error * 10.0, -500, 500)  # Simple P control
            
        # Compute drag
        speed = np.linalg.norm(self.velocity)
        if speed > 0.1:
            drag = -self.drag_coefficient * speed * self.velocity
        else:
            drag = np.zeros(3)
            
        # Gravity
        gravity = np.array([0.0, 0.0, -9.81 * self.mass])
        
        # Wind effects
        wind = np.zeros(3)
        if environmental_effects and 'wind_vector' in environmental_effects:
            wind = np.array(environmental_effects['wind_vector'])
            
        # Total force
        total_force = thrust + drag + gravity
        
        # Update velocity and position
        acceleration = total_force / self.mass
        self.velocity += acceleration * self.dt + wind * 0.1  # Add wind effect
        self.position += self.velocity * self.dt
        
        # Ensure minimum altitude
        if self.position[2] < 10.0:
            self.position[2] = 10.0
            self.velocity[2] = max(0.0, self.velocity[2])
            
        # Compute orientation from velocity
        speed = np.linalg.norm(self.velocity[:2])
        if speed > 0.1:
            heading = np.arctan2(self.velocity[1], self.velocity[0])
        else:
            heading = 0.0
            
        climb_angle = np.arctan2(self.velocity[2], speed) if speed > 0.1 else 0.0
        
        return DynamicsState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=np.array([0.0, climb_angle, heading]),
            angular_velocity=np.zeros(3),
            airspeed=np.linalg.norm(self.velocity - wind),
            throttle=self.throttle
        )
        
    def set_state(self, state: DynamicsState):
        """
        Set the current state of the dynamics model.
        
        Args:
            state: New dynamics state
        """
        if self.model_type == DynamicsType.THREE_DOF:
            # Update Aircraft3DOF state
            self.aircraft_model.state.position = state.position.copy()
            
            # Convert velocity vector to speed and angles
            speed = np.linalg.norm(state.velocity)
            if speed > 0.1:
                heading = np.arctan2(state.velocity[1], state.velocity[0])
                climb_angle = np.arctan2(state.velocity[2], 
                                        np.linalg.norm(state.velocity[:2]))
            else:
                heading = state.orientation[2]  # Use yaw
                climb_angle = state.orientation[1]  # Use pitch
                
            self.aircraft_model.state.velocity = speed
            self.aircraft_model.state.heading = heading
            self.aircraft_model.state.flight_path_angle = climb_angle
            self.aircraft_model.state.bank_angle = state.orientation[0]
            self.aircraft_model.state.throttle = state.throttle
            
        elif self.model_type == DynamicsType.POINT_MASS:
            self.position = state.position.copy()
            self.velocity = state.velocity.copy()
            self.throttle = state.throttle
            
    def get_state(self) -> DynamicsState:
        """
        Get the current state of the dynamics model.
        
        Returns:
            Current dynamics state
        """
        if self.model_type == DynamicsType.THREE_DOF:
            return self._step_three_dof({}, None)  # Get state without stepping
        elif self.model_type == DynamicsType.POINT_MASS:
            speed = np.linalg.norm(self.velocity[:2])
            heading = np.arctan2(self.velocity[1], self.velocity[0]) if speed > 0.1 else 0.0
            climb_angle = np.arctan2(self.velocity[2], speed) if speed > 0.1 else 0.0
            
            return DynamicsState(
                position=self.position.copy(),
                velocity=self.velocity.copy(),
                orientation=np.array([0.0, climb_angle, heading]),
                angular_velocity=np.zeros(3),
                airspeed=np.linalg.norm(self.velocity),
                throttle=self.throttle
            )
            
    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        Reset the dynamics model to initial conditions.
        
        Args:
            initial_state: Optional new initial state
        """
        if initial_state:
            self.config['initial_state'] = initial_state
            
        # Reinitialize based on type
        if self.model_type == DynamicsType.THREE_DOF:
            self._init_three_dof()
        elif self.model_type == DynamicsType.POINT_MASS:
            self._init_point_mass()
            
        self.time = 0.0
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the dynamics model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': self.model_type.value,
            'time': self.time,
            'dt': self.dt
        }
        
        if self.model_type == DynamicsType.THREE_DOF:
            info['aircraft_config'] = self.aircraft_model.config
            info['fuel_remaining'] = self.aircraft_model.state.fuel_remaining
            info['fuel_capacity'] = self.aircraft_model.fuel_capacity
        elif self.model_type == DynamicsType.POINT_MASS:
            info['mass'] = self.mass
            info['max_thrust'] = self.max_thrust
            info['drag_coefficient'] = self.drag_coefficient
            
        return info
        
    def is_valid_state(self) -> bool:
        """
        Check if the current state is valid (no NaN values, within bounds).
        
        Returns:
            True if state is valid
        """
        state = self.get_state()
        
        # Check for NaN values
        if np.any(np.isnan(state.position)) or np.any(np.isnan(state.velocity)):
            return False
            
        # Check altitude
        if state.position[2] < 0:
            return False
            
        # Check speed limits
        speed = np.linalg.norm(state.velocity)
        if speed > 500:  # Max 500 m/s
            return False
            
        return True


class SimplifiedDynamics:
    """
    Simplified dynamics for testing and basic simulations.
    Provides minimal interface compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simplified dynamics"""
        self.config = config or {}
        self.dt = self.config.get('dt', 0.02)
        
        # State
        initial_state = self.config.get('initial_state', {})
        self.position = np.array(initial_state.get('position', [0.0, 0.0, 1000.0]))
        self.velocity = np.array(initial_state.get('velocity', [50.0, 0.0, 0.0]))
        self.heading = initial_state.get('heading', 0.0)
        
    def step(self, control_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simple step function"""
        # Update heading
        if 'heading_rate' in control_input:
            self.heading += control_input['heading_rate'] * self.dt
            
        # Update speed
        if 'acceleration' in control_input:
            speed = np.linalg.norm(self.velocity)
            speed += control_input['acceleration'] * self.dt
            speed = np.clip(speed, 10.0, 100.0)
            
            # Update velocity vector
            self.velocity[0] = speed * np.cos(self.heading)
            self.velocity[1] = speed * np.sin(self.heading)
            
        # Update altitude
        if 'climb_rate' in control_input:
            self.velocity[2] = control_input['climb_rate']
            
        # Update position
        self.position += self.velocity * self.dt
        
        # Ensure minimum altitude
        if self.position[2] < 10.0:
            self.position[2] = 10.0
            self.velocity[2] = max(0.0, self.velocity[2])
            
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'heading': self.heading
        }