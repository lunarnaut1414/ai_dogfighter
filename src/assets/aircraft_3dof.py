# src/assets/aircraft_3dof.py
"""
3DOF (Three Degree of Freedom) fixed-wing aircraft dynamics model.
Point-mass model with flight path angles for trajectory simulation.
Updated to include pitch control for altitude changes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import yaml


class FlightMode(Enum):
    """Aircraft flight modes"""
    NORMAL = "normal"
    STALLED = "stalled"
    GROUND = "ground"
    CRASHED = "crashed"


@dataclass
class AircraftState:
    """Complete 3DOF aircraft state"""
    # Position
    position: np.ndarray  # [x, y, z] in meters
    
    # Velocity
    velocity: float  # True airspeed in m/s
    
    # Flight path angles
    heading: float  # ψ (psi) - heading angle in radians
    flight_path_angle: float  # γ (gamma) - climb angle in radians
    
    # Control
    bank_angle: float  # φ (phi) - bank angle in radians
    pitch_angle: float  # θ (theta) - pitch angle in radians (NEW!)
    throttle: float  # Throttle setting [0, 1]
    
    # Energy
    fuel_remaining: float  # kg or fraction
    
    # Time
    time: float  # Simulation time in seconds
    
    def get_velocity_vector(self) -> np.ndarray:
        """Get velocity vector in world coordinates"""
        vx = self.velocity * np.cos(self.flight_path_angle) * np.cos(self.heading)
        vy = self.velocity * np.cos(self.flight_path_angle) * np.sin(self.heading)
        vz = self.velocity * np.sin(self.flight_path_angle)
        return np.array([vx, vy, vz])
    
    def get_kinetic_energy(self, mass: float) -> float:
        """Calculate kinetic energy"""
        return 0.5 * mass * self.velocity ** 2
    
    def get_potential_energy(self, mass: float) -> float:
        """Calculate potential energy"""
        return mass * 9.81 * self.position[2]
    
    def copy(self) -> 'AircraftState':
        """Create a deep copy of the state"""
        return AircraftState(
            position=self.position.copy(),
            velocity=self.velocity,
            heading=self.heading,
            flight_path_angle=self.flight_path_angle,
            bank_angle=self.bank_angle,
            pitch_angle=self.pitch_angle,
            throttle=self.throttle,
            fuel_remaining=self.fuel_remaining,
            time=self.time
        )


@dataclass
class AircraftForces:
    """Forces and accelerations acting on aircraft"""
    lift: float = 0.0  # N
    drag: float = 0.0  # N
    thrust: float = 0.0  # N
    weight: float = 0.0  # N
    
    # Accelerations
    acceleration_velocity: float = 0.0  # m/s²
    acceleration_gamma: float = 0.0  # rad/s
    acceleration_psi: float = 0.0  # rad/s


class Aircraft3DOF:
    """
    3DOF fixed-wing aircraft dynamics model with pitch control.
    
    Uses point-mass equations with flight path angles.
    Includes pitch control for altitude changes.
    """
    
    def __init__(self, config_file: Optional[str] = None, 
                 config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize aircraft from configuration.
        
        Args:
            config_file: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)['aircraft']
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")
        
        # Extract configuration parameters
        self._load_config()
        
        # Initialize state
        self.state = None
        self.forces = AircraftForces()
        self.mode = FlightMode.NORMAL
        
        # Environmental conditions (will be set by asset manager)
        self.wind_vector = np.zeros(3)
        self.air_density = 1.225  # Sea level standard
        
        # History tracking
        self.state_history = []
        self.max_history = 1000  # Keep last N states
        
    def _load_config(self):
        """Load configuration parameters"""
        # Mass properties
        self.mass = self.config['mass']  # kg
        
        # Aerodynamic properties
        aero = self.config['aerodynamics']
        self.S = aero['reference_area']  # Wing area (m²)
        self.CD0 = aero['cd0']  # Parasitic drag coefficient
        self.k = aero['k']  # Induced drag factor
        self.CL_alpha = aero.get('cl_alpha', 5.0)  # Lift curve slope
        self.CL_max = aero['cl_max']  # Maximum lift coefficient
        self.CL0 = aero.get('cl0', 0.2)  # Zero-angle lift coefficient (NEW!)
        
        # Propulsion
        prop = self.config['propulsion']
        self.thrust_max = prop['thrust_max']  # N
        self.thrust_min = prop.get('thrust_min', 0.0)  # N
        self.sfc = prop.get('sfc', 0.0001)  # Specific fuel consumption
        
        # Performance envelope
        perf = self.config['performance']
        self.v_min = perf['v_min']  # Stall speed
        self.v_max = perf['v_max']  # Max speed
        self.v_cruise = perf['v_cruise']  # Cruise speed
        self.climb_rate_max = perf['climb_rate_max']
        self.load_factor_max = perf['load_factor_max']
        self.service_ceiling = perf.get('service_ceiling', 15000)
        
        # Control limits
        ctrl = self.config['control']
        self.bank_angle_max = np.radians(ctrl['bank_angle_max'])
        self.bank_rate_max = np.radians(ctrl.get('bank_rate_max', 90))
        self.pitch_angle_max = np.radians(ctrl['pitch_angle_max'])
        self.pitch_rate_max = np.radians(ctrl.get('pitch_rate_max', 30))  # NEW!
        self.throttle_rate = ctrl['throttle_rate']
        
        # Fuel
        fuel = self.config['fuel']
        self.fuel_capacity = fuel['capacity']
        self.fuel_initial = fuel['initial']
        
    def initialize_state(self, position: np.ndarray, velocity: float,
                        heading: float, flight_path_angle: float = 0.0,
                        throttle: float = 0.5, fuel: Optional[float] = None):
        """
        Initialize aircraft state.
        
        Args:
            position: Initial position [x, y, z] in meters
            velocity: Initial airspeed in m/s
            heading: Initial heading in radians (0 = East)
            flight_path_angle: Initial climb angle in radians
            throttle: Initial throttle setting [0, 1]
            fuel: Initial fuel (kg), defaults to full
        """
        if fuel is None:
            fuel = self.fuel_initial
            
        self.state = AircraftState(
            position=position.astype(np.float64),
            velocity=velocity,
            heading=heading,
            flight_path_angle=flight_path_angle,
            bank_angle=0.0,
            pitch_angle=flight_path_angle,  # Initialize pitch to match flight path
            throttle=throttle,
            fuel_remaining=fuel,
            time=0.0
        )
        
        # Clear history
        self.state_history = [self.state.copy()]
        
    def set_controls(self, bank_angle: Optional[float] = None,
                    pitch_angle: Optional[float] = None,  # NEW parameter!
                    throttle: Optional[float] = None):
        """
        Set control inputs.
        
        Args:
            bank_angle: Commanded bank angle in radians
            pitch_angle: Commanded pitch angle in radians (NEW!)
            throttle: Commanded throttle [0, 1]
        """
        if bank_angle is not None:
            # Limit bank angle
            self.state.bank_angle = np.clip(
                bank_angle, -self.bank_angle_max, self.bank_angle_max
            )
            
        if pitch_angle is not None:
            # Limit pitch angle
            self.state.pitch_angle = np.clip(
                pitch_angle, -self.pitch_angle_max, self.pitch_angle_max
            )
            
        if throttle is not None:
            # Limit throttle
            self.state.throttle = np.clip(throttle, 0.0, 1.0)
            
    def calculate_forces(self) -> AircraftForces:
        """
        Calculate all forces acting on the aircraft.
        
        Returns:
            AircraftForces object with all force components
        """
        forces = AircraftForces()
        
        # Weight
        forces.weight = self.mass * 9.81
        
        # Thrust
        forces.thrust = self.thrust_min + (self.thrust_max - self.thrust_min) * self.state.throttle
        
        # Dynamic pressure
        q = 0.5 * self.air_density * self.state.velocity ** 2
        
        # Angle of attack (difference between pitch and flight path angle)
        # This is key for altitude control!
        alpha = self.state.pitch_angle - self.state.flight_path_angle
        
        # Limit angle of attack to prevent stall
        alpha_max = np.radians(15)  # Typical stall angle
        alpha = np.clip(alpha, -alpha_max, alpha_max)
        
        # Lift coefficient with pitch influence
        # CL = CL0 + CL_alpha * alpha
        CL = self.CL0 + self.CL_alpha * alpha
        
        # Apply stall model (simple)
        if abs(alpha) > alpha_max * 0.9:
            CL *= 0.7  # Stall reduces lift
            self.mode = FlightMode.STALLED
        else:
            self.mode = FlightMode.NORMAL
            
        # Limit CL
        CL = np.clip(CL, -self.CL_max, self.CL_max)
        
        # Lift force
        forces.lift = q * self.S * CL
        
        # Drag coefficient
        CD = self.CD0 + self.k * CL ** 2
        
        # Drag force
        forces.drag = q * self.S * CD
        
        return forces
        
    def calculate_accelerations(self, forces: AircraftForces) -> Tuple[float, float, float]:
        """
        Calculate accelerations from forces.
        
        Args:
            forces: Current forces on aircraft
            
        Returns:
            Tuple of (dV/dt, dγ/dt, dψ/dt)
        """
        # Velocity rate (along flight path)
        # dV/dt = (T*cos(alpha) - D)/m - g*sin(γ)
        # Account for thrust vector with pitch
        alpha = self.state.pitch_angle - self.state.flight_path_angle
        dV_dt = (forces.thrust * np.cos(alpha) - forces.drag) / self.mass - \
                9.81 * np.sin(self.state.flight_path_angle)
        
        # Flight path angle rate - this controls climb/descent
        # dγ/dt = (L*cos(φ) + T*sin(alpha) - W*cos(γ))/(m*V)
        if self.state.velocity > 0.1:
            dgamma_dt = (forces.lift * np.cos(self.state.bank_angle) + 
                        forces.thrust * np.sin(alpha) -
                        forces.weight * np.cos(self.state.flight_path_angle)) / \
                       (self.mass * self.state.velocity)
        else:
            dgamma_dt = 0.0
            
        # Heading rate (turn rate)
        # dψ/dt = (L*sin(φ))/(m*V*cos(γ))
        if self.state.velocity > 0.1 and np.abs(np.cos(self.state.flight_path_angle)) > 0.1:
            dpsi_dt = (forces.lift * np.sin(self.state.bank_angle)) / \
                     (self.mass * self.state.velocity * np.cos(self.state.flight_path_angle))
        else:
            dpsi_dt = 0.0
            
        # Store in forces object for debugging
        forces.acceleration_velocity = dV_dt
        forces.acceleration_gamma = dgamma_dt
        forces.acceleration_psi = dpsi_dt
        
        return dV_dt, dgamma_dt, dpsi_dt

    def update(self, dt: float, wind: Optional[np.ndarray] = None,
              air_density: Optional[float] = None):
        """
        Update aircraft state by one timestep.
        
        Args:
            dt: Time step in seconds
            wind: Wind vector [vx, vy, vz] in m/s
            air_density: Air density in kg/m³
        """
        if self.state is None:
            raise ValueError("Aircraft state not initialized")
            
        # Don't update if crashed
        if self.mode == FlightMode.CRASHED:
            return
            
        # Update environmental conditions
        if wind is not None:
            self.wind_vector = wind
        if air_density is not None:
            self.air_density = air_density
            
        # Calculate forces
        forces = self.calculate_forces()
        self.forces = forces
        
        # Calculate accelerations
        dV_dt, dgamma_dt, dpsi_dt = self.calculate_accelerations(forces)
        
        # Update velocity and angles (Euler integration)
        self.state.velocity += dV_dt * dt
        self.state.flight_path_angle += dgamma_dt * dt
        self.state.heading += dpsi_dt * dt
        
        # Limit velocity
        self.state.velocity = np.clip(self.state.velocity, 0, self.v_max)
        
        # Limit flight path angle (it will follow pitch angle due to forces)
        self.state.flight_path_angle = np.clip(
            self.state.flight_path_angle,
            -np.radians(30),  # Max dive
            np.radians(30)    # Max climb
        )
        
        # Normalize heading to [0, 2π]
        self.state.heading = self.state.heading % (2 * np.pi)
        
        # Update position
        velocity_vec = self.state.get_velocity_vector()
        self.state.position += velocity_vec * dt
        
        # Update fuel
        if self.sfc > 0:
            fuel_burn = forces.thrust * self.sfc * dt
            self.state.fuel_remaining -= fuel_burn
            self.state.fuel_remaining = max(0, self.state.fuel_remaining)
            
            # Cut thrust if out of fuel
            if self.state.fuel_remaining <= 0:
                self.thrust_max = 0
                
        # Update time
        self.state.time += dt
        
        # Check ground collision
        if self.state.position[2] <= 0:
            self.state.position[2] = 0
            self.mode = FlightMode.GROUND
            if self.state.velocity > 10:  # Crash if moving too fast
                self.mode = FlightMode.CRASHED
                
        # Store in history
        self._update_history()
        
    def _update_history(self):
        """Update state history"""
        self.state_history.append(self.state.copy())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

    def get_turn_radius(self) -> float:
        """
        Calculate current turn radius.
        
        Returns:
            Turn radius in meters
        """
        if abs(self.state.bank_angle) < 0.01:
            return float('inf')
            
        # R = V²/(g*tan(φ))
        return (self.state.velocity ** 2) / (9.81 * np.tan(self.state.bank_angle))
    
    def get_climb_rate(self) -> float:
        """
        Get current climb rate.
        
        Returns:
            Climb rate in m/s (positive = climbing)
        """
        return self.state.velocity * np.sin(self.state.flight_path_angle)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get state as dictionary for external use.
        
        Returns:
            Dictionary containing current state
        """
        if self.state is None:
            return {}
            
        return {
            'position': self.state.position.copy(),
            'velocity': self.state.velocity,
            'heading': self.state.heading,
            'flight_path_angle': self.state.flight_path_angle,
            'bank_angle': self.state.bank_angle,
            'pitch_angle': self.state.pitch_angle,
            'throttle': self.state.throttle,
            'fuel_remaining': self.state.fuel_remaining,
            'time': self.state.time,
            'mode': self.mode.value,
            'climb_rate': self.get_climb_rate(),
            'turn_radius': self.get_turn_radius()
        }
    
    def apply_environmental_effects(self, effects: Dict[str, Any]):
        """
        Apply environmental effects to aircraft.
        
        Args:
            effects: Dictionary with wind, turbulence, density, etc.
        """
        if 'wind' in effects:
            self.wind_vector = np.array(effects['wind'])
        if 'air_density' in effects:
            self.air_density = effects['air_density']
        # Could add turbulence, icing, etc. here