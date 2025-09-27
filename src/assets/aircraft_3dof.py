# src/assets/aircraft_3dof.py
"""
3DOF (Three Degree of Freedom) fixed-wing aircraft dynamics model.
Point-mass model with flight path angles for trajectory simulation.
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
    3DOF fixed-wing aircraft dynamics model.
    
    Uses point-mass equations with flight path angles.
    Suitable for trajectory analysis and guidance algorithm testing.
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
            heading: Initial heading in radians (0 = North)
            flight_path_angle: Initial climb angle in radians
            throttle: Initial throttle setting [0, 1]
            fuel: Initial fuel (kg), defaults to full
        """
        if fuel is None:
            fuel = self.fuel_initial
            
        self.state = AircraftState(
            position=position.astype(np.float64),  # Force float64
            velocity=velocity,
            heading=heading,
            flight_path_angle=flight_path_angle,
            bank_angle=0.0,
            throttle=throttle,
            fuel_remaining=fuel,
            time=0.0
        )
        
        # Clear history
        self.state_history = [self.state.copy()]
        
    def set_controls(self, bank_angle: Optional[float] = None,
                    throttle: Optional[float] = None):
        """
        Set control inputs.
        
        Args:
            bank_angle: Commanded bank angle in radians
            throttle: Commanded throttle [0, 1]
        """
        if bank_angle is not None:
            # Limit bank angle
            self.state.bank_angle = np.clip(
                bank_angle, -self.bank_angle_max, self.bank_angle_max
            )
            
        if throttle is not None:
            # Limit throttle
            self.state.throttle = np.clip(throttle, 0.0, 1.0)
            
    def calculate_forces(self) -> AircraftForces:
            """
            Calculate all forces acting on the aircraft.
            
            Returns:
                AircraftForces object with lift, drag, thrust, weight
            """
            forces = AircraftForces()
            
            # Weight
            forces.weight = self.mass * 9.81
            
            # Get true airspeed (accounting for wind)
            velocity_ground = self.state.get_velocity_vector()
            velocity_air = velocity_ground - self.wind_vector
            v_true = np.linalg.norm(velocity_air)
            
            # Dynamic pressure
            q = 0.5 * self.air_density * v_true ** 2
            
            # Thrust
            forces.thrust = self.state.throttle * self.thrust_max
            
            # Lift coefficient (simplified - assumes coordinated flight)
            # In steady flight: L = W/cos(φ)
            CL = forces.weight * np.abs(np.cos(self.state.bank_angle)) / (q * self.S)
            
            # Check for stall
            if CL > self.CL_max or v_true < self.v_min:
                CL = self.CL_max * 0.5  # Stalled lift
                self.mode = FlightMode.STALLED
            else:
                self.mode = FlightMode.NORMAL
                
            # Limit CL
            CL = np.clip(CL, 0, self.CL_max)
            
            # Lift
            forces.lift = q * self.S * CL
            
            # Drag coefficient (parabolic polar)
            CD = self.CD0 + self.k * CL ** 2
            
            # Drag
            forces.drag = q * self.S * CD
            
            self.forces = forces
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
        # dV/dt = (T - D)/m - g*sin(γ)
        dV_dt = (forces.thrust - forces.drag) / self.mass - \
                9.81 * np.sin(self.state.flight_path_angle)
        
        # Flight path angle rate
        # dγ/dt = (L*cos(φ) - W*cos(γ))/(m*V)
        if self.state.velocity > 0.1:  # Avoid division by zero
            dgamma_dt = (forces.lift * np.cos(self.state.bank_angle) - 
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
        
        # Calculate accelerations
        dV_dt, dgamma_dt, dpsi_dt = self.calculate_accelerations(forces)
        
        # Update velocity and angles (Euler integration)
        self.state.velocity += dV_dt * dt
        self.state.flight_path_angle += dgamma_dt * dt
        self.state.heading += dpsi_dt * dt
        
        # Limit velocity
        self.state.velocity = np.clip(self.state.velocity, 0, self.v_max)
        
        # Limit flight path angle
        self.state.flight_path_angle = np.clip(
            self.state.flight_path_angle,
            -self.pitch_angle_max,
            self.pitch_angle_max
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
            Climb rate in m/s
        """
        return self.state.velocity * np.sin(self.state.flight_path_angle)
    
    def get_range_rate(self, target_position: np.ndarray) -> float:
        """
        Calculate range rate to target (closing velocity).
        
        Args:
            target_position: Target position [x, y, z]
            
        Returns:
            Range rate in m/s (negative = closing)
        """
        # Vector to target
        los_vector = target_position - self.state.position
        range_to_target = np.linalg.norm(los_vector)
        
        if range_to_target < 0.1:
            return 0.0
            
        # Unit vector to target
        los_unit = los_vector / range_to_target
        
        # Project velocity onto LOS
        velocity_vec = self.state.get_velocity_vector()
        range_rate = np.dot(velocity_vec, los_unit)
        
        return range_rate
    
    def get_energy_state(self) -> Dict[str, float]:
        """
        Calculate energy state parameters.
        
        Returns:
            Dictionary with energy metrics
        """
        ke = self.get_kinetic_energy()
        pe = self.get_potential_energy()
        
        return {
            'kinetic_energy': ke,
            'potential_energy': pe,
            'total_energy': ke + pe,
            'specific_energy': (ke + pe) / self.mass,  # Energy per unit mass
            'energy_height': (ke + pe) / (self.mass * 9.81)  # Equivalent altitude
        }
    
    def get_kinetic_energy(self) -> float:
        """Calculate kinetic energy in Joules"""
        return 0.5 * self.mass * self.state.velocity ** 2
    
    def get_potential_energy(self) -> float:
        """Calculate potential energy in Joules"""
        return self.mass * 9.81 * self.state.position[2]
    
    def set_waypoint_commands(self, waypoint: np.ndarray, 
                             desired_speed: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate control commands to fly to waypoint.
        Simple proportional navigation.
        
        Args:
            waypoint: Target position [x, y, z]
            desired_speed: Desired airspeed (m/s)
            
        Returns:
            Tuple of (bank_angle, throttle) commands
        """
        # Vector to waypoint
        delta = waypoint - self.state.position
        range_to_wp = np.linalg.norm(delta[:2])  # Horizontal range
        
        if range_to_wp < 10:  # Within 10m, consider reached
            return 0.0, self.state.throttle
            
        # Desired heading to waypoint
        desired_heading = np.arctan2(delta[1], delta[0])
        
        # Heading error (with wrapping)
        heading_error = desired_heading - self.state.heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
            
        # Bank angle command (proportional control)
        k_bank = 1.0  # Gain
        bank_cmd = k_bank * heading_error
        bank_cmd = np.clip(bank_cmd, -self.bank_angle_max, self.bank_angle_max)
        
        # Altitude control via flight path angle
        altitude_error = waypoint[2] - self.state.position[2]
        desired_climb_rate = np.clip(altitude_error * 0.1, 
                                     -self.climb_rate_max, 
                                     self.climb_rate_max)
        
        # Speed control via throttle
        if desired_speed is None:
            desired_speed = self.v_cruise
            
        speed_error = desired_speed - self.state.velocity
        throttle_cmd = self.state.throttle + speed_error * 0.01
        throttle_cmd = np.clip(throttle_cmd, 0, 1)
        
        return bank_cmd, throttle_cmd
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get state as a vector for recording/analysis.
        
        Returns:
            State vector [x, y, z, V, γ, ψ, φ, throttle, fuel]
        """
        return np.array([
            self.state.position[0],
            self.state.position[1], 
            self.state.position[2],
            self.state.velocity,
            self.state.flight_path_angle,
            self.state.heading,
            self.state.bank_angle,
            self.state.throttle,
            self.state.fuel_remaining
        ])
    
    def __str__(self) -> str:
        """String representation of aircraft state"""
        if self.state is None:
            return "Aircraft3DOF (uninitialized)"
            
        return (f"Aircraft3DOF: Pos=({self.state.position[0]:.0f}, "
                f"{self.state.position[1]:.0f}, {self.state.position[2]:.0f})m, "
                f"V={self.state.velocity:.1f}m/s, "
                f"Heading={np.degrees(self.state.heading):.0f}°, "
                f"Climb={self.get_climb_rate():.1f}m/s, "
                f"Mode={self.mode.value}")