# tests/unit/test_aircraft_3dof.py
"""
Unit tests for 3DOF aircraft dynamics model.
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.assets.aircraft_3dof import Aircraft3DOF, AircraftState, FlightMode


class TestAircraft3DOF:
    """Test suite for 3DOF aircraft dynamics"""
    
    @pytest.fixture
    def aircraft(self):
        """Create test aircraft instance"""
        config = {
            'mass': 100.0,
            'aerodynamics': {
                'reference_area': 2.0,
                'cd0': 0.025,
                'k': 0.04,
                'cl_alpha': 5.0,
                'cl_max': 1.4
            },
            'propulsion': {
                'thrust_max': 500.0,
                'thrust_min': 0.0,
                'sfc': 0.0001
            },
            'performance': {
                'v_min': 20.0,
                'v_max': 80.0,
                'v_cruise': 50.0,
                'climb_rate_max': 10.0,
                'load_factor_max': 4.0,
                'service_ceiling': 10000.0
            },
            'control': {
                'bank_angle_max': 60.0,
                'bank_rate_max': 90.0,
                'pitch_angle_max': 20.0,
                'throttle_rate': 0.5
            },
            'fuel': {
                'capacity': 20.0,
                'initial': 20.0
            }
        }
        
        aircraft = Aircraft3DOF(config_dict=config)
        aircraft.initialize_state(
            position=np.array([0, 0, 1000]),
            velocity=50.0,
            heading=0.0,
            flight_path_angle=0.0,
            throttle=0.5
        )
        return aircraft
    
    def test_initialization(self, aircraft):
        """Test aircraft initialization"""
        assert aircraft.state.position[2] == 1000.0
        assert aircraft.state.velocity == 50.0
        assert aircraft.state.heading == 0.0
        assert aircraft.state.throttle == 0.5
        assert aircraft.state.fuel_remaining == 20.0
        
    def test_velocity_vector(self, aircraft):
        """Test velocity vector calculation"""
        # North heading, level flight
        v_vec = aircraft.state.get_velocity_vector()
        assert np.isclose(v_vec[0], 50.0)  # East component
        assert np.isclose(v_vec[1], 0.0)   # North component
        assert np.isclose(v_vec[2], 0.0)   # Vertical component
        
        # East heading (90 degrees)
        aircraft.state.heading = np.pi/2
        v_vec = aircraft.state.get_velocity_vector()
        assert np.isclose(v_vec[0], 0.0, atol=1e-10)
        assert np.isclose(v_vec[1], 50.0)
        
        # Climbing at 30 degrees
        aircraft.state.flight_path_angle = np.radians(30)
        v_vec = aircraft.state.get_velocity_vector()
        assert np.isclose(v_vec[2], 25.0)  # V*sin(30°) = 50*0.5
        
    def test_energy_conservation(self, aircraft):
        """Test energy conservation without thrust/drag"""
        # Disable thrust and drag for pure energy exchange
        aircraft.thrust_max = 0
        aircraft.CD0 = 0
        aircraft.k = 0
        aircraft.state.throttle = 0
        
        # Give it some climb angle to trade energy
        aircraft.state.flight_path_angle = np.radians(10)
        
        initial_energy = aircraft.get_kinetic_energy() + aircraft.get_potential_energy()
        
        # Simulate for 10 seconds
        dt = 0.02
        for _ in range(500):
            aircraft.update(dt)
            
        final_energy = aircraft.get_kinetic_energy() + aircraft.get_potential_energy()
        
        # Energy should be conserved (within numerical tolerance)
        assert abs(final_energy - initial_energy) / initial_energy < 0.01
        
    def test_turn_radius(self, aircraft):
        """Test turn radius calculation"""
        # Level turn at 30 degrees bank
        aircraft.state.bank_angle = np.radians(30)
        radius = aircraft.get_turn_radius()
        
        # Theoretical: R = V²/(g*tan(φ))
        expected = (50.0**2) / (9.81 * np.tan(np.radians(30)))
        assert np.isclose(radius, expected)
        
    def test_climb_performance(self, aircraft):
        """Test climb rate calculation"""
        aircraft.state.flight_path_angle = np.radians(10)
        climb_rate = aircraft.get_climb_rate()
        
        expected = 50.0 * np.sin(np.radians(10))
        assert np.isclose(climb_rate, expected)
        
    def test_stall_detection(self, aircraft):
        """Test stall at low speeds"""
        aircraft.state.velocity = 15.0  # Below stall speed
        forces = aircraft.calculate_forces()
        
        assert aircraft.mode == FlightMode.STALLED
        
    def test_fuel_consumption(self, aircraft):
        """Test fuel burn over time"""
        initial_fuel = aircraft.state.fuel_remaining
        aircraft.state.throttle = 1.0  # Max thrust
        
        # Run for 10 seconds
        dt = 0.1
        for _ in range(100):
            aircraft.update(dt)
            
        # Fuel should have decreased
        assert aircraft.state.fuel_remaining < initial_fuel
        
        # Verify burn rate
        fuel_burned = initial_fuel - aircraft.state.fuel_remaining
        expected_burn = aircraft.thrust_max * aircraft.sfc * 10.0
        assert np.isclose(fuel_burned, expected_burn, rtol=0.01)
        
    def test_ground_collision(self, aircraft):
        """Test ground collision detection"""
        # Put aircraft near ground
        aircraft.state.position[2] = 10.0
        aircraft.state.flight_path_angle = np.radians(-45)  # Diving
        
        # Simulate until ground contact
        dt = 0.02
        for _ in range(100):
            aircraft.update(dt)
            if aircraft.mode in [FlightMode.GROUND, FlightMode.CRASHED]:
                break
                
        assert aircraft.state.position[2] == 0.0
        assert aircraft.mode in [FlightMode.GROUND, FlightMode.CRASHED]
        
    def test_control_limits(self, aircraft):
        """Test control surface limits"""
        # Try to exceed bank angle limit
        aircraft.set_controls(bank_angle=np.radians(90))
        assert aircraft.state.bank_angle == aircraft.bank_angle_max
        
        # Try to exceed throttle limits
        aircraft.set_controls(throttle=1.5)
        assert aircraft.state.throttle == 1.0
        
        aircraft.set_controls(throttle=-0.5)
        assert aircraft.state.throttle == 0.0
        
    def test_waypoint_navigation(self, aircraft):
        """Test simple waypoint navigation commands"""
        waypoint = np.array([1000, 1000, 1500])
        bank_cmd, throttle_cmd = aircraft.set_waypoint_commands(waypoint)
        
        # Should command a right turn (positive bank)
        assert bank_cmd > 0
        
        # Should climb (waypoint is higher)
        # This would be reflected in the flight path angle in a full implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])