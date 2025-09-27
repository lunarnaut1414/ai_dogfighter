# examples/test_aircraft_dynamics.py
"""
Test script for 3DOF aircraft dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.assets.aircraft_3dof import Aircraft3DOF


def test_straight_flight():
    """Test straight and level flight"""
    # Load configuration
    aircraft = Aircraft3DOF(config_file="configs/aircraft/interceptor_drone.yaml")
    
    # Initialize state
    aircraft.initialize_state(
        position=np.array([0, 0, 1000]),
        velocity=50.0,
        heading=0.0,
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Simulate for 60 seconds
    dt = 0.02
    times = []
    positions = []
    velocities = []
    
    for i in range(3000):  # 60 seconds
        aircraft.update(dt)
        times.append(aircraft.state.time)
        positions.append(aircraft.state.position.copy())
        velocities.append(aircraft.state.velocity)
        
    positions = np.array(positions)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1])
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Horizontal Trajectory')
    ax.grid(True)
    ax.axis('equal')
    
    # Altitude
    ax = axes[0, 1]
    ax.plot(times, positions[:, 2])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile')
    ax.grid(True)
    
    # Velocity
    ax = axes[1, 0]
    ax.plot(times, velocities)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Airspeed')
    ax.grid(True)
    
    # 3D Trajectory
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Trajectory')
    
    plt.suptitle('Straight Flight Test')
    plt.tight_layout()
    plt.show()


def test_circular_orbit():
    """Test circular orbit maneuver"""
    aircraft = Aircraft3DOF(config_file="configs/aircraft/interceptor_drone.yaml")
    
    aircraft.initialize_state(
        position=np.array([1000, 0, 1000]),
        velocity=50.0,
        heading=np.pi/2,  # Facing north
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Set constant bank angle for orbit
    bank_angle = np.radians(30)
    
    # Simulate
    dt = 0.02
    positions = []
    
    for i in range(5000):  # 100 seconds
        aircraft.set_controls(bank_angle=bank_angle, throttle=0.5)
        aircraft.update(dt)
        positions.append(aircraft.state.position.copy())
        
    positions = np.array(positions)
    
    # Plot orbit
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(positions[:, 0], positions[:, 1])
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'Circular Orbit (Bank = {np.degrees(bank_angle):.0f}°)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    
    # Calculate and display turn radius
    theoretical_radius = aircraft.get_turn_radius()
    ax.text(0.05, 0.95, f'Theoretical Radius: {theoretical_radius:.0f}m',
            transform=ax.transAxes, verticalalignment='top')
    
    plt.show()


def test_energy_management():
    """Test energy trading between altitude and speed"""
    aircraft = Aircraft3DOF(config_file="configs/aircraft/interceptor_drone.yaml")
    
    aircraft.initialize_state(
        position=np.array([0, 0, 2000]),
        velocity=40.0,
        heading=0.0,
        flight_path_angle=0.0,
        throttle=0.0  # No thrust - pure energy trade
    )
    
    # Disable thrust to see pure energy exchange
    aircraft.thrust_max = 0
    
    dt = 0.02
    times = []
    altitudes = []
    velocities = []
    total_energies = []
    
    # Start with a dive
    aircraft.state.flight_path_angle = np.radians(-10)
    
    for i in range(2500):  # 50 seconds
        # Pull up when getting low
        if aircraft.state.position[2] < 1000:
            aircraft.state.flight_path_angle = np.radians(10)
        # Dive when getting slow
        elif aircraft.state.velocity < 30:
            aircraft.state.flight_path_angle = np.radians(-10)
            
        aircraft.update(dt)
        
        times.append(aircraft.state.time)
        altitudes.append(aircraft.state.position[2])
        velocities.append(aircraft.state.velocity)
        
        energy = aircraft.get_energy_state()
        total_energies.append(energy['total_energy'])
        
    # Plot energy exchange
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(times, altitudes)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude')
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(times, velocities)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Airspeed')
    ax.grid(True)
    
    ax = axes[1, 0]
    ax.plot(times, total_energies)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Energy (J)')
    ax.set_title('Total Energy (Should be ~constant)')
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(velocities, altitudes)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Energy State Space')
    ax.grid(True)
    
    plt.suptitle('Energy Management Test')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Testing 3DOF Aircraft Dynamics")
    print("=" * 50)
    
    print("\n1. Straight Flight Test")
    test_straight_flight()
    
    print("\n2. Circular Orbit Test")
    test_circular_orbit()
    
    print("\n3. Energy Management Test")
    test_energy_management()
    
    print("\nAll tests complete!")