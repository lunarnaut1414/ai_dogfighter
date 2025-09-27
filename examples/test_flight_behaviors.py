"""
Test script for flight controller behaviors.
Demonstrates autonomous flight modes and target behaviors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

from src.battlespace import Battlespace
from src.assets.asset_manager import AssetManager, AssetType
from src.assets.flight_controller import FlightController, BehaviorMode
from src.assets.aircraft_3dof import Aircraft3DOF


def test_waypoint_following():
    """Test waypoint following behavior"""
    print("\n" + "="*60)
    print("Testing Waypoint Following Behavior")
    print("="*60)
    
    # Create battlespace
    battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
    battlespace.generate(seed=42)
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.02)
    
    # Define waypoint pattern (figure-8)
    waypoints = [
        np.array([10000.0, 10000.0, 2000.0]),
        np.array([15000.0, 15000.0, 2500.0]),
        np.array([20000.0, 10000.0, 2000.0]),
        np.array([15000.0, 5000.0, 1500.0]),
        np.array([10000.0, 10000.0, 2000.0]),  # Close the loop
    ]
    
    # Spawn aircraft
    config = {
        'aircraft': 'configs/aircraft/target_basic.yaml',
        'initial_state': {
            'position': [10000.0, 10000.0, 2000.0],
            'velocity': 40.0,
            'heading': 0.0,
            'throttle': 0.6
        }
    }
    
    aircraft_id = asset_manager.spawn_aircraft(
        config=config,
        asset_id="waypoint_follower",
        asset_type=AssetType.TARGET
    )
    
    # Get aircraft and create controller
    aircraft = asset_manager.assets[aircraft_id].aircraft
    controller = FlightController(aircraft.config)
    controller.set_mode(BehaviorMode.WAYPOINT)
    controller.set_waypoints(waypoints)
    
    # Simulation
    positions = []
    times = []
    waypoint_reached = []
    current_wp_idx = 0
    
    print("Time | Position | Current WP | Distance to WP")
    print("-" * 50)
    
    for i in range(2000):  # 40 seconds at 50Hz
        # Get current state
        state = aircraft.state
        
        # Compute control commands
        commands = controller.compute_commands(state)
        
        # Apply commands
        aircraft.set_controls(
            bank_angle=commands.bank_angle,
            throttle=commands.throttle
        )
        
        # Update aircraft with environmental effects
        position = aircraft.state.position
        env_effects = battlespace.get_aircraft_environment_effects(
            position, aircraft.state.get_velocity_vector()
        )
        aircraft.update(
            0.02,
            wind=env_effects['wind_vector'],
            air_density=env_effects['air_density']
        )
        
        # Store data
        positions.append(position.copy())
        times.append(i * 0.02)
        
        # Check waypoint progress
        if controller.current_waypoint_idx != current_wp_idx:
            waypoint_reached.append(i)
            current_wp_idx = controller.current_waypoint_idx
            print(f"Waypoint {current_wp_idx} reached at t={i*0.02:.1f}s")
        
        # Print status
        if i % 50 == 0:  # Every second
            wp = waypoints[controller.current_waypoint_idx % len(waypoints)]
            dist = np.linalg.norm(position - wp)
            print(f"{i*0.02:5.1f} | ({position[0]:6.0f}, {position[1]:6.0f}, {position[2]:5.0f}) | "
                  f"WP{controller.current_waypoint_idx} | {dist:6.1f}m")
    
    positions = np.array(positions)
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # 2D trajectory
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Path')
    for i, wp in enumerate(waypoints):
        ax1.plot(wp[0], wp[1], 'ro', markersize=10)
        ax1.text(wp[0], wp[1], f'WP{i}', fontsize=8)
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Waypoint Following - 2D Path')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Altitude profile
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(times, positions[:, 2], 'b-')
    for wp_idx in waypoint_reached:
        ax2.axvline(x=wp_idx*0.02, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Profile')
    ax2.grid(True)
    
    # 3D trajectory
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.5)
    for wp in waypoints:
        ax3.scatter(wp[0], wp[1], wp[2], c='r', s=100, marker='o')
    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    ax3.set_zlabel('Altitude (m)')
    ax3.set_title('3D Trajectory')
    
    plt.suptitle('Waypoint Following Behavior Test')
    plt.tight_layout()
    plt.show()
    
    return positions, controller


def test_orbit_behavior():
    """Test orbit behavior"""
    print("\n" + "="*60)
    print("Testing Orbit Behavior")
    print("="*60)
    
    # Create battlespace
    battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
    battlespace.generate(seed=42)
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.02)
    
    # Orbit parameters
    orbit_center = np.array([15000.0, 15000.0, 2000.0])
    orbit_radius = 1500.0
    
    # Spawn aircraft at edge of orbit
    start_pos = orbit_center + np.array([orbit_radius, 0, 0])
    
    config = {
        'aircraft': 'configs/aircraft/target_basic.yaml',
        'initial_state': {
            'position': start_pos.tolist(),
            'velocity': 40.0,
            'heading': 90.0,  # North
            'throttle': 0.6
        }
    }
    
    aircraft_id = asset_manager.spawn_aircraft(
        config=config,
        asset_id="orbiter",
        asset_type=AssetType.TARGET
    )
    
    # Get aircraft and create controller
    aircraft = asset_manager.assets[aircraft_id].aircraft
    controller = FlightController(aircraft.config)
    controller.set_mode(BehaviorMode.ORBIT)
    controller.set_orbit(orbit_center, orbit_radius)
    
    # Simulation
    positions = []
    distances_from_center = []
    times = []
    
    print("Time | Position | Distance from center | Radius error")
    print("-" * 60)
    
    for i in range(3000):  # 60 seconds at 50Hz
        # Get current state
        state = aircraft.state
        
        # Compute control commands
        commands = controller.compute_commands(state)
        
        # Apply commands
        aircraft.set_controls(
            bank_angle=commands.bank_angle,
            throttle=commands.throttle
        )
        
        # Update aircraft
        position = aircraft.state.position
        env_effects = battlespace.get_aircraft_environment_effects(
            position, aircraft.state.get_velocity_vector()
        )
        aircraft.update(
            0.02,
            wind=env_effects['wind_vector'],
            air_density=env_effects['air_density']
        )
        
        # Store data
        positions.append(position.copy())
        times.append(i * 0.02)
        
        # Calculate distance from center
        dist = np.linalg.norm(position[:2] - orbit_center[:2])
        distances_from_center.append(dist)
        radius_error = dist - orbit_radius
        
        # Print status
        if i % 50 == 0:  # Every second
            print(f"{i*0.02:5.1f} | ({position[0]:6.0f}, {position[1]:6.0f}, {position[2]:5.0f}) | "
                  f"{dist:7.1f}m | {radius_error:+6.1f}m")
    
    positions = np.array(positions)
    
    # Calculate orbit statistics
    mean_radius = np.mean(distances_from_center[500:])  # Skip initial approach
    std_radius = np.std(distances_from_center[500:])
    
    print(f"\nOrbit Statistics (after stabilization):")
    print(f"  Target radius: {orbit_radius:.1f}m")
    print(f"  Mean radius: {mean_radius:.1f}m")
    print(f"  Std deviation: {std_radius:.1f}m")
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # 2D orbit
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Path')
    circle = plt.Circle((orbit_center[0], orbit_center[1]), orbit_radius, 
                        fill=False, color='r', linestyle='--', label='Target orbit')
    ax1.add_patch(circle)
    ax1.plot(orbit_center[0], orbit_center[1], 'rx', markersize=10, label='Center')
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Orbit Behavior - 2D Path')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Radius tracking
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(times, distances_from_center, 'b-')
    ax2.axhline(y=orbit_radius, color='r', linestyle='--', label='Target radius')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance from center (m)')
    ax2.set_title('Orbit Radius Tracking')
    ax2.legend()
    ax2.grid(True)
    
    # Altitude
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(times, positions[:, 2], 'b-')
    ax3.axhline(y=orbit_center[2], color='r', linestyle='--', label='Target altitude')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Maintenance')
    ax3.legend()
    ax3.grid(True)
    
    plt.suptitle('Orbit Behavior Test')
    plt.tight_layout()
    plt.show()
    
    return positions, controller


def test_evasion_behavior():
    """Test evasion behavior with threat"""
    print("\n" + "="*60)
    print("Testing Evasion Behavior")
    print("="*60)
    
    # Create battlespace
    battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
    battlespace.generate(seed=42)
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.02)
    
    # Spawn target aircraft
    target_config = {
        'aircraft': 'configs/aircraft/target_basic.yaml',
        'initial_state': {
            'position': [15000.0, 15000.0, 2000.0],
            'velocity': 40.0,
            'heading': 0.0,
            'throttle': 0.6
        }
    }
    
    target_id = asset_manager.spawn_aircraft(
        config=target_config,
        asset_id="evader",
        asset_type=AssetType.TARGET
    )
    
    # Spawn threat (interceptor)
    threat_config = {
        'aircraft': 'configs/aircraft/interceptor_drone.yaml',
        'initial_state': {
            'position': [10000.0, 10000.0, 2000.0],
            'velocity': 50.0,
            'heading': 45.0,  # Toward target
            'throttle': 0.7
        }
    }
    
    threat_id = asset_manager.spawn_aircraft(
        config=threat_config,
        asset_id="threat",
        asset_type=AssetType.INTERCEPTOR
    )
    
    # Get aircraft and controllers
    target_aircraft = asset_manager.assets[target_id].aircraft
    threat_aircraft = asset_manager.assets[threat_id].aircraft
    
    target_controller = FlightController(target_aircraft.config)
    target_controller.set_mode(BehaviorMode.EVADE)
    target_controller.set_evasion_parameters(distance=3000.0, aggressiveness=0.8)
    
    threat_controller = FlightController(threat_aircraft.config)
    threat_controller.set_mode(BehaviorMode.PURSUIT)
    
    # Simulation
    target_positions = []
    threat_positions = []
    ranges = []
    times = []
    
    print("Time | Target Pos | Threat Pos | Range | Target Mode")
    print("-" * 60)
    
    for i in range(2500):  # 50 seconds at 50Hz
        # Get states
        target_state = target_aircraft.state
        threat_state = threat_aircraft.state
        
        # Calculate range
        range_to_threat = np.linalg.norm(
            target_state.position - threat_state.position
        )
        
        # Target evasion
        target_commands = target_controller.compute_commands(
            target_state,
            threat=threat_state.position
        )
        target_aircraft.set_controls(
            bank_angle=target_commands.bank_angle,
            throttle=target_commands.throttle
        )
        
        # Threat pursuit
        threat_commands = threat_controller.compute_commands(
            threat_state,
            target=target_state.position
        )
        threat_aircraft.set_controls(
            bank_angle=threat_commands.bank_angle,
            throttle=threat_commands.throttle
        )
        
        # Update both aircraft
        for aircraft in [target_aircraft, threat_aircraft]:
            position = aircraft.state.position
            env_effects = battlespace.get_aircraft_environment_effects(
                position, aircraft.state.get_velocity_vector()
            )
            aircraft.update(
                0.02,
                wind=env_effects['wind_vector'],
                air_density=env_effects['air_density']
            )
        
        # Store data
        target_positions.append(target_state.position.copy())
        threat_positions.append(threat_state.position.copy())
        ranges.append(range_to_threat)
        times.append(i * 0.02)
        
        # Print status
        if i % 50 == 0:  # Every second
            print(f"{i*0.02:5.1f} | "
                  f"({target_state.position[0]:6.0f}, {target_state.position[1]:6.0f}) | "
                  f"({threat_state.position[0]:6.0f}, {threat_state.position[1]:6.0f}) | "
                  f"{range_to_threat:7.1f}m | {target_commands.mode}")
        
        # Check if caught
        if range_to_threat < 100:
            print(f"\n*** Target caught at t={i*0.02:.1f}s ***")
            break
    
    target_positions = np.array(target_positions)
    threat_positions = np.array(threat_positions)
    
    # Visualize
    fig = plt.figure(figsize=(15, 10))
    
    # 2D chase
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             alpha=0.5, label='Target (evading)')
    ax1.plot(threat_positions[:, 0], threat_positions[:, 1], 'r-', 
             alpha=0.5, label='Threat (pursuing)')
    ax1.plot(target_positions[0, 0], target_positions[0, 1], 'bo', markersize=10)
    ax1.plot(threat_positions[0, 0], threat_positions[0, 1], 'ro', markersize=10)
    ax1.plot(target_positions[-1, 0], target_positions[-1, 1], 'b*', markersize=15)
    ax1.plot(threat_positions[-1, 0], threat_positions[-1, 1], 'r*', markersize=15)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Evasion vs Pursuit - 2D')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Range over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, ranges, 'g-', linewidth=2)
    ax2.axhline(y=3000, color='orange', linestyle='--', 
                label='Evasion trigger distance')
    ax2.axhline(y=100, color='r', linestyle='--', label='Capture distance')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('Separation Distance')
    ax2.legend()
    ax2.grid(True)
    
    # Altitudes
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, target_positions[:, 2], 'b-', label='Target')
    ax3.plot(times, threat_positions[:, 2], 'r-', label='Threat')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profiles')
    ax3.legend()
    ax3.grid(True)
    
    # 3D chase
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(target_positions[:, 0], target_positions[:, 1], 
             target_positions[:, 2], 'b-', alpha=0.5, label='Target')
    ax4.plot(threat_positions[:, 0], threat_positions[:, 1], 
             threat_positions[:, 2], 'r-', alpha=0.5, label='Threat')
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    ax4.set_zlabel('Altitude (m)')
    ax4.set_title('3D Chase Trajectories')
    ax4.legend()
    
    # Speed profiles
    ax5 = plt.subplot(2, 3, 5)
    target_speeds = [np.linalg.norm(target_positions[i+1] - target_positions[i]) / 0.02
                    if i < len(times)-1 else 0 for i in range(len(times))]
    threat_speeds = [np.linalg.norm(threat_positions[i+1] - threat_positions[i]) / 0.02
                    if i < len(times)-1 else 0 for i in range(len(times))]
    ax5.plot(times, target_speeds, 'b-', label='Target', alpha=0.7)
    ax5.plot(times, threat_speeds, 'r-', label='Threat', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed Profiles')
    ax5.legend()
    ax5.grid(True)
    
    # Heading changes
    ax6 = plt.subplot(2, 3, 6)
    target_headings = []
    threat_headings = []
    for i in range(1, len(target_positions)):
        # Calculate heading from velocity
        target_vel = target_positions[i] - target_positions[i-1]
        threat_vel = threat_positions[i] - threat_positions[i-1]
        target_heading = np.arctan2(target_vel[1], target_vel[0])
        threat_heading = np.arctan2(threat_vel[1], threat_vel[0])
        target_headings.append(np.degrees(target_heading))
        threat_headings.append(np.degrees(threat_heading))
    
    ax6.plot(times[1:], target_headings, 'b-', label='Target', alpha=0.7)
    ax6.plot(times[1:], threat_headings, 'r-', label='Threat', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Heading (degrees)')
    ax6.set_title('Heading Changes')
    ax6.legend()
    ax6.grid(True)
    
    plt.suptitle('Evasion Behavior Test')
    plt.tight_layout()
    plt.show()
    
    return target_positions, threat_positions


def test_behavior_transitions():
    """Test transitions between different behaviors"""
    print("\n" + "="*60)
    print("Testing Behavior Transitions")
    print("="*60)
    
    # Create battlespace
    battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
    battlespace.generate(seed=42)
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.02)
    
    # Spawn aircraft
    config = {
        'aircraft': 'configs/aircraft/target_basic.yaml',
        'initial_state': {
            'position': [15000.0, 15000.0, 2000.0],
            'velocity': 40.0,
            'heading': 0.0,
            'throttle': 0.6
        }
    }
    
    aircraft_id = asset_manager.spawn_aircraft(
        config=config,
        asset_id="multi_behavior",
        asset_type=AssetType.TARGET
    )
    
    aircraft = asset_manager.assets[aircraft_id].aircraft
    controller = FlightController(aircraft.config)
    
    # Define behavior sequence
    behaviors = [
        (BehaviorMode.WAYPOINT, 200),  # 4 seconds
        (BehaviorMode.ORBIT, 300),      # 6 seconds
        (BehaviorMode.EVADE, 200),      # 4 seconds
        (BehaviorMode.PATROL, 300),     # 6 seconds
    ]
    
    # Set up waypoints for waypoint mode
    waypoints = [
        np.array([20000.0, 20000.0, 2500.0]),
        np.array([25000.0, 15000.0, 2000.0]),
    ]
    controller.set_waypoints(waypoints)
    
    # Set up orbit for orbit mode
    controller.set_orbit(np.array([20000.0, 20000.0, 2000.0]), 1000.0)
    
    # Simulation
    positions = []
    times = []
    behavior_labels = []
    
    behavior_idx = 0
    step_count = 0
    
    print("Time | Position | Behavior Mode")
    print("-" * 40)
    
    for i in range(1000):  # 20 seconds total
        # Switch behavior if needed
        if behavior_idx < len(behaviors):
            mode, duration = behaviors[behavior_idx]
            if step_count == 0:
                controller.set_mode(mode)
                print(f"\n*** Switching to {mode.value} mode ***")
            
            step_count += 1
            if step_count >= duration:
                step_count = 0
                behavior_idx += 1
        
        # Get state
        state = aircraft.state
        
        # Compute control based on current mode
        if controller.mode == BehaviorMode.EVADE:
            # Simulate a threat for evasion
            threat_pos = state.position + np.array([-2000, -2000, 0])
            commands = controller.compute_commands(state, threat=threat_pos)
        else:
            commands = controller.compute_commands(state)
        
        # Apply commands
        aircraft.set_controls(
            bank_angle=commands.bank_angle,
            throttle=commands.throttle
        )
        
        # Update aircraft
        position = aircraft.state.position
        env_effects = battlespace.get_aircraft_environment_effects(
            position, aircraft.state.get_velocity_vector()
        )
        aircraft.update(
            0.02,
            wind=env_effects['wind_vector'],
            air_density=env_effects['air_density']
        )
        
        # Store data
        positions.append(position.copy())
        times.append(i * 0.02)
        behavior_labels.append(controller.mode.value)
        
        # Print status
        if i % 50 == 0:  # Every second
            print(f"{i*0.02:5.1f} | ({position[0]:6.0f}, {position[1]:6.0f}, {position[2]:5.0f}) | "
                  f"{controller.mode.value}")
    
    positions = np.array(positions)
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # 2D trajectory colored by behavior
    ax1 = plt.subplot(1, 3, 1)
    
    # Color segments by behavior
    behavior_colors = {
        'waypoint': 'blue',
        'orbit': 'green',
        'evade': 'red',
        'patrol': 'orange'
    }
    
    current_behavior = behavior_labels[0]
    segment_start = 0
    
    for i, behavior in enumerate(behavior_labels):
        if behavior != current_behavior or i == len(behavior_labels) - 1:
            # Plot segment
            color = behavior_colors.get(current_behavior, 'black')
            ax1.plot(positions[segment_start:i+1, 0], 
                    positions[segment_start:i+1, 1],
                    color=color, linewidth=2, label=current_behavior, alpha=0.7)
            current_behavior = behavior
            segment_start = i
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Multi-Behavior Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    # Altitude over time
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(times, positions[:, 2], 'b-')
    
    # Add behavior transition lines
    transition_times = []
    cumulative_time = 0
    for mode, duration in behaviors:
        cumulative_time += duration * 0.02
        transition_times.append(cumulative_time)
    
    for t_time in transition_times[:-1]:
        ax2.axvline(x=t_time, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Profile')
    ax2.grid(True)
    
    # 3D trajectory
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.5)
    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    ax3.set_zlabel('Altitude (m)')
    ax3.set_title('3D Trajectory')
    
    plt.suptitle('Behavior Transition Test')
    plt.tight_layout()
    plt.show()
    
    return positions, behavior_labels


def main():
    """Main test execution"""
    print("="*60)
    print("Flight Controller Behavior Tests")
    print("="*60)
    
    # Test 1: Waypoint following
    print("\n1. Waypoint Following Test")
    positions_wp, controller_wp = test_waypoint_following()
    
    # Test 2: Orbit behavior
    print("\n2. Orbit Behavior Test")
    positions_orbit, controller_orbit = test_orbit_behavior()
    
    # Test 3: Evasion behavior
    print("\n3. Evasion Behavior Test")
    target_pos, threat_pos = test_evasion_behavior()
    
    # Test 4: Behavior transitions
    print("\n4. Behavior Transition Test")
    positions_multi, behaviors = test_behavior_transitions()
    
    print("\n" + "="*60)
    print("All behavior tests complete!")
    print("="*60)
    
    # Summary statistics
    print("\nTest Summary:")
    print(f"  Waypoint test: {len(positions_wp)} steps simulated")
    print(f"  Orbit test: {len(positions_orbit)} steps simulated")
    print(f"  Evasion test: {len(target_pos)} steps simulated")
    print(f"  Transition test: {len(set(behaviors))} unique behaviors tested")


if __name__ == "__main__":
    main()