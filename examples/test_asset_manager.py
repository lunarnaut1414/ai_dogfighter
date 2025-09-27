"""
Test script for integrated Asset Manager with Battlespace.
Demonstrates aircraft simulation with environmental effects.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import time

from src.battlespace import Battlespace
from src.assets.asset_manager import AssetManager, AssetType
from src.assets.flight_controller import FlightController, BehaviorMode


def load_scenario(scenario_file: str) -> dict:
    """Load scenario configuration from file"""
    with open(scenario_file, 'r') as f:
        return yaml.safe_load(f)


def run_single_target_scenario():
    """Run the single target interception scenario"""
    print("\n" + "="*60)
    print("Running Single Target Scenario")
    print("="*60)
    
    # Load scenario
    scenario = load_scenario("configs/scenarios/single_target.yaml")['scenario']
    
    # Create battlespace
    print("\nInitializing Battlespace...")
    battlespace = Battlespace(config_file=scenario['battlespace'])
    battlespace.generate(seed=42)
    
    # Create asset manager
    print("Creating Asset Manager...")
    asset_manager = AssetManager(battlespace, dt=1.0/scenario['update_rate'])
    
    # Spawn interceptor
    print("\nSpawning interceptor...")
    interceptor_config = scenario['interceptor'].copy()  # Make a copy to avoid modifying original
    
    # The config has 'aircraft' key pointing to a file path
    interceptor_id = asset_manager.spawn_aircraft(
        config=interceptor_config,
        asset_id=interceptor_config['id'],
        asset_type=AssetType.INTERCEPTOR,
        team=interceptor_config.get('team', 'blue')
    )
    
    # Spawn target
    print("Spawning target...")
    target_config = scenario['targets'][0].copy()  # Make a copy
    
    target_id = asset_manager.spawn_aircraft(
        config=target_config,
        asset_id=target_config['id'],
        asset_type=AssetType.TARGET,
        team=target_config.get('team', 'red')
    )
    
    # Setup flight controller for target
    target_aircraft = asset_manager.assets[target_id].aircraft
    target_controller = FlightController(target_aircraft.config)
    target_controller.set_mode(BehaviorMode.WAYPOINT)
    waypoints = [np.array(wp, dtype=np.float64) for wp in target_config['waypoints']]
    target_controller.set_waypoints(waypoints)
    
    # Simulation parameters
    max_time = scenario['duration']
    dt = 1.0 / scenario['update_rate']
    
    # Storage for visualization
    interceptor_positions = []
    target_positions = []
    times = []
    
    # Main simulation loop
    print(f"\nStarting simulation for {max_time} seconds...")
    print("Time | Interceptor Pos | Target Pos | Range | Closing")
    print("-" * 60)
    
    start_real_time = time.time()
    
    while asset_manager.time < max_time:
        # Get states
        interceptor_state = asset_manager.get_asset_state(interceptor_id)
        target_state = asset_manager.get_asset_state(target_id)
        
        if interceptor_state is None or target_state is None:
            break
            
        # Calculate relative state
        rel_state = asset_manager.get_relative_state(interceptor_id, target_id)
        
        # Simple pursuit guidance for interceptor
        if rel_state:
            # Calculate bank angle to turn toward target
            bearing_error = rel_state['bearing'] - interceptor_state.heading
            while bearing_error > np.pi:
                bearing_error -= 2 * np.pi
            while bearing_error < -np.pi:
                bearing_error += 2 * np.pi
                
            bank_cmd = np.clip(bearing_error * 1.0, -1.0, 1.0)
            
            # Throttle based on range
            if rel_state['range'] > 1000:
                throttle_cmd = 0.8
            else:
                throttle_cmd = 0.6
                
            asset_manager.apply_commands(interceptor_id, bank_cmd, throttle_cmd)
        
        # Update target behavior
        target_cmd = target_controller.compute_commands(target_state)
        asset_manager.apply_commands(target_id, target_cmd.bank_angle, target_cmd.throttle)
        
        # Update all assets
        asset_manager.update()
        
        # Store positions
        interceptor_positions.append(interceptor_state.position.copy())
        target_positions.append(target_state.position.copy())
        times.append(asset_manager.time)
        
        # Print status every second
        if int(asset_manager.time) % 1 == 0 and asset_manager.time - int(asset_manager.time) < dt:
            print(f"{asset_manager.time:5.1f} | "
                  f"({interceptor_state.position[0]:6.0f}, {interceptor_state.position[1]:6.0f}, {interceptor_state.position[2]:5.0f}) | "
                  f"({target_state.position[0]:6.0f}, {target_state.position[1]:6.0f}, {target_state.position[2]:5.0f}) | "
                  f"{rel_state['range']:7.1f} | {rel_state['closing_velocity']:6.1f}")
        
        # Check intercept
        if rel_state and rel_state['range'] < 100:
            print(f"\n*** INTERCEPT! Target intercepted at {asset_manager.time:.1f} seconds ***")
            break
            
    # Simulation complete
    elapsed_real = time.time() - start_real_time
    print(f"\nSimulation complete. Real time: {elapsed_real:.1f}s, Sim time: {asset_manager.time:.1f}s")
    print(f"Real-time factor: {asset_manager.time/elapsed_real:.2f}x")
    
    # Performance stats
    stats = asset_manager.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Mean update time: {stats['mean_update_time']:.2f} ms")
    print(f"  Max update time: {stats['max_update_time']:.2f} ms")
    print(f"  Update rate: {stats['update_rate']:.1f} Hz")
    
    # Convert to arrays for plotting
    interceptor_positions = np.array(interceptor_positions)
    target_positions = np.array(target_positions)
    
    return times, interceptor_positions, target_positions


def visualize_scenario(times, interceptor_positions, target_positions, battlespace):
    """Visualize the scenario results"""
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # 2D trajectory plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], 
             'b-', label='Interceptor', linewidth=2)
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 
             'r-', label='Target', linewidth=2)
    ax1.plot(interceptor_positions[0, 0], interceptor_positions[0, 1], 
             'bo', markersize=10, label='Start')
    ax1.plot(interceptor_positions[-1, 0], interceptor_positions[-1, 1], 
             'b*', markersize=15)
    ax1.plot(target_positions[-1, 0], target_positions[-1, 1], 
             'r*', markersize=15)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('2D Trajectories')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Altitude profile
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, interceptor_positions[:, 2], 'b-', label='Interceptor')
    ax2.plot(times, target_positions[:, 2], 'r-', label='Target')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Profiles')
    ax2.legend()
    ax2.grid(True)
    
    # Range vs time
    ax3 = plt.subplot(2, 3, 3)
    ranges = [np.linalg.norm(interceptor_positions[i] - target_positions[i]) 
              for i in range(len(times))]
    ax3.plot(times, ranges, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Range (m)')
    ax3.set_title('Range to Target')
    ax3.grid(True)
    ax3.axhline(y=100, color='r', linestyle='--', label='Intercept Range')
    ax3.legend()
    
    # 3D trajectory
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], 
             interceptor_positions[:, 2], 'b-', label='Interceptor', linewidth=2)
    ax4.plot(target_positions[:, 0], target_positions[:, 1], 
             target_positions[:, 2], 'r-', label='Target', linewidth=2)
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    ax4.set_zlabel('Altitude (m)')
    ax4.set_title('3D Trajectories')
    ax4.legend()
    
    # Speed profiles
    ax5 = plt.subplot(2, 3, 5)
    interceptor_speeds = [np.linalg.norm(interceptor_positions[i+1] - interceptor_positions[i]) / (times[i+1] - times[i])
                         if i < len(times)-1 else 0 for i in range(len(times))]
    target_speeds = [np.linalg.norm(target_positions[i+1] - target_positions[i]) / (times[i+1] - times[i])
                    if i < len(times)-1 else 0 for i in range(len(times))]
    ax5.plot(times, interceptor_speeds, 'b-', label='Interceptor')
    ax5.plot(times, target_speeds, 'r-', label='Target')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed Profiles')
    ax5.legend()
    ax5.grid(True)
    
    # Terrain with trajectories overlay
    ax6 = plt.subplot(2, 3, 6)
    
    # Get terrain for background
    terrain_step = 10
    terrain = battlespace.terrain.elevation[::terrain_step, ::terrain_step]
    extent = [0, battlespace.width, 0, battlespace.height]
    
    im = ax6.contourf(terrain, levels=20, cmap='terrain', alpha=0.3, extent=extent)
    ax6.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], 
             'b-', linewidth=2, label='Interceptor')
    ax6.plot(target_positions[:, 0], target_positions[:, 1], 
             'r-', linewidth=2, label='Target')
    ax6.set_xlabel('East (m)')
    ax6.set_ylabel('North (m)')
    ax6.set_title('Trajectories over Terrain')
    ax6.legend()
    ax6.axis('equal')
    
    plt.suptitle('Asset Manager Integration Test - Single Target Scenario', fontsize=14)
    plt.tight_layout()
    plt.show()


def test_multi_aircraft_performance():
    """Test performance with multiple aircraft"""
    print("\n" + "="*60)
    print("Multi-Aircraft Performance Test")
    print("="*60)
    
    # Create battlespace
    print("\nInitializing Battlespace...")
    battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
    battlespace.generate(seed=42)
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.02)
    
    # Test different aircraft counts
    aircraft_counts = [10, 25, 50]
    
    for n_aircraft in aircraft_counts:
        print(f"\nTesting with {n_aircraft} aircraft...")
        
        # Clear previous assets
        asset_manager.assets.clear()
        asset_manager.asset_count = 0
        asset_manager._init_spatial_index()
        
        # Spawn aircraft in a grid
        grid_size = int(np.ceil(np.sqrt(n_aircraft)))
        spacing = 2000  # meters
        
        for i in range(n_aircraft):
            row = i // grid_size
            col = i % grid_size
            
            position = np.array([
                5000 + col * spacing,
                5000 + row * spacing,
                1000 + (i % 5) * 200  # Vary altitude
            ], dtype=np.float64)
            
            config = {
                'aircraft': 'configs/aircraft/target_basic.yaml',
                'initial_state': {
                    'position': position.tolist(),
                    'velocity': 40.0 + (i % 3) * 10,  # Vary speed
                    'heading': (i * 30) % 360,  # Vary heading
                    'throttle': 0.6
                },
                'behavior': 'orbit'
            }
            
            asset_manager.spawn_aircraft(
                config=config,
                asset_id=f"aircraft_{i}",
                asset_type=AssetType.TARGET
            )
        
        # Run performance test
        test_duration = 10.0  # seconds
        start_time = time.perf_counter()
        update_count = 0
        
        while asset_manager.time < test_duration:
            asset_manager.update()
            update_count += 1
            
        elapsed = time.perf_counter() - start_time
        
        # Report results
        stats = asset_manager.get_performance_stats()
        print(f"  Aircraft: {n_aircraft}")
        print(f"  Updates: {update_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Sim time: {asset_manager.time:.2f}s")
        print(f"  Real-time factor: {asset_manager.time/elapsed:.2f}x")
        print(f"  Mean update: {stats['mean_update_time']:.2f}ms")
        print(f"  Max update: {stats['max_update_time']:.2f}ms")
        print(f"  Update rate: {stats['update_rate']:.1f}Hz")
        
        # Reset for next test
        asset_manager.time = 0.0


def main():
    """Main test execution"""
    print("="*60)
    print("Asset Manager Integration Tests")
    print("="*60)
    
    # Check for required config files
    required_files = [
        "configs/battlespace/default_battlespace.yaml",
        "configs/aircraft/interceptor_drone.yaml",
        "configs/aircraft/target_basic.yaml",
        "configs/scenarios/single_target.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\nWarning: Missing configuration files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all configuration files are created.")
        print("Continuing with available tests...")
    
    # Run tests
    try:
        # Test 1: Single target scenario
        times, interceptor_pos, target_pos = run_single_target_scenario()
        
        # Create battlespace for visualization
        battlespace = Battlespace(config_file="configs/battlespace/default_battlespace.yaml")
        battlespace.generate(seed=42)
        
        # Visualize results
        visualize_scenario(times, interceptor_pos, target_pos, battlespace)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Skipping single target scenario test.")
    
    # Test 2: Multi-aircraft performance
    try:
        test_multi_aircraft_performance()
    except Exception as e:
        print(f"\nError in performance test: {e}")
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()