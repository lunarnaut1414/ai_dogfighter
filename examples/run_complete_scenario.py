#!/usr/bin/env python3
"""
Complete example of running a scenario with the integrated simulation system.
Demonstrates full pipeline from setup through execution to results analysis.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.scenario_runner import ScenarioRunner
from src.simulation.scenario_visualizer import ScenarioVisualizer
from src.guidance_core.guidance_laws import ProportionalNavigation


class BasicInterceptGuidance:
    """
    Simple guidance algorithm for testing scenario execution.
    Uses proportional navigation for intercept.
    """
    
    def __init__(self):
        self.pn_guidance = ProportionalNavigation(navigation_gain=3.0)
        self.current_target = None
        
    def compute_commands(self, 
                        interceptor_state: dict,
                        target_states: dict,
                        battlespace=None) -> dict:
        """
        Compute guidance commands for the interceptor.
        
        Args:
            interceptor_state: Current interceptor state
            target_states: Dictionary of detected target states
            battlespace: Battlespace instance for environmental queries
            
        Returns:
            Command dictionary with heading, altitude, and throttle commands
        """
        if not target_states:
            # No targets - maintain current state
            return {
                'commanded_heading': interceptor_state.get('heading', 0),
                'commanded_altitude': interceptor_state['position'][2],
                'commanded_throttle': 0.7
            }
            
        # Select closest target
        interceptor_pos = np.array(interceptor_state['position'])
        min_range = float('inf')
        selected_target = None
        selected_target_id = None
        
        for target_id, target_state in target_states.items():
            target_pos = np.array(target_state['position'])
            range_to_target = np.linalg.norm(target_pos - interceptor_pos)
            
            if range_to_target < min_range:
                min_range = range_to_target
                selected_target = target_state
                selected_target_id = target_id
                
        if not selected_target:
            return {
                'commanded_heading': interceptor_state.get('heading', 0),
                'commanded_altitude': interceptor_state['position'][2],
                'commanded_throttle': 0.7
            }
            
        # Compute intercept trajectory using proportional navigation
        target_pos = np.array(selected_target['position'])
        target_vel = np.array(selected_target.get('velocity', [0, 0, 0]))
        interceptor_vel = np.array(interceptor_state.get('velocity_vector', [0, 0, 0]))
        
        # Calculate line of sight angles
        los_vector = target_pos - interceptor_pos
        range_to_target = np.linalg.norm(los_vector)
        
        # Heading command (2D intercept)
        heading_to_target = np.arctan2(los_vector[1], los_vector[0])
        
        # Simple proportional navigation
        if range_to_target > 100:  # Beyond close range
            # Calculate closing velocity
            closing_velocity = -np.dot(los_vector / range_to_target, 
                                      interceptor_vel - target_vel)
            
            # Time to go estimate
            if closing_velocity > 1.0:
                time_to_go = range_to_target / closing_velocity
            else:
                time_to_go = 100.0  # Large value if not closing
                
            # Lead angle calculation
            lead_angle = 0.0
            if time_to_go < 20.0:  # Within reasonable intercept time
                # Estimate future target position
                future_target_pos = target_pos + target_vel * min(time_to_go, 5.0)
                future_los = future_target_pos - interceptor_pos
                heading_to_future = np.arctan2(future_los[1], future_los[0])
                lead_angle = heading_to_future - heading_to_target
                
            commanded_heading = heading_to_target + lead_angle * 0.5
        else:
            # Direct pursuit at close range
            commanded_heading = heading_to_target
            
        # Altitude command
        target_altitude = target_pos[2]
        altitude_error = target_altitude - interceptor_pos[2]
        
        # Proportional altitude control with rate limiting
        altitude_rate = np.clip(altitude_error / 5.0, -10.0, 10.0)  # Max 10 m/s climb/descent
        commanded_altitude = interceptor_pos[2] + altitude_rate * 0.1
        
        # Throttle command based on range
        if range_to_target > 5000:
            commanded_throttle = 1.0  # Full throttle for distant targets
        elif range_to_target > 1000:
            commanded_throttle = 0.8
        elif range_to_target > 500:
            commanded_throttle = 0.7
        else:
            commanded_throttle = 0.6  # Reduce speed for final approach
            
        # Ensure altitude stays within limits
        commanded_altitude = np.clip(commanded_altitude, 100, 10000)
        
        return {
            'commanded_heading': commanded_heading,
            'commanded_altitude': commanded_altitude,
            'commanded_throttle': commanded_throttle,
            'target_id': selected_target_id,
            'range': range_to_target
        }


def run_single_target_scenario():
    """Run a single target interception scenario."""
    
    print("="*70)
    print("SINGLE TARGET INTERCEPTION SCENARIO")
    print("="*70)
    
    # Create guidance algorithm
    guidance = BasicInterceptGuidance()
    
    # Initialize scenario runner
    scenario_file = 'configs/scenarios/single_target.yaml'
    
    # Check if file exists
    if not Path(scenario_file).exists():
        print(f"ERROR: Scenario file not found: {scenario_file}")
        print("Please ensure the configuration file exists.")
        return None
        
    runner = ScenarioRunner(
        scenario_config=scenario_file,
        guidance_algorithm=guidance,
        realtime=False,  # Run as fast as possible
        verbose=True
    )
    
    # Setup scenario
    try:
        runner.setup()
    except Exception as e:
        print(f"ERROR during setup: {e}")
        return None
        
    # Run scenario
    print("\nStarting scenario execution...")
    print("-"*70)
    
    try:
        results = runner.run()
    except Exception as e:
        print(f"ERROR during execution: {e}")
        return None
        
    # Display results
    print("\n" + "="*70)
    print("SCENARIO RESULTS")
    print("="*70)
    
    print(f"\nFinal Status: {results['state']}")
    print(f"Duration: {results['duration']:.1f} seconds")
    print(f"Total Steps: {results['metrics']['total_steps']}")
    
    print("\nObjectives:")
    obj_summary = results['objectives']
    print(f"  Completed: {obj_summary['completed']}/{obj_summary['total']}")
    print(f"  Success Rate: {obj_summary['completion_rate']:.1%}")
    
    print("\nIntercepts:")
    print(f"  Total: {results['metrics']['intercepts']}")
    if results['events']:
        for event in results['events']:
            print(f"  - Target {event['target_id']} at {event['range']:.1f}m "
                  f"(T={event['time']:.1f}s)")
            
    print("\nPerformance:")
    print(f"  Mean Update Time: {results['metrics']['mean_update_time_ms']:.2f}ms")
    print(f"  Max Update Time: {results['metrics']['max_update_time_ms']:.2f}ms")
    print(f"  Effective Rate: {1000/results['metrics']['mean_update_time_ms']:.1f}Hz")
    
    return results


def run_multi_target_scenario():
    """Run a multi-target engagement scenario."""
    
    print("\n" + "="*70)
    print("MULTI-TARGET ENGAGEMENT SCENARIO")
    print("="*70)
    
    # Create guidance algorithm
    guidance = BasicInterceptGuidance()
    
    # Initialize scenario runner
    scenario_file = 'configs/scenarios/multi_target.yaml'
    
    if not Path(scenario_file).exists():
        print(f"ERROR: Scenario file not found: {scenario_file}")
        return None
        
    runner = ScenarioRunner(
        scenario_config=scenario_file,
        guidance_algorithm=guidance,
        realtime=False,
        verbose=True
    )
    
    # Setup and run
    try:
        runner.setup()
        results = runner.run()
    except Exception as e:
        print(f"ERROR: {e}")
        return None
        
    return results


def visualize_scenario_results(recording_file: str):
    """Visualize recorded scenario data."""
    
    print("\n" + "="*70)
    print("SCENARIO VISUALIZATION")
    print("="*70)
    
    if not Path(recording_file).exists():
        print(f"Recording file not found: {recording_file}")
        return
        
    # Create visualizer
    visualizer = ScenarioVisualizer(recording_file=recording_file)
    
    # Create visualization
    print("Creating visualization...")
    visualizer.create_replay_display()
    
    # Show visualization
    print("Displaying results (close window to continue)...")
    visualizer.show()


def run_performance_test():
    """Test simulation performance with multiple aircraft."""
    
    print("\n" + "="*70)
    print("PERFORMANCE TEST")
    print("="*70)
    
    from src.simulation.environment import SimulationEnvironment
    
    # Create environment
    env = SimulationEnvironment(
        battlespace_config='configs/battlespace/default_battlespace.yaml',
        dt=0.02,
        enable_physics=True,
        enable_sensors=True,
        enable_weather=True
    )
    
    # Spawn multiple aircraft
    n_targets = 10
    print(f"\nSpawning {n_targets} target aircraft...")
    
    for i in range(n_targets):
        # Random position in battlespace
        x = np.random.uniform(5000, 45000)
        y = np.random.uniform(5000, 45000)
        z = np.random.uniform(1000, 5000)
        heading = np.random.uniform(0, 360)
        
        env.spawn_aircraft(
            aircraft_type='target',
            position=[x, y, z],
            heading=heading,
            velocity=40 + np.random.uniform(-10, 10),
            team='red'
        )
        
    # Spawn interceptor
    interceptor_id = env.spawn_aircraft(
        aircraft_type='interceptor',
        position=[25000, 25000, 3000],
        heading=0,
        velocity=50,
        team='blue'
    )
    
    print(f"Total aircraft: {len(env.asset_manager.assets)}")
    
    # Run performance test
    print("\nRunning 10-second performance test...")
    test_duration = 10.0
    start_time = time.time()
    
    while env.time < test_duration:
        step_result = env.step()
        
        # Print periodic updates
        if int(env.time) > int(env.time - env.dt) and int(env.time) % 2 == 0:
            print(f"  T={env.time:.0f}s: "
                  f"Step time={step_result['step_time_ms']:.2f}ms, "
                  f"RT factor={step_result['realtime_factor']:.1f}x")
            
    elapsed = time.time() - start_time
    
    # Display results
    metrics = env.get_metrics()
    print("\nPerformance Results:")
    print(f"  Simulation time: {env.time:.1f}s")
    print(f"  Real time: {elapsed:.1f}s")
    print(f"  Speed factor: {env.time/elapsed:.1f}x realtime")
    print(f"  Total steps: {metrics['steps']}")
    print(f"  Mean step time: {metrics['mean_step_time_ms']:.2f}ms")
    print(f"  Max step time: {metrics['max_step_time_ms']:.2f}ms")
    print(f"  Achieved rate: {metrics['steps']/elapsed:.1f}Hz")


def main():
    """Main entry point for scenario execution examples."""
    
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description='Run interceptor guidance scenarios'
    )
    parser.add_argument(
        '--scenario', 
        choices=['single', 'multi', 'performance', 'all'],
        default='single',
        help='Which scenario to run'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize results after execution'
    )
    parser.add_argument(
        '--record',
        action='store_true', 
        help='Enable scenario recording'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INTERCEPTOR GUIDANCE SIMULATION SYSTEM")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Visualize: {args.visualize}")
    print(f"Record: {args.record}")
    
    # Create data directory if needed
    Path('data/scenarios').mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Run requested scenarios
    if args.scenario in ['single', 'all']:
        result = run_single_target_scenario()
        if result:
            results.append(('Single Target', result))
            
    if args.scenario in ['multi', 'all']:
        result = run_multi_target_scenario()
        if result:
            results.append(('Multi Target', result))
            
    if args.scenario in ['performance', 'all']:
        run_performance_test()
        
    # Visualize if requested
    if args.visualize and args.record:
        # Look for most recent recording
        import glob
        recordings = glob.glob('data/scenarios/*.json')
        if recordings:
            latest = max(recordings, key=os.path.getctime)
            visualize_scenario_results(latest)
            
    # Summary
    if results:
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        
        for scenario_name, result in results:
            print(f"\n{scenario_name}:")
            print(f"  Status: {result['state']}")
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Intercepts: {result['metrics']['intercepts']}")
            print(f"  Success Rate: {result['objectives']['completion_rate']:.1%}")
            
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()