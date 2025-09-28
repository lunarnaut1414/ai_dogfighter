#!/usr/bin/env python3
"""
Simple 1v1 Interceptor Demonstration
Showcases a single interceptor pursuing a single target using the integrated guidance system.

Location: examples/simple_intercept_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Import battlespace and assets directly
from src.battlespace.core import Battlespace
from src.assets.asset_manager import AssetManager, AssetType
from src.assets.aircraft_3dof import Aircraft3DOF

# Import guidance components
from src.guidance_core.state_machine import GuidanceStateMachine, MissionPhase
from src.guidance_core.guidance_laws import ProportionalNavigation, AugmentedProportionalNavigation
from src.guidance_core.target_manager import TargetManager
from src.guidance_core.safety_monitor import SafetyMonitor


class SimpleInterceptorGuidance:
    """Simplified guidance system for 1v1 demonstration"""
    
    def __init__(self):
        # Initialize guidance components
        self.state_machine = GuidanceStateMachine(initial_state=MissionPhase.STARTUP)
        self.pn_guidance = ProportionalNavigation({'N': 3.0})
        self.apn_guidance = AugmentedProportionalNavigation({'N': 4.0})
        self.target_manager = TargetManager({'max_tracks': 1})  # Use correct parameter name
        self.safety_monitor = SafetyMonitor()
        
        # Mission parameters
        self.intercept_range = 25  # meters - tighter intercept requirement
        self.current_phase = MissionPhase.STARTUP
        self.mission_time = 0
        self.mission_complete = False  # Track completion separately
        
    def update(self, interceptor_state, target_state, dt):
        """Update guidance system and compute commands"""
        
        self.mission_time += dt
        
        # Update mission phase based on conditions
        if self.current_phase == MissionPhase.STARTUP and self.mission_time > 2.0:
            self.current_phase = MissionPhase.SEARCH
            print(f"[T={self.mission_time:.1f}s] Transitioning to SEARCH phase")
            
        # Check if target detected (always true in this simple demo)
        if self.current_phase == MissionPhase.SEARCH and target_state is not None:
            self.current_phase = MissionPhase.TRACK
            print(f"[T={self.mission_time:.1f}s] Target detected! Transitioning to TRACK phase")
            
        # Calculate range to target
        if target_state is not None:
            range_to_target = np.linalg.norm(
                np.array(target_state.position) - np.array(interceptor_state.position)
            )
            
            # Check for intercept phase transition
            if self.current_phase == MissionPhase.TRACK and range_to_target < 2000:
                self.current_phase = MissionPhase.INTERCEPT
                print(f"[T={self.mission_time:.1f}s] Entering INTERCEPT phase - Range: {range_to_target:.0f}m")
            
            # Check for successful intercept - tighter threshold
            if range_to_target < 25:  # 25 meter intercept range
                if not self.mission_complete:
                    self.mission_complete = True
                    print(f"[T={self.mission_time:.1f}s] INTERCEPT SUCCESSFUL! Final range: {range_to_target:.1f}m")
                return None
        
        # Compute guidance commands based on phase
        if self.current_phase in [MissionPhase.TRACK, MissionPhase.INTERCEPT]:
            # Convert states to dictionaries for guidance laws
            own_dict = {
                'position': interceptor_state.position,
                'velocity': interceptor_state.velocity * np.array([
                    np.cos(interceptor_state.heading) * np.cos(interceptor_state.flight_path_angle),
                    np.sin(interceptor_state.heading) * np.cos(interceptor_state.flight_path_angle),
                    -np.sin(interceptor_state.flight_path_angle)
                ])
            }
            
            target_dict = {
                'position': target_state.position,
                'velocity': target_state.velocity * np.array([
                    np.cos(target_state.heading) * np.cos(target_state.flight_path_angle),
                    np.sin(target_state.heading) * np.cos(target_state.flight_path_angle),
                    -np.sin(target_state.flight_path_angle)
                ])
            }
            
            # Use appropriate guidance law
            if self.current_phase == MissionPhase.TRACK:
                guidance_cmd = self.pn_guidance.compute(own_dict, target_dict, dt)
            else:
                guidance_cmd = self.apn_guidance.compute(own_dict, target_dict, dt)
            
            # Compute heading to intercept point (lead pursuit)
            rel_pos = target_dict['position'] - own_dict['position']
            rel_vel = target_dict['velocity'] - own_dict['velocity']
            
            # Simple lead calculation
            time_to_intercept = np.linalg.norm(rel_pos) / max(np.linalg.norm(own_dict['velocity']), 1.0)
            lead_point = target_dict['position'] + target_dict['velocity'] * min(time_to_intercept * 0.5, 5.0)
            
            # Heading to lead point
            lead_vector = lead_point - own_dict['position']
            desired_heading = np.arctan2(lead_vector[1], lead_vector[0])
            
            return {
                'commanded_heading': desired_heading,
                'commanded_altitude': target_state.position[2],
                'commanded_throttle': 0.8 if self.current_phase == MissionPhase.TRACK else 1.0
            }
        else:
            # Default search pattern
            return {
                'commanded_heading': interceptor_state.heading,
                'commanded_altitude': 2000,
                'commanded_throttle': 0.6
            }


def create_default_aircraft_config():
    """Create a default aircraft configuration"""
    return {
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


def run_simple_intercept():
    """Run a simple 1v1 intercept scenario"""
    
    print("="*70)
    print("SIMPLE INTERCEPTOR DEMONSTRATION")
    print("1 Interceptor vs 1 Target Scenario")
    print("="*70)
    
    # Create battlespace directly
    battlespace = Battlespace()
    battlespace.generate()
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.1)
    
    # Create interceptor aircraft configuration
    interceptor_config = create_default_aircraft_config()
    interceptor_config['performance']['v_max'] = 100.0  # Faster interceptor
    interceptor_config['fuel']['capacity'] = 1000.0  # More fuel for long chase
    interceptor_config['fuel']['initial'] = 1000.0
    
    # Create target aircraft configuration  
    target_config = create_default_aircraft_config()
    target_config['performance']['v_max'] = 60.0  # Slower target
    
    # Add interceptor
    interceptor = Aircraft3DOF(config_dict=interceptor_config)
    interceptor.initialize_state(
        position=np.array([1000, 1000, 2000], dtype=float),
        velocity=50.0,
        heading=0.0,
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Add target
    target = Aircraft3DOF(config_dict=target_config)
    target.initialize_state(
        position=np.array([10000, 8000, 2500], dtype=float),
        velocity=40.0,
        heading=np.arctan2(-20, 40),
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Create guidance system
    guidance = SimpleInterceptorGuidance()
    
    # Simulation parameters
    dt = 0.1  # 10 Hz update rate
    max_time = 300*5  # Extend to 5 minutes to ensure intercept
    
    # Storage for trajectory
    interceptor_trajectory = []
    target_trajectory = []
    
    # Run simulation
    print("\n" + "-"*70)
    print("Starting simulation...")
    print("-"*70)
    
    sim_time = 0
    last_print_time = 0
    intercept_achieved = False
    
    while sim_time < max_time:
        # Store positions for plotting
        interceptor_trajectory.append(interceptor.state.position.copy())
        target_trajectory.append(target.state.position.copy())
        
        # Calculate current range
        current_range = np.linalg.norm(
            target.state.position - interceptor.state.position
        )
        
        # Check for intercept
        if current_range < 25:  # 25 meter intercept range
            intercept_achieved = True
            print(f"\n[T={sim_time:.1f}s] INTERCEPT ACHIEVED! Final range: {current_range:.1f}m")
            break
            
        # Check for mission failures
        if interceptor.state.velocity < 20:  # Stall condition
            print(f"\n[T={sim_time:.1f}s] INTERCEPTOR STALLED! Speed: {interceptor.state.velocity:.1f} m/s")
            break
            
        if interceptor.state.fuel_remaining <= 0:
            print(f"\n[T={sim_time:.1f}s] INTERCEPTOR OUT OF FUEL!")
            break
        # Store positions for plotting
        interceptor_trajectory.append(interceptor.state.position.copy())
        target_trajectory.append(target.state.position.copy())
        
        # Update guidance
        guidance_commands = guidance.update(
            interceptor.state,
            target.state,
            dt
        )
        
        # Check if intercept complete (guidance returns None when complete)
        if guidance_commands is None:
            intercept_achieved = True
            break
            
        # Apply guidance commands to interceptor
        if guidance_commands:
            # Convert commanded heading to bank angle
            heading_error = guidance_commands['commanded_heading'] - interceptor.state.heading
            # Wrap angle to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            # More aggressive bank angle for better turning
            desired_bank = np.clip(heading_error * 3.0, -np.radians(60), np.radians(60))
            
            # Also update altitude control
            alt_error = guidance_commands['commanded_altitude'] - interceptor.state.position[2]
            pitch_adjust = np.clip(alt_error * 0.001, -0.1, 0.1)
            
            interceptor.set_controls(
                bank_angle=desired_bank,
                throttle=guidance_commands['commanded_throttle']
            )
            
            # Apply some pitch control through velocity vector adjustment if needed
            if abs(alt_error) > 100:
                interceptor.state.flight_path_angle = pitch_adjust
        
        # Simple evasive maneuver for target (optional - mild evasion)
        if sim_time > 40 and sim_time < 45:
            # Target performs a gentle turn
            target.set_controls(
                bank_angle=np.radians(20),
                throttle=0.6
            )
        elif sim_time > 60 and sim_time < 65:
            # Another evasive turn
            target.set_controls(
                bank_angle=np.radians(-25),
                throttle=0.7
            )
        else:
            target.set_controls(
                bank_angle=0,
                throttle=0.5
            )
        
        # Update aircraft dynamics
        interceptor.update(dt)
        target.update(dt)
        
        sim_time += dt
        
        # Print status every 5 seconds
        if sim_time - last_print_time >= 5.0:
            range_to_target = np.linalg.norm(
                target.state.position - interceptor.state.position
            )
            print(f"T={sim_time:6.1f}s | Phase: {guidance.current_phase.name:10s} | "
                  f"Range: {range_to_target:7.1f}m | "
                  f"Int Speed: {interceptor.state.velocity:6.1f} m/s")
            last_print_time = sim_time
    
    # Convert trajectories to arrays
    interceptor_trajectory = np.array(interceptor_trajectory)
    target_trajectory = np.array(target_trajectory)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Final simulation time: {sim_time:.1f} seconds")
    print(f"Final mission phase: {guidance.current_phase.name}")
    print(f"Mission complete: {guidance.mission_complete}")
    
    if guidance.mission_complete or intercept_achieved:
        final_range = np.linalg.norm(target_trajectory[-1] - interceptor_trajectory[-1])
        print(f"Result: SUCCESSFUL INTERCEPT! ✅")
        print(f"Final range: {final_range:.1f} meters")
    else:
        print("Result: Target escaped or timeout")
    
    # Plot results
    plot_intercept_results(interceptor_trajectory, target_trajectory, guidance)
    
    return interceptor_trajectory, target_trajectory, guidance


def plot_intercept_results(interceptor_traj, target_traj, guidance):
    """Plot the intercept scenario results"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], interceptor_traj[:, 2], 
             'b-', linewidth=2, label='Interceptor')
    ax1.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], 
             'r-', linewidth=2, label='Target')
    
    # Mark start and end points
    ax1.scatter(*interceptor_traj[0], color='blue', s=100, marker='o', label='Int. Start')
    ax1.scatter(*interceptor_traj[-1], color='blue', s=100, marker='X')
    ax1.scatter(*target_traj[0], color='red', s=100, marker='o', label='Tgt. Start')
    ax1.scatter(*target_traj[-1], color='red', s=100, marker='X')
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Engagement Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Top-down view
    ax2 = fig.add_subplot(222)
    ax2.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], 'b-', linewidth=2, label='Interceptor')
    ax2.plot(target_traj[:, 0], target_traj[:, 1], 'r-', linewidth=2, label='Target')
    
    # Add markers every 10 seconds
    time_markers = np.arange(0, len(interceptor_traj), 100)  # Every 10 seconds at 10Hz
    ax2.scatter(interceptor_traj[time_markers, 0], interceptor_traj[time_markers, 1], 
                color='blue', s=20, alpha=0.5)
    ax2.scatter(target_traj[time_markers, 0], target_traj[time_markers, 1], 
                color='red', s=20, alpha=0.5)
    
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_title('Top-Down View')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Range vs Time
    ax3 = fig.add_subplot(223)
    time_array = np.arange(len(interceptor_traj)) * 0.1  # 10 Hz
    ranges = np.linalg.norm(target_traj - interceptor_traj, axis=1)
    ax3.plot(time_array, ranges, 'g-', linewidth=2)
    ax3.axhline(y=50, color='r', linestyle='--', label='Intercept Range')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Range (m)')
    ax3.set_title('Range to Target vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # Altitude profiles
    ax4 = fig.add_subplot(224)
    ax4.plot(time_array, interceptor_traj[:, 2], 'b-', linewidth=2, label='Interceptor')
    ax4.plot(time_array, target_traj[:, 2], 'r-', linewidth=2, label='Target')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Altitude (m)')
    ax4.set_title('Altitude Profiles')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(f'1v1 Intercept Scenario - Final Phase: {guidance.current_phase.name}', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Run the simple intercept demonstration
    run_simple_intercept()