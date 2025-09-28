#!/usr/bin/env python3
"""
Simple 1v1 Interceptor Demonstration - FIXED VERSION
Properly integrates with guidance laws to achieve successful intercepts.

Location: examples/simple_intercept_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# Import battlespace and assets directly
from src.battlespace.core import Battlespace
from src.assets.asset_manager import AssetManager
from src.assets.aircraft_3dof import Aircraft3DOF

# Import guidance components
from src.guidance_core.state_machine import GuidanceStateMachine, MissionPhase
from src.guidance_core.guidance_laws import ProportionalNavigation, AugmentedProportionalNavigation
from src.guidance_core.target_manager import TargetManager
from src.guidance_core.safety_monitor import SafetyMonitor


class SimpleInterceptorGuidance:
    """Fixed guidance system that properly uses PN guidance commands"""
    
    def __init__(self):
        # Initialize guidance components
        self.state_machine = GuidanceStateMachine(initial_state=MissionPhase.STARTUP)
        self.pn_guidance = ProportionalNavigation({'N': 6.0})  # Was 4.0
        self.apn_guidance = AugmentedProportionalNavigation({'N': 6.5})  # Was 4.5
        
        self.target_manager = TargetManager({'max_tracks': 1})
        self.safety_monitor = SafetyMonitor()
        
        # Mission parameters
        self.intercept_range = 25  # meters
        self.current_phase = MissionPhase.STARTUP
        self.mission_time = 0
        self.mission_complete = False
        
    def update(self, interceptor_state, target_state, dt):
        """Update guidance system and compute commands"""
        
        self.mission_time += dt
        
        # Update mission phase
        if self.current_phase == MissionPhase.STARTUP and self.mission_time > 2.0:
            self.current_phase = MissionPhase.SEARCH
            print(f"[T={self.mission_time:.1f}s] Transitioning to SEARCH phase")
            
        # Check if target detected
        if self.current_phase == MissionPhase.SEARCH and target_state is not None:
            self.current_phase = MissionPhase.TRACK
            print(f"[T={self.mission_time:.1f}s] Target detected! Transitioning to TRACK phase")
            
        # Calculate range to target
        if target_state is not None:
            range_to_target = np.linalg.norm(
                target_state.position - interceptor_state.position
            )
            
            # Phase transitions
            if self.current_phase == MissionPhase.TRACK and range_to_target < 2000:
                self.current_phase = MissionPhase.INTERCEPT
                print(f"[T={self.mission_time:.1f}s] Entering INTERCEPT phase - Range: {range_to_target:.0f}m")
            
            # Check for successful intercept
            if range_to_target < self.intercept_range:
                if not self.mission_complete:
                    self.mission_complete = True
                    print(f"[T={self.mission_time:.1f}s] INTERCEPT SUCCESSFUL! Final range: {range_to_target:.1f}m")
                return None
        
        # Compute guidance commands
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
            if self.current_phase == MissionPhase.TRACK or range_to_target > 500:
                guidance_cmd = self.pn_guidance.compute(own_dict, target_dict, dt)
            else:
                # Use APN for terminal phase
                guidance_cmd = self.apn_guidance.compute(own_dict, target_dict, dt)
            
            # CRITICAL: Actually use the guidance command outputs!
            # Convert acceleration command to heading change
            accel_cmd = guidance_cmd.acceleration_command
            own_vel = own_dict['velocity']
            own_speed = np.linalg.norm(own_vel)
            
            if own_speed > 10:
                # Get current heading
                current_heading = interceptor_state.heading
                
                # Use the heading rate from guidance
                desired_heading = current_heading + guidance_cmd.heading_rate * dt
                
                # Add some lead pursuit bias at close range
                if range_to_target < 1000:
                    rel_pos = target_dict['position'] - own_dict['position']
                    rel_vel = target_dict['velocity'] - own_dict['velocity']
                    closing_speed = -np.dot(rel_vel, rel_pos) / range_to_target
                    
                    if closing_speed > 0:
                        t_go = min(range_to_target / closing_speed, 5.0)
                        lead_point = target_dict['position'] + target_dict['velocity'] * t_go * 0.3
                        lead_vector = lead_point - own_dict['position']
                        lead_heading = np.arctan2(lead_vector[1], lead_vector[0])
                        
                        # Blend with PN heading
                        blend_factor = min(0.3, 50.0 / range_to_target)
                        desired_heading = (1 - blend_factor) * desired_heading + blend_factor * lead_heading
            else:
                # If too slow, just point at target
                rel_pos = target_dict['position'] - own_dict['position']
                desired_heading = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Use guidance throttle command - THIS IS CRITICAL!
            throttle = guidance_cmd.throttle_command
            
            # Target altitude based on target position with some prediction
            target_alt = target_state.position[2]
            if guidance_cmd.time_to_go and guidance_cmd.time_to_go < 10:
                # Predict target altitude
                target_vert_vel = target_dict['velocity'][2]
                target_alt += target_vert_vel * min(guidance_cmd.time_to_go * 0.3, 3.0)
            
            return {
                'commanded_heading': desired_heading,
                'commanded_altitude': target_alt,
                'commanded_throttle': throttle,  # USE THE GUIDANCE THROTTLE!
                'time_to_go': guidance_cmd.time_to_go,
                'predicted_miss': guidance_cmd.predicted_miss_distance
            }
        else:
            # Search pattern
            return {
                'commanded_heading': interceptor_state.heading,
                'commanded_altitude': 2000,
                'commanded_throttle': 0.6,
                'time_to_go': None,
                'predicted_miss': None
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
            'capacity': 200.0,
            'initial': 200.0
        }
    }


def run_simple_intercept():
    """Run a simple 1v1 intercept scenario"""
    
    print("="*70)
    print("SIMPLE INTERCEPTOR DEMONSTRATION - FIXED VERSION")
    print("1 Interceptor vs 1 Target Scenario")
    print("="*70)
    
    # Create battlespace
    battlespace = Battlespace()
    battlespace.generate()
    
    # Create asset manager
    asset_manager = AssetManager(battlespace, dt=0.1)
    
    # Create interceptor configuration
    interceptor_config = create_default_aircraft_config()
    interceptor_config['performance']['v_max'] = 150.0  # Was 100
    interceptor_config['performance']['v_cruise'] = 120.0  # Add cruise speed
    interceptor_config['fuel']['capacity'] = 1000.0
    interceptor_config['fuel']['initial'] = 1000.0
    
    # Create target configuration  
    target_config = create_default_aircraft_config()
    target_config['performance']['v_max'] = 60.0  # Keep as is
    target_config['performance']['v_cruise'] = 50.0  # Add cruise speed
    
    # Initialize interceptor
    interceptor = Aircraft3DOF(config_dict=interceptor_config)
    interceptor.initialize_state(
        position=np.array([1000, 1000, 2000], dtype=float),
        velocity=50.0,
        heading=0.0,
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Initialize target
    target = Aircraft3DOF(config_dict=target_config)
    target.initialize_state(
        position=np.array([10000, 8000, 2500], dtype=float),
        velocity=40.0,
        heading=np.arctan2(-20, 40),  # Moving away initially
        flight_path_angle=0.0,
        throttle=0.5
    )
    
    # Create guidance system
    guidance = SimpleInterceptorGuidance()
    
    # Simulation parameters
    dt = 0.1  # 10 Hz update rate
    max_time = 6000  # 60 minutes max
    
    # Storage for trajectory
    interceptor_trajectory = []
    target_trajectory = []
    performance_data = {
        'time': [],
        'range': [],
        'closing_speed': [],
        'int_speed': [],
        'throttle': [],
        'predicted_miss': []
    }
    
    # Run simulation
    print("\n" + "-"*70)
    print("Starting simulation...")
    print("-"*70)
    
    sim_time = 0
    last_print_time = 0
    intercept_achieved = False
    
    while sim_time < max_time:
        # Store positions
        interceptor_trajectory.append(interceptor.state.position.copy())
        target_trajectory.append(target.state.position.copy())
        
        # Calculate metrics
        rel_pos = target.state.position - interceptor.state.position
        current_range = np.linalg.norm(rel_pos)
        
        int_vel_vec = interceptor.state.velocity * np.array([
            np.cos(interceptor.state.heading) * np.cos(interceptor.state.flight_path_angle),
            np.sin(interceptor.state.heading) * np.cos(interceptor.state.flight_path_angle),
            -np.sin(interceptor.state.flight_path_angle)
        ])
        tgt_vel_vec = target.state.velocity * np.array([
            np.cos(target.state.heading) * np.cos(target.state.flight_path_angle),
            np.sin(target.state.heading) * np.cos(target.state.flight_path_angle),
            -np.sin(target.state.flight_path_angle)
        ])
        rel_vel = tgt_vel_vec - int_vel_vec
        closing_speed = -np.dot(rel_vel, rel_pos) / current_range if current_range > 0 else 0
        
        # Check for intercept
        if current_range < 25:
            intercept_achieved = True
            print(f"\n[T={sim_time:.1f}s] INTERCEPT ACHIEVED! Final range: {current_range:.1f}m")
            break
            
        # Check for mission failures
        if interceptor.state.velocity < 25:
            print(f"\n[T={sim_time:.1f}s] INTERCEPTOR STALLED! Speed: {interceptor.state.velocity:.1f} m/s")
            break
            
        if interceptor.state.fuel_remaining <= 0:
            print(f"\n[T={sim_time:.1f}s] INTERCEPTOR OUT OF FUEL!")
            break
            
        # Update guidance
        guidance_commands = guidance.update(
            interceptor.state,
            target.state,
            dt
        )
        
        # Check if intercept complete
        if guidance_commands is None:
            intercept_achieved = True
            break
            
        # Apply guidance commands to interceptor
        if guidance_commands:
            # Convert commanded heading to bank angle
            heading_error = guidance_commands['commanded_heading'] - interceptor.state.heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Proportional-derivative control for bank angle
            if not hasattr(interceptor, 'prev_heading_error'):
                interceptor.prev_heading_error = 0
            heading_rate_error = (heading_error - interceptor.prev_heading_error) / dt
            interceptor.prev_heading_error = heading_error
            
            # More responsive banking
            desired_bank = np.clip(
                heading_error * 2.0 + heading_rate_error * 0.3,
                -np.radians(60), 
                np.radians(60)
            )
            
            # Altitude control
            alt_error = guidance_commands['commanded_altitude'] - interceptor.state.position[2]
            pitch_adjust = np.clip(alt_error * 0.001, -0.15, 0.15)
            
            # Set controls
            interceptor.set_controls(
                bank_angle=desired_bank,
                throttle=guidance_commands['commanded_throttle']
            )
            
            # Adjust flight path angle for altitude control
            if abs(alt_error) > 50:
                interceptor.state.flight_path_angle = pitch_adjust
                
            # Store performance data
            if sim_time % 1.0 < dt:  # Every second
                performance_data['time'].append(sim_time)
                performance_data['range'].append(current_range)
                performance_data['closing_speed'].append(closing_speed)
                performance_data['int_speed'].append(interceptor.state.velocity)
                performance_data['throttle'].append(guidance_commands['commanded_throttle'])
                performance_data['predicted_miss'].append(
                    guidance_commands.get('predicted_miss', current_range)
                )
        
        # Target maneuvers (optional)
        if sim_time > 30 and sim_time < 35:
            # Mild evasive turn
            target.set_controls(
                bank_angle=np.radians(15),
                throttle=0.6
            )
        elif sim_time > 50 and sim_time < 55:
            # Another turn
            target.set_controls(
                bank_angle=np.radians(-20),
                throttle=0.7
            )
        else:
            # Straight flight
            target.set_controls(
                bank_angle=0,
                throttle=0.5
            )
        
        # Update aircraft dynamics
        interceptor.update(dt)
        target.update(dt)
        
        sim_time += dt
        
        # Print status
        if sim_time - last_print_time >= 5.0:
            print(f"T={sim_time:6.1f}s | Phase: {guidance.current_phase.name:10s} | "
                  f"Range: {current_range:7.1f}m | "
                  f"Closing: {closing_speed:6.1f} m/s | "
                  f"Int Speed: {interceptor.state.velocity:6.1f} m/s | "
                  f"Throttle: {guidance_commands['commanded_throttle']:.2f}")
            last_print_time = sim_time
    
    # Convert trajectories to arrays
    interceptor_trajectory = np.array(interceptor_trajectory)
    target_trajectory = np.array(target_trajectory)
    
    # Final summary
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
        print(f"Average closing speed: {np.mean(performance_data['closing_speed']):.1f} m/s")
    else:
        min_range = np.min(np.linalg.norm(
            target_trajectory[:len(interceptor_trajectory)] - interceptor_trajectory, axis=1
        ))
        print(f"Result: Target escaped or timeout")
        print(f"Minimum range achieved: {min_range:.1f} meters")
    
    # Plot results
    plot_intercept_results(interceptor_trajectory, target_trajectory, guidance, performance_data)
    
    return interceptor_trajectory, target_trajectory, guidance


def plot_intercept_results(interceptor_traj, target_traj, guidance, perf_data):
    """Enhanced plotting with performance metrics"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], interceptor_traj[:, 2], 
             'b-', linewidth=2, label='Interceptor')
    ax1.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], 
             'r-', linewidth=2, label='Target')
    ax1.scatter(*interceptor_traj[0], color='blue', s=100, marker='o')
    ax1.scatter(*target_traj[0], color='red', s=100, marker='o')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Engagement Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Top-down view
    ax2 = fig.add_subplot(232)
    ax2.plot(interceptor_traj[:, 0], interceptor_traj[:, 1], 'b-', linewidth=2, label='Interceptor')
    ax2.plot(target_traj[:, 0], target_traj[:, 1], 'r-', linewidth=2, label='Target')
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_title('Top-Down View')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Range vs Time
    ax3 = fig.add_subplot(233)
    if perf_data['time']:
        ax3.plot(perf_data['time'], perf_data['range'], 'g-', linewidth=2)
        ax3.axhline(y=25, color='r', linestyle='--', label='Intercept Range')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Range (m)')
        ax3.set_title('Range to Target')
        ax3.legend()
        ax3.grid(True)
    
    # Closing Speed
    ax4 = fig.add_subplot(234)
    if perf_data['time']:
        ax4.plot(perf_data['time'], perf_data['closing_speed'], 'c-', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Closing Speed (m/s)')
        ax4.set_title('Closing Velocity (positive = closing)')
        ax4.grid(True)
    
    # Speed and Throttle
    ax5 = fig.add_subplot(235)
    if perf_data['time']:
        ax5.plot(perf_data['time'], perf_data['int_speed'], 'b-', linewidth=2, label='Speed')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(perf_data['time'], perf_data['throttle'], 'r--', linewidth=2, label='Throttle')
        ax5_twin.set_ylabel('Throttle', color='r')
        ax5_twin.tick_params(axis='y', labelcolor='r')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Speed (m/s)', color='b')
        ax5.tick_params(axis='y', labelcolor='b')
        ax5.set_title('Speed & Throttle Control')
        ax5.grid(True)
    
    # Altitude profiles
    ax6 = fig.add_subplot(236)
    time_array = np.arange(len(interceptor_traj)) * 0.1
    ax6.plot(time_array, interceptor_traj[:, 2], 'b-', linewidth=2, label='Interceptor')
    ax6.plot(time_array[:len(target_traj)], target_traj[:, 2], 'r-', linewidth=2, label='Target')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Altitude (m)')
    ax6.set_title('Altitude Profiles')
    ax6.legend()
    ax6.grid(True)
    
    plt.suptitle(f'1v1 Intercept Scenario - Final Phase: {guidance.current_phase.name}', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simple_intercept()