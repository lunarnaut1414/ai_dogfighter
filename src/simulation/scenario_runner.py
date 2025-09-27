"""
Scenario runner for executing simulation scenarios.
Manages simulation flow, objectives evaluation, and data recording.
"""

import yaml
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..battlespace.core import Battlespace
from ..assets.asset_manager import AssetManager, AssetType
from ..assets.sensor_model import SensorModel
from .objectives import (
    ObjectiveManager, ObjectiveStatus, InterceptEvent
)


class ScenarioState(Enum):
    """Scenario execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    start_time: float = 0.0
    end_time: float = 0.0
    total_steps: int = 0
    intercept_events: List[InterceptEvent] = field(default_factory=list)
    update_times: List[float] = field(default_factory=list)
    objectives_completed: int = 0
    objectives_failed: int = 0
    
    @property
    def mean_update_time(self) -> float:
        return np.mean(self.update_times) if self.update_times else 0.0
    
    @property
    def max_update_time(self) -> float:
        return np.max(self.update_times) if self.update_times else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.objectives_completed + self.objectives_failed
        return self.objectives_completed / total if total > 0 else 0.0


class ScenarioRunner:
    """
    Executes simulation scenarios with objectives and metrics.
    """
    
    def __init__(self, 
                 scenario_config: Any,
                 guidance_algorithm: Optional[Callable] = None,
                 realtime: bool = False,
                 verbose: bool = True):
        """
        Initialize scenario runner.
        
        Args:
            scenario_config: Path to YAML file or config dictionary
            guidance_algorithm: Guidance algorithm instance with compute_commands method
            realtime: Run in realtime vs as-fast-as-possible
            verbose: Print status updates
        """
        # Load configuration
        if isinstance(scenario_config, str):
            with open(scenario_config, 'r') as f:
                self.config = yaml.safe_load(f)['scenario']
        else:
            self.config = scenario_config
            
        self.guidance_algorithm = guidance_algorithm
        self.realtime = realtime
        self.verbose = verbose
        
        # Core components
        self.battlespace = None
        self.asset_manager = None
        self.sensor_model = None
        self.objective_manager = ObjectiveManager()
        
        # Asset tracking
        self.interceptor_id = None
        self.target_ids = []
        
        # State management
        self.state = ScenarioState.INITIALIZING
        self.metrics = SimulationMetrics()
        self.termination_conditions = []
        
        # Recording
        self.recording_enabled = False
        self.recording_data = {
            'scenario': self.config.get('name', 'Unknown'),
            'config': self.config,
            'frames': [],
            'events': [],
            'metrics': {}
        }
        
    def setup(self):
        """Initialize all components for the scenario."""
        if self.verbose:
            print("\n" + "="*60)
            print(f"Setting up scenario: {self.config.get('name', 'Unnamed')}")
            print(f"Description: {self.config.get('description', 'No description')}")
            print("="*60)
            
        # Initialize components
        self._setup_battlespace()
        self._setup_asset_manager()
        self._spawn_assets()
        self._setup_sensors()
        self._setup_objectives()
        self._setup_termination()
        self._setup_recording()
        
        # Start objectives
        self.objective_manager.start_all(0.0)
        
        self.state = ScenarioState.RUNNING
        self.metrics.start_time = time.time()
        
        if self.verbose:
            print("\nScenario setup complete!")
            print(f"  Interceptor: {self.interceptor_id}")
            print(f"  Targets: {len(self.target_ids)}")
            print(f"  Objectives: {len(self.objective_manager.objectives)}")
            
    def _setup_battlespace(self):
        """Initialize battlespace environment."""
        battlespace_config = self.config.get('battlespace', 
                                            'configs/battlespace/default_battlespace.yaml')
        
        if self.verbose:
            print(f"\nInitializing battlespace: {battlespace_config}")
            
        self.battlespace = Battlespace(config_file=battlespace_config)
        self.battlespace.generate(seed=self.config.get('seed', 42))
        
        # Apply weather override if specified
        if 'weather_preset' in self.config:
            self.battlespace.weather.set_preset(self.config['weather_preset'])
        if 'wind_override' in self.config:
            wind = self.config['wind_override']
            self.battlespace.weather.set_wind(wind.get('base_vector', [0, 0, 0]))
            
    def _setup_asset_manager(self):
        """Initialize asset manager."""
        update_rate = self.config.get('update_rate', 50)
        dt = 1.0 / update_rate
        
        if self.verbose:
            print(f"Creating asset manager (dt={dt:.3f}s, rate={update_rate}Hz)")
            
        self.asset_manager = AssetManager(self.battlespace, dt=dt)
        
    def _spawn_assets(self):
        """Spawn all aircraft defined in the scenario."""
        # Spawn interceptor
        interceptor_config = self.config['interceptor']
        aircraft_file = interceptor_config.get('aircraft', 'configs/aircraft/interceptor_drone.yaml')
        
        # Load aircraft configuration
        with open(aircraft_file, 'r') as f:
            aircraft_params = yaml.safe_load(f)
            
        # Merge with initial state
        spawn_config = {
            **aircraft_params,
            'initial_state': interceptor_config['initial_state']
        }
        
        if self.verbose:
            print(f"\nSpawning interceptor: {interceptor_config['id']}")
            
        self.interceptor_id = self.asset_manager.spawn_aircraft(
            config=spawn_config,
            asset_id=interceptor_config['id'],
            asset_type=AssetType.INTERCEPTOR,
            team=interceptor_config.get('team', 'blue')
        )
        
        # Spawn targets
        for target_config in self.config.get('targets', []):
            aircraft_file = target_config.get('aircraft', 'configs/aircraft/target_basic.yaml')
            
            with open(aircraft_file, 'r') as f:
                aircraft_params = yaml.safe_load(f)
                
            spawn_config = {
                **aircraft_params,
                'initial_state': target_config['initial_state'],
                'behavior': target_config.get('behavior', 'waypoint')
            }
            
            # Add behavior-specific parameters
            if 'waypoints' in target_config:
                spawn_config['waypoints'] = target_config['waypoints']
            if 'orbit_center' in target_config:
                spawn_config['orbit_center'] = target_config['orbit_center']
                spawn_config['orbit_radius'] = target_config.get('orbit_radius', 1000)
                
            if self.verbose:
                print(f"Spawning target: {target_config['id']} "
                      f"({target_config.get('behavior', 'waypoint')} behavior)")
                
            target_id = self.asset_manager.spawn_aircraft(
                config=spawn_config,
                asset_id=target_config['id'],
                asset_type=AssetType.TARGET,
                team=target_config.get('team', 'red')
            )
            
            self.target_ids.append(target_id)
            
            # Set priority if specified
            if 'priority' in target_config:
                # Don't try to subscript the assets directly
                self.asset_manager.assets._assets[target_id].priority = target_config['priority']
                
    def _setup_sensors(self):
        """Initialize sensor model."""
        sensor_config = self.config.get('sensors', {})
        
        if self.verbose:
            print("\nInitializing sensor model")
        
        # Create sensor configuration dictionary in the format SensorModel expects
        # The SensorModel constructor expects a single 'sensor_config' parameter
        sensor_dict = {
            'type': sensor_config.get('type', 'radar'),
            'max_range': sensor_config.get('max_range', 10000.0),
            'fov_azimuth': sensor_config.get('fov_azimuth', 60.0),  # in degrees
            'fov_elevation': sensor_config.get('fov_elevation', 45.0),  # in degrees
            'update_rate': sensor_config.get('update_rate', 10.0),
            'position_error': sensor_config.get('position_error', 50.0),
            'velocity_error': sensor_config.get('velocity_error', 5.0),
            'min_rcs': sensor_config.get('min_rcs', 0.1)
        }
        
        # Initialize sensor model with configuration dictionary
        # IMPORTANT: Pass as 'sensor_config' parameter, not individual kwargs
        self.sensor_model = SensorModel(sensor_config=sensor_dict)
        
        if self.verbose:
            print(f"  Sensor type: {sensor_dict['type']}")
            print(f"  Max range: {sensor_dict['max_range']}m")
            print(f"  FOV: {sensor_dict['fov_azimuth']}° x {sensor_dict['fov_elevation']}°")
            print(f"  Update rate: {sensor_dict['update_rate']}Hz")
        
    def _setup_objectives(self):
        """Setup scenario objectives."""
        objectives_config = self.config.get('objectives', {})
        
        # Add primary objectives
        for obj_config in objectives_config.get('primary', []):
            self.objective_manager.add_objective(obj_config, primary=True)
            
        # Add secondary objectives
        for obj_config in objectives_config.get('secondary', []):
            self.objective_manager.add_objective(obj_config, primary=False)
            
        if self.verbose:
            print(f"\nConfigured objectives:")
            print(f"  Primary: {len(self.objective_manager.primary_objectives)}")
            print(f"  Secondary: {len(self.objective_manager.secondary_objectives)}")
            
    def _setup_termination(self):
        """Setup termination conditions."""
        termination = self.config.get('termination', {})
        conditions = termination.get('conditions', [])
        
        for condition in conditions:
            if isinstance(condition, dict):
                self.termination_conditions.append(condition)
                
        if self.verbose and self.termination_conditions:
            print(f"\nTermination conditions: {len(self.termination_conditions)}")
            
    def _setup_recording(self):
        """Setup data recording if enabled."""
        recording_config = self.config.get('recording', {})
        self.recording_enabled = recording_config.get('enabled', False)
        
        if self.recording_enabled:
            self.recording_frequency = recording_config.get('frequency', 10)
            self.recording_file = recording_config.get('output_file', 
                                                      f"data/scenario_run_{time.time()}.json")
            self.last_record_time = 0
            
            if self.verbose:
                print(f"\nRecording enabled: {self.recording_file}")
                
    def run(self) -> Dict[str, Any]:
        """
        Run the scenario to completion.
        
        Returns:
            Dictionary with scenario results
        """
        if self.state != ScenarioState.RUNNING:
            raise RuntimeError(f"Cannot run scenario in state {self.state}")
            
        if self.verbose:
            print("\n" + "="*60)
            print("SCENARIO EXECUTION STARTED")
            print("="*60)
            
        max_duration = self.config.get('duration', 600)
        last_status_time = 0
        status_interval = 5.0  # Print status every 5 seconds
        
        try:
            while self.state == ScenarioState.RUNNING:
                step_start = time.time()
                
                # Update simulation - temporarily swap back to original assets
                self.asset_manager.assets = self.asset_manager._original_assets
                self.asset_manager.update()
                self.asset_manager.assets = self.asset_manager._wrapped_assets
                
                # Get interceptor state
                interceptor = self.asset_manager.get_asset(self.interceptor_id)
                if not interceptor:
                    if self.verbose:
                        print("WARNING: Interceptor lost!")
                    self.state = ScenarioState.FAILED
                    break
                    
                # Update sensors
                if self.sensor_model:
                    detections = []
                    interceptor_pos = interceptor['state']['position']
                    interceptor_vel = interceptor['state'].get('velocity_vector', [0, 0, 0])
                    
                    for target_id in self.target_ids:
                        target = self.asset_manager.get_asset(target_id)
                        if target:
                            detection = self.sensor_model.detect(
                                sensor_position=interceptor_pos,
                                sensor_velocity=interceptor_vel,
                                target_position=target['state']['position'],
                                target_velocity=target['state'].get('velocity_vector', [0, 0, 0]),
                                battlespace=self.battlespace
                            )
                            if detection:
                                detections.append((target_id, detection))
                                
                # Run guidance algorithm
                if self.guidance_algorithm and detections:
                    # Prepare detection data for guidance
                    target_states = {}
                    for target_id, detection in detections:
                        target_states[target_id] = {
                            'position': detection['position'],
                            'velocity': detection.get('velocity', [0, 0, 0]),
                            'priority': self.asset_manager.get_asset(target_id).get('priority', 'medium')
                        }
                        
                    # Compute guidance commands
                    commands = self.guidance_algorithm.compute_commands(
                        interceptor_state=interceptor['state'],
                        target_states=target_states,
                        battlespace=self.battlespace
                    )
                    
                    # Apply commands to interceptor
                    if commands:
                        self.asset_manager.set_commands(self.interceptor_id, commands)
                        
                # Evaluate objectives
                obj_results = self.objective_manager.evaluate_all(
                    self.asset_manager, 
                    self.asset_manager.time
                )
                
                # Check termination conditions
                if self._check_termination(obj_results):
                    self.state = ScenarioState.COMPLETED
                    
                # Record data
                if self.recording_enabled:
                    self._record_frame()
                    
                # Status update
                if self.verbose and self.asset_manager.time - last_status_time >= status_interval:
                    self._print_status(obj_results)
                    last_status_time = self.asset_manager.time
                    
                # Track metrics
                self.metrics.total_steps += 1
                self.metrics.update_times.append(time.time() - step_start)
                
                # Check time limit
                if self.asset_manager.time >= max_duration:
                    if self.verbose:
                        print(f"\nMax duration reached ({max_duration}s)")
                    self.state = ScenarioState.COMPLETED
                    
                # Realtime synchronization
                if self.realtime:
                    elapsed = time.time() - step_start
                    sleep_time = self.asset_manager.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            if self.verbose:
                print("\n\nScenario interrupted by user")
            self.state = ScenarioState.ABORTED
            
        except Exception as e:
            if self.verbose:
                print(f"\n\nERROR: {e}")
            self.state = ScenarioState.FAILED
            raise
            
        finally:
            # Finalize metrics
            self.metrics.end_time = time.time()
            
            # Get final objective status
            final_objectives = self.objective_manager.evaluate_all(
                self.asset_manager,
                self.asset_manager.time
            )
            
            self.metrics.objectives_completed = len(self.objective_manager.completed_objectives)
            self.metrics.objectives_failed = len(self.objective_manager.failed_objectives)
            self.metrics.intercept_events = self.objective_manager.get_intercept_events()
            
            # Save recording
            if self.recording_enabled:
                self._save_recording()
                
            # Print final report
            if self.verbose:
                self._print_final_report()
                
        return self.get_results()
        
    def _check_termination(self, obj_results: Dict[str, Any]) -> bool:
        """Check if any termination condition is met."""
        # Check objective-based termination
        if obj_results['all_completed']:
            if self.verbose:
                print("\nAll objectives completed!")
            return True
            
        if obj_results['any_failed']:
            required_failed = any(obj.required and obj.status == ObjectiveStatus.FAILED 
                                 for obj in self.objective_manager.objectives)
            if required_failed:
                if self.verbose:
                    print("\nRequired objective failed!")
                return True
                
        # Check specific termination conditions
        for condition in self.termination_conditions:
            if 'max_time' in condition:
                if self.asset_manager.time >= condition['max_time']:
                    return True
                    
            if 'all_targets_neutralized' in condition:
                if condition['all_targets_neutralized']:
                    all_intercepted = all(
                        event.success for event in self.metrics.intercept_events
                    )
                    if all_intercepted and len(self.metrics.intercept_events) >= len(self.target_ids):
                        return True
                        
            if 'interceptor_crashed' in condition:
                interceptor = self.asset_manager.get_asset(self.interceptor_id)
                if interceptor and interceptor['state']['position'][2] <= 0:
                    return True
                    
            if 'interceptor_fuel_empty' in condition:
                interceptor = self.asset_manager.get_asset(self.interceptor_id)
                if interceptor and interceptor['state'].get('fuel_fraction', 1.0) <= 0:
                    return True
                    
        return False
        
    def _record_frame(self):
        """Record current frame data."""
        if self.asset_manager.time - self.last_record_time < 1.0 / self.recording_frequency:
            return
            
        frame_data = {
            'time': self.asset_manager.time,
            'interceptor': {
                'position': self.asset_manager.get_asset(self.interceptor_id)['state']['position'].tolist(),
                'velocity': self.asset_manager.get_asset(self.interceptor_id)['state'].get('velocity', 0),
                'heading': self.asset_manager.get_asset(self.interceptor_id)['state'].get('heading', 0),
                'altitude': self.asset_manager.get_asset(self.interceptor_id)['state']['position'][2],
                'fuel': self.asset_manager.get_asset(self.interceptor_id)['state'].get('fuel_fraction', 1.0)
            },
            'targets': {}
        }
        
        for target_id in self.target_ids:
            target = self.asset_manager.get_asset(target_id)
            if target:
                frame_data['targets'][target_id] = {
                    'position': target['state']['position'].tolist(),
                    'velocity': target['state'].get('velocity', 0),
                    'heading': target['state'].get('heading', 0),
                    'altitude': target['state']['position'][2]
                }
                
        self.recording_data['frames'].append(frame_data)
        self.last_record_time = self.asset_manager.time
        
    def _save_recording(self):
        """Save recorded data to file."""
        if not self.recording_data['frames']:
            return
            
        # Add metrics to recording
        self.recording_data['metrics'] = {
            'duration': self.asset_manager.time,
            'steps': self.metrics.total_steps,
            'intercepts': len(self.metrics.intercept_events),
            'success_rate': self.metrics.success_rate,
            'mean_update_time': self.metrics.mean_update_time
        }
        
        # Create output directory if needed
        output_path = Path(self.recording_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(self.recording_data, f, indent=2)
            
        if self.verbose:
            print(f"\nRecording saved to: {output_path}")
            
    def _print_status(self, obj_results: Dict[str, Any]):
        """Print current scenario status."""
        print(f"\n[T={self.asset_manager.time:6.1f}s] Status Update")
        print("-" * 40)
        
        # Interceptor status
        interceptor = self.asset_manager.get_asset(self.interceptor_id)
        if interceptor:
            pos = interceptor['state']['position']
            fuel = interceptor['state'].get('fuel_fraction', 1.0)
            print(f"Interceptor: Pos({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}m) "
                  f"Fuel: {fuel:.1%}")
            
        # Target status
        for target_id in self.target_ids[:3]:  # Show first 3 targets
            target = self.asset_manager.get_asset(target_id)
            if target:
                pos = target['state']['position']
                range_to_target = np.linalg.norm(
                    np.array(pos) - np.array(interceptor['state']['position'])
                )
                print(f"  {target_id}: {range_to_target:.0f}m")
                
        # Objective status
        print(f"Objectives: {obj_results['overall_progress']:.0%} complete")
        
    def _print_final_report(self):
        """Print final simulation report."""
        print("\n" + "="*60)
        print("SCENARIO COMPLETE")
        print("="*60)
        
        print(f"\nScenario: {self.config.get('name', 'Unknown')}")
        print(f"Status: {self.state.value}")
        print(f"Duration: {self.asset_manager.time:.1f}s")
        print(f"Real time: {self.metrics.end_time - self.metrics.start_time:.1f}s")
        print(f"Time factor: {self.asset_manager.time / (self.metrics.end_time - self.metrics.start_time):.1f}x")
        
        print("\nObjectives Summary:")
        summary = self.objective_manager.get_summary()
        print(f"  Total: {summary['total']}")
        print(f"  Completed: {summary['completed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['completion_rate']:.1%}")
        
        if self.metrics.intercept_events:
            print(f"\nIntercept Events: {len(self.metrics.intercept_events)}")
            for event in self.metrics.intercept_events[:5]:
                print(f"  {event.target_id}: {event.range:.1f}m @ T={event.time:.1f}s")
                
        print("\nPerformance Metrics:")
        print(f"  Total steps: {self.metrics.total_steps}")
        print(f"  Mean update: {self.metrics.mean_update_time*1000:.2f}ms")
        print(f"  Max update: {self.metrics.max_update_time*1000:.2f}ms")
        print(f"  Update rate: {1.0/self.metrics.mean_update_time:.1f}Hz")
        
    def get_results(self) -> Dict[str, Any]:
        """Get scenario results as dictionary."""
        return {
            'scenario': self.config.get('name'),
            'state': self.state.value,
            'duration': self.asset_manager.time if self.asset_manager else 0,
            'objectives': self.objective_manager.get_summary(),
            'metrics': {
                'total_steps': self.metrics.total_steps,
                'intercepts': len(self.metrics.intercept_events),
                'success_rate': self.metrics.success_rate,
                'mean_update_time_ms': self.metrics.mean_update_time * 1000,
                'max_update_time_ms': self.metrics.max_update_time * 1000
            },
            'events': [asdict(event) for event in self.metrics.intercept_events]
        }
        
    def pause(self):
        """Pause scenario execution."""
        self.state = ScenarioState.PAUSED
        
    def resume(self):
        """Resume scenario execution."""
        if self.state == ScenarioState.PAUSED:
            self.state = ScenarioState.RUNNING
            
    def abort(self):
        """Abort scenario execution."""
        self.state = ScenarioState.ABORTED