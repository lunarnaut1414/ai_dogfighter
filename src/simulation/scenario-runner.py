"""
Scenario execution engine for running and evaluating intercept missions.
Manages simulation flow, metrics collection, and success criteria.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
import time
from pathlib import Path
from datetime import datetime

from src.battlespace import Battlespace
from src.assets.asset_manager import AssetManager, AssetType
from src.assets.flight_controller import FlightController, BehaviorMode
from src.assets.target_behaviors import TargetBehaviorController, TargetDifficulty, ThreatAssessment
from src.assets.sensor_model import SensorModel, TargetTrack
from src.assets.aircraft_3dof import FlightMode


class ScenarioState(Enum):
    """Scenario execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class ObjectiveType(Enum):
    """Types of scenario objectives"""
    INTERCEPT = "intercept"
    INTERCEPT_ALL = "intercept_all"
    PRIORITIZED_INTERCEPT = "prioritized_intercept"
    TIME_LIMIT = "time_limit"
    FUEL_REMAINING = "fuel_remaining"
    NO_COLLISION = "no_collision"
    SURVIVE = "survive"
    DEFEND_AREA = "defend_area"


@dataclass
class ObjectiveStatus:
    """Status of a scenario objective"""
    objective_type: ObjectiveType
    description: str
    required: bool = True  # Primary vs secondary
    completed: bool = False
    failed: bool = False
    progress: float = 0.0  # 0-1 progress indicator
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterceptEvent:
    """Record of an intercept event"""
    time: float
    interceptor_id: str
    target_id: str
    range: float
    closing_velocity: float
    interceptor_fuel: float
    success: bool


@dataclass
class SimulationMetrics:
    """Comprehensive simulation metrics"""
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    total_steps: int = 0
    
    # Performance
    intercept_events: List[InterceptEvent] = field(default_factory=list)
    min_range_achieved: Dict[str, float] = field(default_factory=dict)
    total_distance_traveled: Dict[str, float] = field(default_factory=dict)
    fuel_consumed: Dict[str, float] = field(default_factory=dict)
    
    # Computation
    update_times: List[float] = field(default_factory=list)
    mean_update_time: float = 0.0
    max_update_time: float = 0.0
    
    # Success metrics
    objectives_completed: int = 0
    objectives_failed: int = 0
    success_rate: float = 0.0
    
    # Tactical metrics
    time_in_pursuit: float = 0.0
    time_evading: float = 0.0
    max_g_pulled: float = 0.0
    terrain_collisions: int = 0


class ScenarioRunner:
    """
    Main scenario execution engine.
    Manages simulation flow and evaluates objectives.
    """
    
    def __init__(self, scenario_config: Dict[str, Any], 
                 guidance_algorithm: Optional[Callable] = None,
                 realtime: bool = False,
                 verbose: bool = True):
        """
        Initialize scenario runner.
        
        Args:
            scenario_config: Scenario configuration dictionary or file path
            guidance_algorithm: Guidance algorithm to control interceptor
            realtime: Run in realtime vs as-fast-as-possible
            verbose: Print status updates
        """
        # Load scenario configuration
        if isinstance(scenario_config, str):
            with open(scenario_config, 'r') as f:
                self.config = yaml.safe_load(f)['scenario']
        else:
            self.config = scenario_config
            
        self.guidance_algorithm = guidance_algorithm
        self.realtime = realtime
        self.verbose = verbose
        
        # Core components (will be initialized in setup)
        self.battlespace = None
        self.asset_manager = None
        self.sensor_model = None
        
        # Asset tracking
        self.interceptor_id = None
        self.target_ids = []
        self.target_controllers = {}
        
        # Objectives
        self.objectives: List[ObjectiveStatus] = []
        self.termination_conditions = []
        
        # State
        self.state = ScenarioState.INITIALIZING
        self.metrics = SimulationMetrics()
        
        # Recording
        self.recording_enabled = False
        self.recording_data = []
        
    def setup(self):
        """
        Initialize all components for the scenario.
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"Setting up scenario: {self.config.get('name', 'Unnamed')}")
            print(f"Description: {self.config.get('description', 'No description')}")
            print("="*60)
            
        # Initialize battlespace
        self._setup_battlespace()
        
        # Initialize asset manager
        self._setup_asset_manager()
        
        # Spawn interceptor and targets
        self._spawn_assets()
        
        # Initialize sensor model
        self._setup_sensors()
        
        # Setup objectives
        self._setup_objectives()
        
        # Setup recording if enabled
        self._setup_recording()
        
        self.state = ScenarioState.RUNNING
        self.metrics.start_time = time.time()
        
        if self.verbose:
            print("\nScenario setup complete!")
            print(f"  Interceptor: {self.interceptor_id}")
            print(f"  Targets: {len(self.target_ids)} spawned")
            print(f"  Objectives: {len(self.objectives)} defined")
            
    def _setup_battlespace(self):
        """Initialize battlespace environment"""
        battlespace_config = self.config.get('battlespace', 
                                            'configs/battlespace/default_battlespace.yaml')
        
        if self.verbose:
            print(f"\nInitializing battlespace: {battlespace_config}")
            
        self.battlespace = Battlespace(config_file=battlespace_config)
        self.battlespace.generate(seed=self.config.get('seed', 42))
        
    def _setup_asset_manager(self):
        """Initialize asset manager"""
        update_rate = self.config.get('update_rate', 50)
        dt = 1.0 / update_rate
        
        if self.verbose:
            print(f"Creating asset manager (dt={dt:.3f}s, rate={update_rate}Hz)")
            
        self.asset_manager = AssetManager(self.battlespace, dt=dt)
        
    def _spawn_assets(self):
        """Spawn all aircraft in the scenario"""
        # Spawn interceptor
        interceptor_config = self.config['interceptor']
        
        if self.verbose:
            print(f"\nSpawning interceptor: {interceptor_config['id']}")
            
        self.interceptor_id = self.asset_manager.spawn_aircraft(
            config=interceptor_config,
            asset_id=interceptor_config['id'],
            asset_type=AssetType.INTERCEPTOR,
            team=interceptor_config.get('team', 'blue')
        )
        
        # Spawn targets
        for target_config in self.config.get('targets', []):
            if self.verbose:
                print(f"Spawning target: {target_config['id']}")
                
            target_id = self.asset_manager.spawn_aircraft(
                config=target_config,
                asset_id=target_config['id'],
                asset_type=AssetType.TARGET,
                team=target_config.get('team', 'red')
            )
            self.target_ids.append(target_id)
            
            # Create behavior controller for target
            aircraft = self.asset_manager.assets[target_id].aircraft
            
            # Determine difficulty level
            difficulty_map = {
                'static': TargetDifficulty.STATIC,
                'predictable': TargetDifficulty.PREDICTABLE,
                'reactive': TargetDifficulty.REACTIVE,
                'tactical': TargetDifficulty.TACTICAL,
                'intelligent': TargetDifficulty.INTELLIGENT
            }
            
            difficulty = difficulty_map.get(
                target_config.get('difficulty', 'predictable'),
                TargetDifficulty.PREDICTABLE
            )
            
            controller = TargetBehaviorController(aircraft.config, difficulty)
            
            # Set up waypoints if specified
            if 'waypoints' in target_config:
                waypoints = [np.array(wp, dtype=np.float64) 
                           for wp in target_config['waypoints']]
                controller.base_controller.set_waypoints(waypoints)
                controller.base_controller.set_mode(BehaviorMode.WAYPOINT)
                
            self.target_controllers[target_id] = controller
            
    def _setup_sensors(self):
        """Initialize sensor model for interceptor"""
        # Get sensor config from interceptor aircraft
        interceptor = self.asset_manager.assets[self.interceptor_id].aircraft
        
        sensor_config = interceptor.config.get('sensors', {}).get('radar', {
            'max_range': 10000,
            'fov_azimuth': 120,
            'fov_elevation': 60,
            'update_rate': 10,
            'position_error': 50,
            'velocity_error': 5
        })
        
        self.sensor_model = SensorModel(sensor_config)
        
        if self.verbose:
            print(f"Sensor initialized: Range={sensor_config['max_range']}m, "
                  f"FOV={sensor_config['fov_azimuth']}°")
            
    def _setup_objectives(self):
        """Parse and setup scenario objectives"""
        objectives_config = self.config.get('objectives', {})
        
        # Primary objectives
        for obj_config in objectives_config.get('primary', []):
            objective = self._create_objective(obj_config, required=True)
            self.objectives.append(objective)
            
        # Secondary objectives
        for obj_config in objectives_config.get('secondary', []):
            objective = self._create_objective(obj_config, required=False)
            self.objectives.append(objective)
            
        # Termination conditions
        self.termination_conditions = self.config.get('termination', {}).get('conditions', [])
        
    def _create_objective(self, config: Dict[str, Any], required: bool) -> ObjectiveStatus:
        """Create objective from configuration"""
        obj_type = ObjectiveType(config['type'])
        
        if obj_type == ObjectiveType.INTERCEPT:
            description = f"Intercept {config.get('target_id', 'any target')}"
            details = {
                'target_id': config.get('target_id'),
                'range': config.get('range', 100),
                'time_limit': config.get('time_limit')
            }
        elif obj_type == ObjectiveType.TIME_LIMIT:
            description = f"Complete within {config.get('seconds', 300)}s"
            details = {'time_limit': config.get('seconds', 300)}
        elif obj_type == ObjectiveType.FUEL_REMAINING:
            description = f"Maintain {config.get('min_fraction', 0.2)*100:.0f}% fuel"
            details = {'min_fraction': config.get('min_fraction', 0.2)}
        else:
            description = f"{obj_type.value} objective"
            details = config
            
        return ObjectiveStatus(
            objective_type=obj_type,
            description=description,
            required=required,
            details=details
        )
        
    def _setup_recording(self):
        """Setup data recording if enabled"""
        recording_config = self.config.get('recording', {})
        self.recording_enabled = recording_config.get('enabled', False)
        
        if self.recording_enabled:
            self.recording_frequency = recording_config.get('frequency', 10)
            self.recording_file = recording_config.get('output_file', 
                                                      f"scenario_run_{datetime.now():%Y%m%d_%H%M%S}.json")
            self.recording_data = []
            
            if self.verbose:
                print(f"Recording enabled: {self.recording_file} @ {self.recording_frequency}Hz")
                
    def run(self) -> SimulationMetrics:
        """
        Execute the scenario.
        
        Returns:
            Simulation metrics and results
        """
        if self.state != ScenarioState.RUNNING:
            self.setup()
            
        max_time = self.config.get('duration', 300)
        dt = self.asset_manager.dt
        
        if self.verbose:
            print(f"\nStarting simulation for {max_time}s...")
            print("\nTime | Interceptor | Closest Target | Range | Status")
            print("-" * 60)
            
        last_print_time = 0
        recording_counter = 0
        
        try:
            while self.asset_manager.time < max_time:
                step_start = time.perf_counter()
                
                # Update targets with behaviors
                self._update_target_behaviors()
                
                # Get sensor detections
                detections = self._get_sensor_detections()
                
                # Run guidance algorithm for interceptor
                if self.guidance_algorithm:
                    self._run_guidance_algorithm(detections)
                else:
                    # Simple default pursuit for testing
                    self._default_interceptor_behavior(detections)
                    
                # Update all assets
                self.asset_manager.update()
                
                # Check objectives and termination
                self._evaluate_objectives()
                
                if self._check_termination():
                    break
                    
                # Record data
                if self.recording_enabled:
                    recording_counter += 1
                    if recording_counter >= (50 / self.recording_frequency):
                        self._record_frame()
                        recording_counter = 0
                        
                # Track metrics
                step_time = time.perf_counter() - step_start
                self.metrics.update_times.append(step_time)
                self.metrics.total_steps += 1
                
                # Print status
                if self.verbose and self.asset_manager.time - last_print_time >= 1.0:
                    self._print_status()
                    last_print_time = self.asset_manager.time
                    
                # Realtime delay if requested
                if self.realtime:
                    sleep_time = dt - step_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            self.state = ScenarioState.TERMINATED
            
        # Finalize simulation
        self._finalize()
        
        return self.metrics
        
    def _update_target_behaviors(self):
        """Update all target aircraft behaviors"""
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        
        for target_id in self.target_ids:
            if target_id not in self.asset_manager.assets:
                continue  # Target destroyed/removed
                
            target_state = self.asset_manager.get_asset_state(target_id)
            controller = self.target_controllers[target_id]
            
            # Create threat assessment for interceptor
            rel_state = self.asset_manager.get_relative_state(target_id, self.interceptor_id)
            if rel_state:
                threat = ThreatAssessment(
                    threat_id=self.interceptor_id,
                    position=interceptor_state.position,
                    velocity=interceptor_state.get_velocity_vector(),
                    range=rel_state['range'],
                    closing_velocity=rel_state['closing_velocity'],
                    time_to_intercept=rel_state['time_to_intercept'],
                    threat_level=0.8
                )
                threats = [threat]
            else:
                threats = []
                
            # Get terrain height
            terrain_height = self.battlespace.get_elevation(
                target_state.position[0], 
                target_state.position[1]
            )
            
            # Compute behavior
            commands = controller.compute_behavior(
                target_state, 
                threats, 
                terrain_height,
                self.asset_manager.time
            )
            
            # Apply commands
            self.asset_manager.apply_commands(target_id, 
                                             commands.bank_angle, 
                                             commands.throttle)
            
    def _get_sensor_detections(self) -> List[TargetTrack]:
        """Get sensor detections for interceptor"""
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        
        # Get true target states
        true_targets = {}
        for target_id in self.target_ids:
            if target_id in self.asset_manager.assets:
                true_targets[target_id] = self.asset_manager.get_asset_state(target_id)
                
        # Generate detections with sensor model
        detections = self.sensor_model.detect_targets(
            interceptor_state,
            true_targets,
            self.battlespace,
            self.asset_manager.time
        )
        
        return detections
        
    def _run_guidance_algorithm(self, detections: List[TargetTrack]):
        """Run user-provided guidance algorithm"""
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        
        # Package inputs for guidance algorithm
        guidance_input = {
            'own_state': interceptor_state,
            'sensor_tracks': detections,
            'battlespace': self.battlespace,
            'time': self.asset_manager.time
        }
        
        # Call guidance algorithm
        guidance_output = self.guidance_algorithm(guidance_input)
        
        # Apply guidance commands
        if 'bank_angle' in guidance_output and 'throttle' in guidance_output:
            self.asset_manager.apply_commands(
                self.interceptor_id,
                guidance_output['bank_angle'],
                guidance_output['throttle']
            )
            
    def _default_interceptor_behavior(self, detections: List[TargetTrack]):
        """Default interceptor behavior for testing"""
        if not detections:
            return
            
        # Pick closest detection
        closest_track = min(detections, key=lambda t: t.range)
        
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        aircraft = self.asset_manager.assets[self.interceptor_id].aircraft
        
        # Simple proportional navigation
        los_vector = closest_track.position - interceptor_state.position
        desired_heading = np.arctan2(los_vector[1], los_vector[0])
        
        heading_error = desired_heading - interceptor_state.heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
            
        bank_cmd = np.clip(heading_error * 1.5, -1.0, 1.0)
        
        # Throttle based on range
        if closest_track.range > 2000:
            throttle_cmd = 0.9
        elif closest_track.range > 500:
            throttle_cmd = 0.7
        else:
            throttle_cmd = 0.5
            
        self.asset_manager.apply_commands(self.interceptor_id, bank_cmd, throttle_cmd)
        
    def _evaluate_objectives(self):
        """Evaluate current objective status"""
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        
        for objective in self.objectives:
            if objective.completed or objective.failed:
                continue
                
            if objective.objective_type == ObjectiveType.INTERCEPT:
                # Check intercept range
                target_id = objective.details.get('target_id')
                req_range = objective.details.get('range', 100)
                
                if target_id and target_id in self.asset_manager.assets:
                    rel_state = self.asset_manager.get_relative_state(
                        self.interceptor_id, target_id
                    )
                    if rel_state and rel_state['range'] < req_range:
                        objective.completed = True
                        objective.progress = 1.0
                        
                        # Record intercept event
                        event = InterceptEvent(
                            time=self.asset_manager.time,
                            interceptor_id=self.interceptor_id,
                            target_id=target_id,
                            range=rel_state['range'],
                            closing_velocity=rel_state['closing_velocity'],
                            interceptor_fuel=interceptor_state.fuel_remaining,
                            success=True
                        )
                        self.metrics.intercept_events.append(event)
                        
                        if self.verbose:
                            print(f"\n*** OBJECTIVE COMPLETE: Intercepted {target_id} at {rel_state['range']:.1f}m ***")
                    else:
                        # Update progress
                        if rel_state:
                            objective.progress = max(0, 1.0 - rel_state['range'] / 10000)
                            
            elif objective.objective_type == ObjectiveType.TIME_LIMIT:
                time_limit = objective.details['time_limit']
                if self.asset_manager.time > time_limit:
                    objective.failed = True
                else:
                    objective.progress = self.asset_manager.time / time_limit
                    
            elif objective.objective_type == ObjectiveType.FUEL_REMAINING:
                min_fraction = objective.details['min_fraction']
                fuel_fraction = interceptor_state.fuel_remaining / self.asset_manager.assets[self.interceptor_id].aircraft.fuel_capacity
                
                objective.progress = fuel_fraction / min_fraction
                if fuel_fraction < min_fraction:
                    objective.failed = True
                    
    def _check_termination(self) -> bool:
        """Check if simulation should terminate"""
        # Check termination conditions
        interceptor = self.asset_manager.assets.get(self.interceptor_id)
        
        if not interceptor:
            self.state = ScenarioState.FAILED
            return True
            
        # Check specific conditions
        for condition in self.termination_conditions:
            if isinstance(condition, dict):
                if 'max_time' in condition:
                    if self.asset_manager.time >= condition['max_time']:
                        if self.verbose:
                            print("\nTerminating: Max time reached")
                        return True
                        
                if 'all_targets_neutralized' in condition:
                    all_neutralized = all(
                        tid not in self.asset_manager.assets or
                        self.asset_manager.assets[tid].aircraft.mode == FlightMode.CRASHED
                        for tid in self.target_ids
                    )
                    if all_neutralized:
                        if self.verbose:
                            print("\nTerminating: All targets neutralized")
                        return True
                        
                if 'interceptor_crashed' in condition:
                    if interceptor.aircraft.mode == FlightMode.CRASHED:
                        if self.verbose:
                            print("\nTerminating: Interceptor crashed")
                        self.state = ScenarioState.FAILED
                        return True
                        
                if 'interceptor_fuel_empty' in condition:
                    if interceptor.aircraft.state.fuel_remaining <= 0:
                        if self.verbose:
                            print("\nTerminating: Interceptor out of fuel")
                        return True
                        
        # Check if all primary objectives are complete or failed
        primary_done = all(
            obj.completed or obj.failed 
            for obj in self.objectives 
            if obj.required
        )
        
        if primary_done:
            all_complete = all(obj.completed for obj in self.objectives if obj.required)
            if all_complete:
                self.state = ScenarioState.COMPLETED
                if self.verbose:
                    print("\nTerminating: All primary objectives complete!")
            else:
                self.state = ScenarioState.FAILED
                if self.verbose:
                    print("\nTerminating: Primary objectives failed")
            return True
            
        return False
        
    def _record_frame(self):
        """Record current frame data"""
        frame_data = {
            'time': self.asset_manager.time,
            'interceptor': {
                'id': self.interceptor_id,
                'position': self.asset_manager.get_asset_state(self.interceptor_id).position.tolist(),
                'velocity': self.asset_manager.get_asset_state(self.interceptor_id).velocity,
                'fuel': self.asset_manager.get_asset_state(self.interceptor_id).fuel_remaining
            },
            'targets': {},
            'objectives': [
                {
                    'type': obj.objective_type.value,
                    'progress': obj.progress,
                    'completed': obj.completed,
                    'failed': obj.failed
                }
                for obj in self.objectives
            ]
        }
        
        for target_id in self.target_ids:
            if target_id in self.asset_manager.assets:
                state = self.asset_manager.get_asset_state(target_id)
                frame_data['targets'][target_id] = {
                    'position': state.position.tolist(),
                    'velocity': state.velocity
                }
                
        self.recording_data.append(frame_data)
        
    def _print_status(self):
        """Print current simulation status"""
        interceptor_state = self.asset_manager.get_asset_state(self.interceptor_id)
        
        # Find closest target
        closest_target = None
        min_range = float('inf')
        
        for target_id in self.target_ids:
            if target_id in self.asset_manager.assets:
                rel_state = self.asset_manager.get_relative_state(
                    self.interceptor_id, target_id
                )
                if rel_state and rel_state['range'] < min_range:
                    min_range = rel_state['range']
                    closest_target = target_id
                    
        # Get objectives status
        obj_complete = sum(1 for obj in self.objectives if obj.completed)
        obj_total = len(self.objectives)
        
        print(f"{self.asset_manager.time:5.1f} | "
              f"({interceptor_state.position[0]:6.0f}, {interceptor_state.position[1]:6.0f}, {interceptor_state.position[2]:4.0f}) | "
              f"{closest_target if closest_target else 'None':8s} | "
              f"{min_range:6.1f} | "
              f"Obj: {obj_complete}/{obj_total}")
              
    def _finalize(self):
        """Finalize simulation and compute final metrics"""
        self.metrics.end_time = time.time()
        
        # Compute final metrics
        if self.metrics.update_times:
            self.metrics.mean_update_time = np.mean(self.metrics.update_times)
            self.metrics.max_update_time = np.max(self.metrics.update_times)
            
        # Count objective results
        self.metrics.objectives_completed = sum(1 for obj in self.objectives if obj.completed)
        self.metrics.objectives_failed = sum(1 for obj in self.objectives if obj.failed)
        
        total_objectives = len([obj for obj in self.objectives if obj.required])
        if total_objectives > 0:
            self.metrics.success_rate = self.metrics.objectives_completed / total_objectives
            
        # Save recording if enabled
        if self.recording_enabled and self.recording_data:
            self._save_recording()
            
        if self.verbose:
            self._print_final_report()
            
    def _save_recording(self):
        """Save recorded data to file"""
        output_path = Path(self.recording_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        recording = {
            'scenario': self.config.get('name', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'duration': self.asset_manager.time,
                'steps': self.metrics.total_steps,
                'intercepts': len(self.metrics.intercept_events),
                'success_rate': self.metrics.success_rate
            },
            'frames': self.recording_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(recording, f, indent=2)
            
        if self.verbose:
            print(f"\nRecording saved to: {output_path}")
            
    def _print_final_report(self):
        """Print final simulation report"""
        print("\n" + "="*60)
        print("SCENARIO COMPLETE")
        print("="*60)
        
        print(f"\nScenario: {self.config.get('name', 'Unknown')}")
        print(f"Status: {self.state.value}")
        print(f"Duration: {self.asset_manager.time:.1f}s (real: {self.metrics.end_time - self.metrics.start_time:.1f}s)")
        print(f"Steps: {self.metrics.total_steps}")
        
        print("\nObjectives:")
        for obj in self.objectives:
            status = "✓" if obj.completed else "✗" if obj.failed else "○"
            req = " (Required)" if obj.required else ""
            print(f"  {status} {obj.description}{req} - Progress: {obj.progress*100:.1f}%")
            
        print(f"\nIntercept Events: {len(self.metrics.intercept_events)}")
        for event in self.metrics.intercept_events:
            print(f"  - {event.target_id} at t={event.time:.1f}s, range={event.range:.1f}m")
            
        print("\nPerformance:")
        if self.metrics.update_times:
            print(f"  Mean update time: {self.metrics.mean_update_time*1000:.2f}ms")
            print(f"  Max update time: {self.metrics.max_update_time*1000:.2f}ms")
            print(f"  Update rate: {1.0/self.metrics.mean_update_time:.1f}Hz")
            
        print(f"\nFinal Score: {self.metrics.success_rate*100:.1f}%")
        
    def get_results(self) -> Dict[str, Any]:
        """
        Get scenario results as dictionary.
        
        Returns:
            Dictionary with results and metrics
        """
        return {
            'scenario': self.config.get('name'),
            'state': self.state.value,
            'duration': self.asset_manager.time,
            'objectives': [
                {
                    'description': obj.description,
                    'required': obj.required,
                    'completed': obj.completed,
                    'failed': obj.failed,
                    'progress': obj.progress
                }
                for obj in self.objectives
            ],
            'metrics': {
                'total_steps': self.metrics.total_steps,
                'intercept_count': len(self.metrics.intercept_events),
                'success_rate': self.metrics.success_rate,
                'mean_update_time_ms': self.metrics.mean_update_time * 1000,
                'objectives_completed': self.metrics.objectives_completed,
                'objectives_failed': self.metrics.objectives_failed
            },
            'intercept_events': [
                {
                    'time': event.time,
                    'target': event.target_id,
                    'range': event.range,
                    'fuel_remaining': event.interceptor_fuel
                }
                for event in self.metrics.intercept_events
            ]
        }