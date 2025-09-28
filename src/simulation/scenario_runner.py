"""
ScenarioRunner: Main execution engine for scenario-based simulations.
Integrates asset manager, guidance algorithms, and objective tracking.
"""

import yaml
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback

from src.battlespace import Battlespace
from src.assets.asset_manager import AssetManager, AssetType
from src.assets.flight_controller import FlightController, BehaviorMode
from src.assets.sensor_model import SensorModel
from src.simulation.objective_manager import ObjectiveManager


class ScenarioState(Enum):
    """States of scenario execution"""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class ScenarioMetrics:
    """Performance and outcome metrics for a scenario run"""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    total_steps: int = 0
    
    # Performance metrics
    mean_update_time_ms: float = 0.0
    max_update_time_ms: float = 0.0
    min_update_time_ms: float = float('inf')
    
    # Outcome metrics
    intercepts: int = 0
    collisions: int = 0
    fuel_consumed: Dict[str, float] = field(default_factory=dict)
    distance_traveled: Dict[str, float] = field(default_factory=dict)
    
    # Guidance metrics
    guidance_mode_changes: int = 0
    target_switches: int = 0
    
    def update_timing(self, step_time_ms: float):
        """Update timing statistics"""
        self.mean_update_time_ms = (
            (self.mean_update_time_ms * self.total_steps + step_time_ms) / 
            (self.total_steps + 1)
        )
        self.max_update_time_ms = max(self.max_update_time_ms, step_time_ms)
        self.min_update_time_ms = min(self.min_update_time_ms, step_time_ms)


class ScenarioRunner:
    """
    Main scenario execution engine.
    Coordinates all simulation components and manages execution flow.
    """
    
    def __init__(self,
                 scenario_config: str,
                 guidance_algorithm: Optional[Any] = None,
                 realtime: bool = False,
                 verbose: bool = True,
                 record: bool = False):
        """
        Initialize scenario runner.
        
        Args:
            scenario_config: Path to scenario YAML file or config dict
            guidance_algorithm: Guidance algorithm instance
            realtime: Whether to run in realtime mode
            verbose: Enable verbose output
            record: Enable data recording
        """
        # Load configuration
        if isinstance(scenario_config, str):
            with open(scenario_config, 'r') as f:
                self.config = yaml.safe_load(f)['scenario']
        else:
            self.config = scenario_config['scenario']
            
        self.guidance_algorithm = guidance_algorithm
        self.realtime = realtime
        self.verbose = verbose
        self.record = record
        
        # Core components (initialized in setup)
        self.battlespace: Optional[Battlespace] = None
        self.asset_manager: Optional[AssetManager] = None
        self.objective_manager: Optional[ObjectiveManager] = None
        self.sensors: Dict[str, SensorModel] = {}
        self.controllers: Dict[str, FlightController] = {}
        
        # State tracking
        self.state = ScenarioState.UNINITIALIZED
        self.time = 0.0
        self.dt = 1.0 / self.config.get('update_rate', 50)
        self.max_time = self.config.get('termination', {}).get('time_limit', 600)
        
        # Metrics and recording
        self.metrics = ScenarioMetrics()
        self.events: List[Dict[str, Any]] = []
        self.recording_data: List[Dict[str, Any]] = [] if record else None
        
        # Asset tracking
        self.interceptor_id: Optional[str] = None
        self.target_ids: List[str] = []
        self.all_asset_ids: List[str] = []
        
    def setup(self):
        """Initialize all simulation components"""
        try:
            if self.verbose:
                print("Setting up scenario...")
                
            # Create battlespace
            battlespace_config = self.config.get('battlespace', 
                                                'configs/battlespace/default_battlespace.yaml')
            self.battlespace = Battlespace(config_file=battlespace_config)
            self.battlespace.generate(seed=self.config.get('seed', 42))
            
            # Create asset manager
            self.asset_manager = AssetManager(self.battlespace, dt=self.dt)
            
            # Spawn interceptor
            self._spawn_interceptor()
            
            # Spawn targets
            self._spawn_targets()
            
            # Initialize sensors
            self._initialize_sensors()
            
            # Initialize controllers
            self._initialize_controllers()
            
            # Setup objectives
            self._setup_objectives()
            
            # Initialize metrics
            self.metrics.start_time = time.time()
            
            self.state = ScenarioState.READY
            
            if self.verbose:
                print(f"Scenario setup complete:")
                print(f"  - Interceptor: {self.interceptor_id}")
                print(f"  - Targets: {len(self.target_ids)}")
                print(f"  - Objectives: {len(self.objective_manager.objectives)}")
                print(f"  - Update rate: {1/self.dt:.1f} Hz")
                print(f"  - Max time: {self.max_time}s")
                
        except Exception as e:
            self.state = ScenarioState.FAILED
            print(f"ERROR: Scenario setup failed: {e}")
            traceback.print_exc()
            raise
            
    def _spawn_interceptor(self):
        """Spawn the interceptor aircraft"""
        interceptor_config = self.config['interceptor']
        
        self.interceptor_id = interceptor_config.get('id', 'interceptor_1')
        
        self.asset_manager.spawn_aircraft(
            config=interceptor_config,
            asset_id=self.interceptor_id,
            asset_type=AssetType.INTERCEPTOR,
            team=interceptor_config.get('team', 'blue')
        )
        
        self.all_asset_ids.append(self.interceptor_id)
        
    def _spawn_targets(self):
        """Spawn all target aircraft"""
        targets = self.config.get('targets', [])
        
        for target_config in targets:
            target_id = target_config['id']
            
            self.asset_manager.spawn_aircraft(
                config=target_config,
                asset_id=target_id,
                asset_type=AssetType.TARGET,
                team=target_config.get('team', 'red')
            )
            
            # Set behavior if specified
            if 'behavior' in target_config:
                self._configure_target_behavior(target_id, target_config['behavior'])
                
            self.target_ids.append(target_id)
            self.all_asset_ids.append(target_id)
            
    def _configure_target_behavior(self, target_id: str, behavior_config: Any):
        """Configure target aircraft behavior"""
        asset_info = self.asset_manager.assets[target_id]
        
        if isinstance(behavior_config, str):
            # Simple behavior mode
            asset_info.behavior_mode = behavior_config
        elif isinstance(behavior_config, dict):
            # Complex behavior with parameters
            asset_info.behavior_mode = behavior_config.get('type', 'waypoint')
            
            if 'waypoints' in behavior_config:
                asset_info.waypoints = [
                    np.array(wp) for wp in behavior_config['waypoints']
                ]
                
    def _initialize_sensors(self):
        """Initialize sensor models for all aircraft"""
        # Interceptor sensor
        interceptor = self.asset_manager.assets[self.interceptor_id].aircraft
        if hasattr(interceptor, 'sensors'):
            sensor_config = interceptor.config.get('sensors', {})
            self.sensors[self.interceptor_id] = SensorModel(sensor_config)
            
        # Target sensors (if needed)
        for target_id in self.target_ids:
            target = self.asset_manager.assets[target_id].aircraft
            if hasattr(target, 'sensors'):
                sensor_config = target.config.get('sensors', {})
                self.sensors[target_id] = SensorModel(sensor_config)
                
    def _initialize_controllers(self):
        """Initialize flight controllers for all aircraft"""
        # Controllers for targets (interceptor uses guidance)
        for target_id in self.target_ids:
            asset_info = self.asset_manager.assets[target_id]
            aircraft = asset_info.aircraft
            
            controller = FlightController(aircraft.config)
            
            # Configure based on behavior
            if asset_info.behavior_mode == 'waypoint' and asset_info.waypoints:
                controller.set_mode(BehaviorMode.WAYPOINT)
                controller.set_waypoints(asset_info.waypoints)
            elif asset_info.behavior_mode == 'orbit':
                controller.set_mode(BehaviorMode.ORBIT)
                # Set orbit center from waypoints or default
                if asset_info.waypoints:
                    controller.set_orbit_center(asset_info.waypoints[0])
                    
            self.controllers[target_id] = controller
            
    def _setup_objectives(self):
        """Setup scenario objectives"""
        from src.simulation.objective_manager import ObjectiveManager
        
        objectives_config = self.config.get('objectives', {})
        self.objective_manager = ObjectiveManager(objectives_config)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the scenario to completion.
        
        Returns:
            Dictionary with results and metrics
        """
        if self.state != ScenarioState.READY:
            raise RuntimeError(f"Cannot run scenario in state {self.state}")
            
        self.state = ScenarioState.RUNNING
        
        if self.verbose:
            print(f"\nStarting scenario execution...")
            
        try:
            while self.state == ScenarioState.RUNNING:
                # Step simulation
                self.step()
                
                # Check termination conditions
                if self._check_termination():
                    break
                    
                # Realtime synchronization
                if self.realtime:
                    time.sleep(self.dt)
                    
            # Finalize
            self.metrics.end_time = time.time()
            self.metrics.duration = self.time
            
            # Save recording if enabled
            if self.record and self.recording_data:
                self._save_recording()
                
            return self._compile_results()
            
        except Exception as e:
            self.state = ScenarioState.FAILED
            print(f"ERROR: Scenario execution failed: {e}")
            traceback.print_exc()
            raise
            
    def step(self):
        """Execute one simulation step"""
        step_start = time.time()
        
        # 1. Get sensor detections
        detections = self._update_sensors()
        
        # 2. Update guidance for interceptor
        if self.guidance_algorithm:
            self._update_guidance(detections)
            
        # 3. Update controllers for targets
        self._update_controllers()
        
        # 4. Step physics
        self.asset_manager.update()
        
        # 5. Check for events
        self._check_events()
        
        # 6. Update objectives
        if self.objective_manager:
            self.objective_manager.update(self.asset_manager, self.time)
            
        # 7. Record data
        if self.record:
            self._record_frame()
            
        # Update metrics
        step_time_ms = (time.time() - step_start) * 1000
        self.metrics.update_timing(step_time_ms)
        self.metrics.total_steps += 1
        
        # Update time
        self.time += self.dt
        
        # Verbose output
        if self.verbose and self.metrics.total_steps % 50 == 0:
            self._print_status()
            
    def _update_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Update all sensors and get detections"""
        detections = {}
        
        # Interceptor sensor detections
        if self.interceptor_id in self.sensors:
            sensor = self.sensors[self.interceptor_id]
            interceptor = self.asset_manager.assets[self.interceptor_id].aircraft
            
            # Get all potential targets
            all_assets = self.asset_manager.get_all_assets()
            targets = {
                aid: ainfo for aid, ainfo in all_assets.items()
                if aid != self.interceptor_id
            }
            
            # Get detections
            interceptor_detections = sensor.get_detections(
                interceptor.state,
                targets,
                self.battlespace
            )
            
            detections[self.interceptor_id] = interceptor_detections
            
        return detections
        
    def _update_guidance(self, detections: Dict[str, Dict[str, Any]]):
        """Update guidance algorithm and apply commands"""
        interceptor_state = self.asset_manager.get_asset(self.interceptor_id)
        
        # Get detected targets for interceptor
        interceptor_detections = detections.get(self.interceptor_id, {})
        
        # Convert detections to target states
        target_states = {}
        for detection in interceptor_detections.get('detections', []):
            target_id = detection['id']
            target_states[target_id] = {
                'position': detection['measured_position'],
                'velocity': detection.get('measured_velocity', np.zeros(3)),
                'heading': detection.get('heading', 0),
            }
            
        # Compute guidance commands
        commands = self.guidance_algorithm.compute_commands(
            interceptor_state['state'],
            target_states,
            self.battlespace
        )
        
        # Apply commands to interceptor
        self.asset_manager.set_commands(self.interceptor_id, commands)
        
    def _update_controllers(self):
        """Update flight controllers for all targets"""
        for target_id, controller in self.controllers.items():
            if target_id in self.asset_manager.assets:
                aircraft = self.asset_manager.assets[target_id].aircraft
                
                # Get control commands
                commands = controller.compute_commands(aircraft.state)
                
                # Apply to aircraft
                aircraft.set_controls(
                    bank_angle=commands.bank_angle,
                    throttle=commands.throttle
                )
                
    def _check_events(self):
        """Check for and record significant events"""
        # Check for intercepts
        interceptor_pos = self.asset_manager.assets[self.interceptor_id].aircraft.state.position
        
        for target_id in self.target_ids:
            if target_id not in self.asset_manager.assets:
                continue
                
            target = self.asset_manager.assets[target_id].aircraft
            range_to_target = np.linalg.norm(target.state.position - interceptor_pos)
            
            # Check intercept criteria
            if range_to_target < self.config.get('intercept_range', 50):
                event = {
                    'type': 'intercept',
                    'time': self.time,
                    'interceptor_id': self.interceptor_id,
                    'target_id': target_id,
                    'range': range_to_target,
                    'interceptor_fuel': self.asset_manager.assets[self.interceptor_id].aircraft.state.fuel_remaining
                }
                self.events.append(event)
                self.metrics.intercepts += 1
                
                # Remove target after intercept
                self.asset_manager.remove_asset(target_id)
                self.target_ids.remove(target_id)
                
                if self.verbose:
                    print(f"  INTERCEPT: {target_id} at {range_to_target:.1f}m")
                    
    def _check_termination(self) -> bool:
        """Check termination conditions"""
        # Time limit
        if self.time >= self.max_time:
            self.state = ScenarioState.COMPLETED
            if self.verbose:
                print("Time limit reached")
            return True
            
        # All targets neutralized
        if not self.target_ids:
            self.state = ScenarioState.COMPLETED
            if self.verbose:
                print("All targets neutralized")
            return True
            
        # Interceptor fuel depleted
        interceptor = self.asset_manager.assets.get(self.interceptor_id)
        if interceptor and interceptor.aircraft.state.fuel_remaining <= 0:
            self.state = ScenarioState.COMPLETED
            if self.verbose:
                print("Interceptor fuel depleted")
            return True
            
        # Check objective completion
        if self.objective_manager and self.objective_manager.all_primary_complete():
            self.state = ScenarioState.COMPLETED
            if self.verbose:
                print("All primary objectives completed")
            return True
            
        return False
        
    def _record_frame(self):
        """Record current frame data"""
        frame_data = {
            'time': self.time,
            'assets': {},
            'events': list(self.events),  # Copy current events
            'objectives': self.objective_manager.get_status() if self.objective_manager else {}
        }
        
        # Record all asset states
        for asset_id in self.all_asset_ids:
            if asset_id in self.asset_manager.assets:
                asset = self.asset_manager.assets[asset_id]
                frame_data['assets'][asset_id] = {
                    'position': asset.aircraft.state.position.tolist(),
                    'velocity': asset.aircraft.state.velocity,
                    'heading': asset.aircraft.state.heading,
                    'altitude': asset.aircraft.state.position[2],
                    'fuel': asset.aircraft.state.fuel_remaining,
                }
                
        self.recording_data.append(frame_data)
        
    def _save_recording(self):
        """Save recorded data to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data/scenarios/recording_{timestamp}.json"
        
        Path("data/scenarios").mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                'config': self.config,
                'metrics': self.metrics.__dict__,
                'frames': self.recording_data
            }, f, indent=2)
            
        if self.verbose:
            print(f"Recording saved to {filename}")
            
    def _print_status(self):
        """Print current simulation status"""
        interceptor = self.asset_manager.assets[self.interceptor_id].aircraft
        
        print(f"T={self.time:6.1f}s | "
              f"Alt={interceptor.state.position[2]:5.0f}m | "
              f"V={interceptor.state.velocity:4.1f}m/s | "
              f"Fuel={interceptor.state.fuel_remaining/interceptor.fuel_capacity:4.1%} | "
              f"Targets={len(self.target_ids)} | "
              f"FPS={1000/self.metrics.mean_update_time_ms:4.1f}")
              
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results"""
        return {
            'state': self.state.value,
            'duration': self.time,
            'metrics': {
                'total_steps': self.metrics.total_steps,
                'mean_update_time_ms': self.metrics.mean_update_time_ms,
                'max_update_time_ms': self.metrics.max_update_time_ms,
                'intercepts': self.metrics.intercepts,
                'collisions': self.metrics.collisions,
            },
            'events': self.events,
            'objectives': self.objective_manager.get_summary() if self.objective_manager else {},
            'recording_file': None  # Set if recording was saved
        }
        
    def pause(self):
        """Pause scenario execution"""
        if self.state == ScenarioState.RUNNING:
            self.state = ScenarioState.PAUSED
            
    def resume(self):
        """Resume scenario execution"""
        if self.state == ScenarioState.PAUSED:
            self.state = ScenarioState.RUNNING
            
    def terminate(self):
        """Terminate scenario execution"""
        self.state = ScenarioState.TERMINATED