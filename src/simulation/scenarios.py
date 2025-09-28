# src/simulation/scenarios.py
"""
Scenario system for configuration-driven simulation setup.
Handles scenario loading, execution, and success evaluation.
"""

import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import json

from ..battlespace.core import Battlespace
from ..assets.asset_manager import AssetManager, AssetType
from ..assets.flight_controller import FlightController, BehaviorMode
from ..simulation.environment import SimulationEnvironment


class MissionPhase(Enum):
    """Mission phases for scenario progression"""
    SETUP = "setup"
    LAUNCH = "launch"
    SEARCH = "search"
    TRACK = "track"
    APPROACH = "approach"
    TERMINAL = "terminal"
    COMPLETE = "complete"
    FAILED = "failed"


class ThreatLevel(Enum):
    """Target threat classification"""
    HOSTILE = "hostile"
    UNKNOWN = "unknown"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"


@dataclass
class ScenarioObjective:
    """Defines a scenario objective/success criterion"""
    name: str
    type: str  # 'intercept', 'defend_zone', 'survive_time', 'reach_waypoint'
    required: bool = True
    
    # Type-specific parameters
    target_id: Optional[str] = None
    position: Optional[np.ndarray] = None
    radius: Optional[float] = None
    time_limit: Optional[float] = None
    min_distance: Optional[float] = None
    
    # Status tracking
    completed: bool = False
    completion_time: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class TargetBehaviorConfig:
    """Configuration for target behavior state machine"""
    initial_state: str
    states: Dict[str, Dict[str, Any]]
    transitions: Dict[str, Dict[str, str]]
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TargetBehaviorConfig':
        """Create from configuration dictionary"""
        return cls(
            initial_state=config.get('initial_state', 'patrol'),
            states=config.get('states', {}),
            transitions=config.get('transitions', {})
        )


@dataclass 
class ScenarioMetrics:
    """Metrics collected during scenario execution"""
    start_time: float
    end_time: Optional[float] = None
    total_steps: int = 0
    
    # Engagement metrics
    intercepts_attempted: int = 0
    intercepts_successful: int = 0
    min_intercept_distance: float = float('inf')
    mean_intercept_time: float = 0.0
    
    # Resource metrics
    fuel_consumed: float = 0.0
    distance_traveled: float = 0.0
    max_g_pulled: float = 0.0
    time_in_violation: float = 0.0  # Time violating constraints
    
    # Target metrics
    targets_tracked: int = 0
    targets_lost: int = 0
    false_tracks: int = 0
    
    # Performance metrics
    mean_step_time_ms: float = 0.0
    max_step_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'duration': self.end_time - self.start_time if self.end_time else None,
            'total_steps': self.total_steps,
            'intercepts': {
                'attempted': self.intercepts_attempted,
                'successful': self.intercepts_successful,
                'success_rate': self.intercepts_successful / max(1, self.intercepts_attempted),
                'min_distance': self.min_intercept_distance,
                'mean_time': self.mean_intercept_time
            },
            'resources': {
                'fuel_consumed': self.fuel_consumed,
                'distance_traveled': self.distance_traveled,
                'max_g': self.max_g_pulled
            },
            'tracking': {
                'targets_tracked': self.targets_tracked,
                'targets_lost': self.targets_lost,
                'false_tracks': self.false_tracks
            },
            'performance': {
                'mean_step_time_ms': self.mean_step_time_ms,
                'max_step_time_ms': self.max_step_time_ms
            }
        }


class TargetBehaviorStateMachine:
    """
    State machine for target aircraft behaviors.
    Implements various evasion and movement patterns.
    """
    
    def __init__(self, config: TargetBehaviorConfig):
        """Initialize behavior state machine"""
        self.config = config
        self.current_state = config.initial_state
        self.state_start_time = 0.0
        self.state_history = []
        
    def update(self, 
               target_state: Dict[str, Any],
               interceptor_state: Optional[Dict[str, Any]], 
               time: float) -> Dict[str, Any]:
        """
        Update state machine and return control commands.
        
        Args:
            target_state: Current target aircraft state
            interceptor_state: Interceptor state if being tracked
            time: Current simulation time
            
        Returns:
            Control commands dictionary
        """
        # Get current state configuration
        state_config = self.config.states.get(self.current_state, {})
        state_type = state_config.get('type', 'waypoint')
        
        # Check for state transitions
        self._check_transitions(target_state, interceptor_state, time)
        
        # Execute current state behavior
        if state_type == 'waypoint':
            return self._waypoint_behavior(state_config, target_state)
        elif state_type == 'evasive':
            return self._evasive_behavior(state_config, target_state, interceptor_state)
        elif state_type == 'orbit':
            return self._orbit_behavior(state_config, target_state)
        elif state_type == 'flee':
            return self._flee_behavior(state_config, target_state, interceptor_state)
        elif state_type == 'aggressive':
            return self._aggressive_behavior(state_config, target_state, interceptor_state)
        else:
            return {'mode': 'maintain', 'throttle': 0.7}
            
    def _check_transitions(self,
                          target_state: Dict[str, Any],
                          interceptor_state: Optional[Dict[str, Any]],
                          time: float):
        """Check and execute state transitions"""
        transitions = self.config.transitions.get(self.current_state, {})
        
        for condition, next_state in transitions.items():
            if self._evaluate_condition(condition, target_state, interceptor_state, time):
                self._transition_to(next_state, time)
                break
                
    def _evaluate_condition(self,
                           condition: str,
                           target_state: Dict[str, Any],
                           interceptor_state: Optional[Dict[str, Any]],
                           time: float) -> bool:
        """Evaluate transition condition"""
        if condition == 'threat_detected':
            if interceptor_state is None:
                return False
            range_to_interceptor = np.linalg.norm(
                target_state['position'] - interceptor_state['position']
            )
            return range_to_interceptor < 5000  # Detection range
            
        elif condition == 'threat_clear':
            if interceptor_state is None:
                return True
            range_to_interceptor = np.linalg.norm(
                target_state['position'] - interceptor_state['position']
            )
            return range_to_interceptor > 8000
            
        elif condition == 'fuel_low':
            return target_state.get('fuel_fraction', 1.0) < 0.3
            
        elif condition == 'fuel_critical':
            return target_state.get('fuel_fraction', 1.0) < 0.1
            
        elif condition == 'time_exceeded':
            return (time - self.state_start_time) > 30.0
            
        elif condition == 'waypoint_reached':
            return target_state.get('waypoint_reached', False)
            
        return False
        
    def _transition_to(self, next_state: str, time: float):
        """Execute state transition"""
        self.state_history.append({
            'state': self.current_state,
            'duration': time - self.state_start_time,
            'exit_time': time
        })
        self.current_state = next_state
        self.state_start_time = time
        
    def _waypoint_behavior(self, config: Dict[str, Any], 
                          target_state: Dict[str, Any]) -> Dict[str, Any]:
        """Standard waypoint following"""
        return {
            'mode': 'waypoint',
            'waypoints': config.get('waypoints', []),
            'speed': config.get('speed', 40.0),
            'throttle': config.get('throttle', 0.7)
        }
        
    def _evasive_behavior(self, config: Dict[str, Any],
                         target_state: Dict[str, Any],
                         interceptor_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evasive maneuver behavior"""
        if interceptor_state is None:
            return {'mode': 'maintain', 'throttle': 0.7}
            
        # Calculate escape vector
        threat_vector = interceptor_state['position'] - target_state['position']
        threat_bearing = np.arctan2(threat_vector[1], threat_vector[0])
        
        maneuver = config.get('maneuver', 'break_turn')
        
        if maneuver == 'break_turn':
            # Hard turn away from threat
            escape_heading = threat_bearing + np.pi + np.pi/4
            return {
                'mode': 'heading',
                'heading': escape_heading,
                'bank_angle': np.radians(60),  # Max bank
                'throttle': 1.0,  # Max power
                'g_limit': config.get('g_limit', 4.0)
            }
            
        elif maneuver == 'barrel_roll':
            # Barrel roll evasion (simplified as sinusoidal heading changes)
            phase = (time.time() % 4.0) / 4.0
            heading_offset = np.sin(phase * 2 * np.pi) * np.pi/3
            return {
                'mode': 'heading',
                'heading': threat_bearing + np.pi + heading_offset,
                'bank_angle': np.radians(45),
                'throttle': 0.9
            }
            
        elif maneuver == 'split_s':
            # Dive and reverse
            return {
                'mode': 'dive_reverse',
                'target_altitude': target_state['position'][2] - 500,
                'reverse_heading': threat_bearing + np.pi,
                'throttle': 0.6
            }
            
    def _orbit_behavior(self, config: Dict[str, Any],
                       target_state: Dict[str, Any]) -> Dict[str, Any]:
        """Orbit around a point"""
        center = np.array(config.get('center', [25000, 25000, 2000]))
        radius = config.get('radius', 5000)
        
        # Calculate tangent heading for orbit
        to_center = center[:2] - target_state['position'][:2]
        bearing_to_center = np.arctan2(to_center[1], to_center[0])
        orbit_heading = bearing_to_center + np.pi/2  # Tangent to circle
        
        return {
            'mode': 'orbit',
            'center': center,
            'radius': radius,
            'heading': orbit_heading,
            'speed': config.get('speed', 40.0),
            'throttle': config.get('throttle', 0.7)
        }
        
    def _flee_behavior(self, config: Dict[str, Any],
                      target_state: Dict[str, Any],
                      interceptor_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Flee directly away from threat"""
        if interceptor_state is None:
            return {'mode': 'maintain', 'throttle': 0.7}
            
        # Calculate escape vector
        escape_vector = target_state['position'] - interceptor_state['position']
        escape_heading = np.arctan2(escape_vector[1], escape_vector[0])
        
        return {
            'mode': 'flee',
            'heading': escape_heading,
            'throttle': 1.0,  # Maximum speed
            'altitude': config.get('altitude', target_state['position'][2])
        }
        
    def _aggressive_behavior(self, config: Dict[str, Any],
                           target_state: Dict[str, Any],
                           interceptor_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggressive counter-maneuver"""
        if interceptor_state is None:
            return {'mode': 'maintain', 'throttle': 0.7}
            
        # Head-on merge tactics
        to_threat = interceptor_state['position'] - target_state['position']
        bearing_to_threat = np.arctan2(to_threat[1], to_threat[0])
        
        return {
            'mode': 'aggressive',
            'heading': bearing_to_threat,  # Head toward interceptor
            'throttle': 1.0,
            'altitude': interceptor_state['position'][2],  # Match altitude
            'break_distance': config.get('break_distance', 1000)  # When to break off
        }


class ScenarioRunner:
    """
    Main scenario execution engine.
    Loads scenarios from configuration and manages execution.
    """
    
    def __init__(self, scenario_config: str):
        """
        Initialize scenario runner.
        
        Args:
            scenario_config: Path to scenario configuration file
        """
        self.config_path = Path(scenario_config)
        self.scenario_config = self._load_scenario_config()
        
        # Initialize components
        self.env = None
        self.objectives = []
        self.metrics = None
        self.behavior_machines = {}  # Target ID -> StateMachine
        
        # Mission phase tracking
        self.mission_phase = MissionPhase.SETUP
        self.phase_start_time = 0.0
        
    def _load_scenario_config(self) -> Dict[str, Any]:
        """Load scenario configuration from file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)['scenario']
            
    def setup(self):
        """Set up the scenario environment and assets"""
        print(f"\n{'='*60}")
        print(f"Setting up scenario: {self.scenario_config['name']}")
        print(f"{'='*60}")
        
        # Create environment
        self.env = SimulationEnvironment(
            battlespace_config=self.scenario_config.get('battlespace'),
            dt=1.0 / self.scenario_config.get('update_rate', 50),
            enable_physics=True,
            enable_sensors=True,
            enable_weather=self.scenario_config.get('enable_weather', True)
        )
        
        # Set weather conditions
        weather = self.scenario_config.get('weather_preset', 'clear')
        if weather == 'clear':
            self.env.battlespace.weather.set_clear()
        elif weather == 'cloudy':
            self.env.battlespace.weather.set_cloudy()
        elif weather == 'stormy':
            self.env.battlespace.weather.set_stormy()
            
        # Apply wind override if specified
        if 'wind_override' in self.scenario_config:
            wind = self.scenario_config['wind_override']
            self.env.battlespace.weather.wind.base_vector = np.array(wind['base_vector'])
            
        # Spawn interceptor
        self._spawn_interceptor()
        
        # Spawn targets
        self._spawn_targets()
        
        # Create objectives
        self._create_objectives()
        
        # Initialize metrics
        self.metrics = ScenarioMetrics(start_time=time.time())
        
        self.mission_phase = MissionPhase.LAUNCH
        
    def _spawn_interceptor(self):
        """Spawn the interceptor aircraft"""
        config = self.scenario_config['interceptor']
        initial = config['initial_state']
        
        self.interceptor_id = self.env.spawn_aircraft(
            aircraft_type='interceptor',
            position=initial['position'],
            heading=np.degrees(initial.get('heading', 0)),
            velocity=initial.get('velocity', 50),
            team='blue'
        )
        
        print(f"Spawned interceptor: {self.interceptor_id}")
        
    def _spawn_targets(self):
        """Spawn all target aircraft"""
        targets = self.scenario_config.get('targets', [])
        
        for target_config in targets:
            initial = target_config['initial_state']
            
            # Spawn target
            target_id = self.env.spawn_aircraft(
                aircraft_type='target',
                position=initial['position'],
                heading=np.degrees(initial.get('heading', 0)),
                velocity=initial.get('velocity', 40),
                team='red' if target_config.get('threat_level') == 'hostile' else 'neutral'
            )
            
            # Create behavior state machine if specified
            if 'behavior' in target_config:
                behavior_config = TargetBehaviorConfig.from_dict(target_config['behavior'])
                self.behavior_machines[target_id] = TargetBehaviorStateMachine(behavior_config)
                
            print(f"Spawned target: {target_id} ({target_config.get('threat_level', 'unknown')})")
            
    def _create_objectives(self):
        """Create scenario objectives from configuration"""
        objectives_config = self.scenario_config.get('objectives', [])
        
        for obj_config in objectives_config:
            objective = ScenarioObjective(
                name=obj_config['name'],
                type=obj_config['type'],
                required=obj_config.get('required', True),
                target_id=obj_config.get('target_id'),
                position=np.array(obj_config['position']) if 'position' in obj_config else None,
                radius=obj_config.get('radius'),
                time_limit=obj_config.get('time_limit'),
                min_distance=obj_config.get('min_distance')
            )
            self.objectives.append(objective)
            
        print(f"Created {len(self.objectives)} objectives")
        
    def run(self, max_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the scenario to completion.
        
        Args:
            max_duration: Maximum simulation time in seconds
            
        Returns:
            Results dictionary with metrics and outcomes
        """
        if self.env is None:
            self.setup()
            
        max_duration = max_duration or self.scenario_config.get('max_duration', 300.0)
        
        print(f"\nRunning scenario (max duration: {max_duration}s)")
        print("-" * 40)
        
        # Main simulation loop
        while self.env.time < max_duration:
            # Update environment
            step_result = self.env.step()
            
            # Update target behaviors
            self._update_target_behaviors()
            
            # Update guidance for interceptor (placeholder for your algorithm)
            self._update_interceptor_guidance()
            
            # Check objectives
            self._evaluate_objectives()
            
            # Update metrics
            self._update_metrics(step_result)
            
            # Update mission phase
            self._update_mission_phase()
            
            # Print periodic status
            if int(self.env.time) > int(self.env.time - self.env.dt) and int(self.env.time) % 5 == 0:
                self._print_status()
                
            # Check for completion
            if self._check_completion():
                break
                
        # Finalize metrics
        self.metrics.end_time = time.time()
        
        # Generate results
        return self._generate_results()
        
    def _update_target_behaviors(self):
        """Update all target behavior state machines"""
        interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
        interceptor_state = {
            'position': interceptor['state']['position'],
            'velocity': interceptor['state'].get('velocity', 0),
            'heading': interceptor['state'].get('heading', 0)
        } if interceptor else None
        
        for target_id, behavior_machine in self.behavior_machines.items():
            target = self.env.asset_manager.get_asset(target_id)
            if target:
                target_state = {
                    'position': target['state']['position'],
                    'velocity': target['state'].get('velocity', 0),
                    'heading': target['state'].get('heading', 0),
                    'fuel_fraction': target['state'].get('fuel_remaining', 1.0)
                }
                
                # Get behavior commands
                commands = behavior_machine.update(
                    target_state,
                    interceptor_state,
                    self.env.time
                )
                
                # Apply commands to aircraft controller
                if 'aircraft' in target:
                    aircraft = target['aircraft']
                    if commands['mode'] == 'waypoint':
                        aircraft.controller.set_mode(BehaviorMode.WAYPOINT)
                        aircraft.controller.set_waypoints(commands['waypoints'])
                    elif commands['mode'] == 'heading':
                        aircraft.controller.set_mode(BehaviorMode.HEADING_HOLD)
                        aircraft.controller.set_target_heading(commands['heading'])
                    # Add other command modes as needed
                    
    def _update_interceptor_guidance(self):
        """Update interceptor guidance (placeholder for your algorithm)"""
        # This is where your guidance algorithm will integrate
        # For now, using simple pursuit toward nearest target
        
        interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
        if not interceptor or 'aircraft' not in interceptor:
            return
            
        # Find nearest hostile target
        nearest_target = None
        min_distance = float('inf')
        
        for asset_id, asset in self.env.asset_manager.assets.items():
            if asset.asset_type == AssetType.TARGET and asset.team == 'red':
                distance = np.linalg.norm(
                    asset.aircraft.state.position - interceptor['state']['position']
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = asset
                    
        if nearest_target:
            # Simple pursuit guidance
            target_pos = nearest_target.aircraft.state.position
            interceptor['aircraft'].controller.set_mode(BehaviorMode.WAYPOINT)
            interceptor['aircraft'].controller.set_waypoints([target_pos])
            
            # Update mission phase based on distance
            if min_distance < 500:
                self.mission_phase = MissionPhase.TERMINAL
            elif min_distance < 2000:
                self.mission_phase = MissionPhase.APPROACH
            elif min_distance < 5000:
                self.mission_phase = MissionPhase.TRACK
                
    def _evaluate_objectives(self):
        """Check objective completion status"""
        for objective in self.objectives:
            if objective.completed:
                continue
                
            if objective.type == 'intercept':
                # Check if interceptor is close enough to target
                interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
                target = self.env.asset_manager.get_asset(objective.target_id)
                
                if interceptor and target:
                    distance = np.linalg.norm(
                        interceptor['state']['position'] - target['state']['position']
                    )
                    
                    if distance < (objective.min_distance or 100):
                        objective.completed = True
                        objective.completion_time = self.env.time
                        self.metrics.intercepts_successful += 1
                        print(f"✓ Objective completed: {objective.name}")
                        
            elif objective.type == 'defend_zone':
                # Check if targets are outside defense zone
                zone_clear = True
                for asset_id, asset in self.env.asset_manager.assets.items():
                    if asset.asset_type == AssetType.TARGET and asset.team == 'red':
                        distance = np.linalg.norm(
                            asset.aircraft.state.position - objective.position
                        )
                        if distance < objective.radius:
                            zone_clear = False
                            break
                            
                if zone_clear and self.env.time > 10.0:  # Give some time
                    objective.completed = True
                    objective.completion_time = self.env.time
                    
            elif objective.type == 'survive_time':
                if self.env.time >= objective.time_limit:
                    objective.completed = True
                    objective.completion_time = self.env.time
                    
    def _update_metrics(self, step_result: Dict[str, Any]):
        """Update scenario metrics"""
        self.metrics.total_steps += 1
        
        # Update performance metrics
        step_time_ms = step_result['step_time_ms']
        if step_time_ms > self.metrics.max_step_time_ms:
            self.metrics.max_step_time_ms = step_time_ms
            
        # Running average of step time
        n = self.metrics.total_steps
        self.metrics.mean_step_time_ms = (
            (self.metrics.mean_step_time_ms * (n-1) + step_time_ms) / n
        )
        
        # Track fuel consumption
        interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
        if interceptor and 'aircraft' in interceptor:
            fuel_used = 1.0 - interceptor['aircraft'].state.fuel_remaining
            self.metrics.fuel_consumed = fuel_used
            
    def _update_mission_phase(self):
        """Update current mission phase based on conditions"""
        # Phase transitions handled in guidance update
        pass
        
    def _print_status(self):
        """Print periodic status update"""
        interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
        if interceptor:
            pos = interceptor['state']['position']
            vel = interceptor['state'].get('velocity', 0)
            
            print(f"T={self.env.time:6.1f}s | "
                  f"Phase: {self.mission_phase.value:8s} | "
                  f"Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) | "
                  f"Vel: {vel:.1f}m/s")
                  
    def _check_completion(self) -> bool:
        """Check if scenario is complete"""
        # Check if all required objectives are complete
        required_complete = all(
            obj.completed for obj in self.objectives if obj.required
        )
        
        # Check for failure conditions
        interceptor = self.env.asset_manager.get_asset(self.interceptor_id)
        if not interceptor or interceptor.get('crashed', False):
            self.mission_phase = MissionPhase.FAILED
            return True
            
        if required_complete:
            self.mission_phase = MissionPhase.COMPLETE
            return True
            
        return False
        
    def _generate_results(self) -> Dict[str, Any]:
        """Generate final scenario results"""
        # Objective summary
        objectives_summary = []
        for obj in self.objectives:
            objectives_summary.append({
                'name': obj.name,
                'type': obj.type,
                'completed': obj.completed,
                'completion_time': obj.completion_time,
                'required': obj.required
            })
            
        # Success determination
        success = all(obj.completed for obj in self.objectives if obj.required)
        
        results = {
            'scenario': self.scenario_config['name'],
            'success': success,
            'final_phase': self.mission_phase.value,
            'duration': self.env.time,
            'objectives': objectives_summary,
            'metrics': self.metrics.to_dict()
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Scenario Complete: {self.scenario_config['name']}")
        print(f"{'='*60}")
        print(f"Success: {'✓' if success else '✗'}")
        print(f"Duration: {self.env.time:.1f}s")
        print(f"Final Phase: {self.mission_phase.value}")
        print(f"Objectives: {sum(1 for o in self.objectives if o.completed)}/{len(self.objectives)}")
        
        if self.metrics.intercepts_attempted > 0:
            print(f"Intercept Success Rate: {self.metrics.intercepts_successful}/{self.metrics.intercepts_attempted}")
            
        print(f"Mean Step Time: {self.metrics.mean_step_time_ms:.2f}ms")
        
        return results
        
    def save_results(self, filename: str):
        """Save scenario results to file"""
        if self.metrics and self.metrics.end_time:
            results = self._generate_results()
            
            output_path = Path('results') / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            print(f"\nResults saved to: {output_path}")


# Example usage function
def run_scenario_from_config(config_file: str) -> Dict[str, Any]:
    """
    Helper function to run a scenario from a configuration file.
    
    Args:
        config_file: Path to scenario configuration file
        
    Returns:
        Scenario results dictionary
    """
    runner = ScenarioRunner(config_file)
    runner.setup()
    results = runner.run()
    runner.save_results(f"scenario_{int(time.time())}.json")
    return results