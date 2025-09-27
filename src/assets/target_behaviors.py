"""
Progressive target behavior system for increasing difficulty.
Implements levels 0-4 of target complexity.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from enum import IntEnum
from dataclasses import dataclass

from src.assets.aircraft_3dof import AircraftState
from src.assets.flight_controller import FlightController, BehaviorMode, ControlCommand


class TargetDifficulty(IntEnum):
    """Target difficulty levels"""
    STATIC = 0          # Stationary or simple orbit
    PREDICTABLE = 1     # Straight lines, waypoints
    REACTIVE = 2        # Simple evasion when threatened
    TACTICAL = 3        # Energy management, terrain use
    INTELLIGENT = 4     # Predictive, optimal tactics


@dataclass
class ThreatAssessment:
    """Assessment of threat state"""
    threat_id: str
    position: np.ndarray
    velocity: np.ndarray
    range: float
    closing_velocity: float
    time_to_intercept: float
    threat_level: float  # 0-1 based on capability and intent


class TargetBehaviorController:
    """
    Advanced target behavior controller with progressive difficulty levels.
    """
    
    def __init__(self, aircraft_config: Dict[str, Any], 
                 difficulty: TargetDifficulty = TargetDifficulty.PREDICTABLE):
        """
        Initialize target behavior controller.
        
        Args:
            aircraft_config: Aircraft configuration
            difficulty: Target difficulty level
        """
        self.base_controller = FlightController(aircraft_config)
        self.difficulty = difficulty
        self.config = aircraft_config
        
        # Behavior parameters by difficulty
        self.setup_difficulty_parameters()
        
        # State memory for intelligent behaviors
        self.threat_history: List[ThreatAssessment] = []
        self.last_evasion_time = 0.0
        self.energy_state = 1.0  # Normalized energy (altitude + speed)
        
        # Tactical parameters
        self.home_base = None
        self.terrain_aware = False
        self.last_threat_position = None
        
    def setup_difficulty_parameters(self):
        """Set parameters based on difficulty level"""
        
        if self.difficulty == TargetDifficulty.STATIC:
            self.reaction_time = float('inf')  # Never reacts
            self.evasion_skill = 0.0
            self.prediction_horizon = 0.0
            self.energy_awareness = 0.0
            
        elif self.difficulty == TargetDifficulty.PREDICTABLE:
            self.reaction_time = float('inf')  # Doesn't react to threats
            self.evasion_skill = 0.0
            self.prediction_horizon = 0.0
            self.energy_awareness = 0.2
            
        elif self.difficulty == TargetDifficulty.REACTIVE:
            self.reaction_time = 2.0  # 2 second reaction delay
            self.evasion_skill = 0.5
            self.prediction_horizon = 1.0  # Look ahead 1 second
            self.energy_awareness = 0.5
            
        elif self.difficulty == TargetDifficulty.TACTICAL:
            self.reaction_time = 0.5  # Quick reactions
            self.evasion_skill = 0.8
            self.prediction_horizon = 3.0  # Look ahead 3 seconds
            self.energy_awareness = 0.8
            self.terrain_aware = True
            
        elif self.difficulty == TargetDifficulty.INTELLIGENT:
            self.reaction_time = 0.1  # Near instant
            self.evasion_skill = 1.0
            self.prediction_horizon = 5.0  # Long-term planning
            self.energy_awareness = 1.0
            self.terrain_aware = True
            
    def compute_behavior(self, state: AircraftState, 
                        threats: Optional[List[ThreatAssessment]] = None,
                        terrain_height: Optional[float] = None,
                        current_time: float = 0.0) -> ControlCommand:
        """
        Compute control commands based on difficulty level.
        
        Args:
            state: Current aircraft state
            threats: List of threat assessments
            terrain_height: Terrain height at current position
            current_time: Current simulation time
            
        Returns:
            Control commands
        """
        
        # Level 0: Static behavior
        if self.difficulty == TargetDifficulty.STATIC:
            return self.static_behavior(state)
            
        # Level 1: Predictable behavior
        elif self.difficulty == TargetDifficulty.PREDICTABLE:
            return self.predictable_behavior(state)
            
        # Level 2: Reactive behavior
        elif self.difficulty == TargetDifficulty.REACTIVE:
            return self.reactive_behavior(state, threats, current_time)
            
        # Level 3: Tactical behavior
        elif self.difficulty == TargetDifficulty.TACTICAL:
            return self.tactical_behavior(state, threats, terrain_height, current_time)
            
        # Level 4: Intelligent behavior
        elif self.difficulty == TargetDifficulty.INTELLIGENT:
            return self.intelligent_behavior(state, threats, terrain_height, current_time)
            
        return ControlCommand(0.0, state.throttle)
        
    def static_behavior(self, state: AircraftState) -> ControlCommand:
        """
        Level 0: Static or simple orbit.
        """
        # Simple orbit around current position
        if not hasattr(self, 'orbit_center_static'):
            self.orbit_center_static = state.position.copy()
            self.orbit_center_static[2] = state.position[2]  # Maintain altitude
            
        self.base_controller.set_mode(BehaviorMode.ORBIT)
        self.base_controller.set_orbit(self.orbit_center_static, 500.0)
        
        return self.base_controller.compute_commands(state)
        
    def predictable_behavior(self, state: AircraftState) -> ControlCommand:
        """
        Level 1: Predictable waypoint following.
        """
        # Follow preset waypoints
        if not hasattr(self, 'waypoints_set'):
            # Create simple waypoint pattern
            center = state.position[:2]
            self.base_controller.set_waypoints([
                np.array([center[0] + 2000, center[1], state.position[2]]),
                np.array([center[0] + 2000, center[1] + 2000, state.position[2]]),
                np.array([center[0] - 2000, center[1] + 2000, state.position[2]]),
                np.array([center[0] - 2000, center[1] - 2000, state.position[2]]),
                np.array([center[0] + 2000, center[1] - 2000, state.position[2]]),
            ])
            self.waypoints_set = True
            self.base_controller.set_mode(BehaviorMode.WAYPOINT)
            
        return self.base_controller.compute_commands(state)
        
    def reactive_behavior(self, state: AircraftState,
                         threats: Optional[List[ThreatAssessment]],
                         current_time: float) -> ControlCommand:
        """
        Level 2: React to threats with simple evasion.
        """
        if not threats:
            # No threat, continue patrol
            return self.predictable_behavior(state)
            
        # Find closest threat
        closest_threat = min(threats, key=lambda t: t.range)
        
        # Check if we should react (with reaction delay)
        if closest_threat.range > 3000:  # Outside reaction range
            return self.predictable_behavior(state)
            
        if current_time - self.last_evasion_time < self.reaction_time:
            # Still in reaction delay
            return self.predictable_behavior(state)
            
        # Simple evasion - turn away from threat
        self.last_evasion_time = current_time
        self.base_controller.set_mode(BehaviorMode.EVADE)
        self.base_controller.evasion_aggressiveness = self.evasion_skill
        
        return self.base_controller.compute_commands(
            state, threat=closest_threat.position
        )
        
    def tactical_behavior(self, state: AircraftState,
                         threats: Optional[List[ThreatAssessment]],
                         terrain_height: Optional[float],
                         current_time: float) -> ControlCommand:
        """
        Level 3: Tactical behavior with energy management.
        """
        # Update energy state
        self.update_energy_state(state)
        
        if not threats:
            # Patrol at optimal altitude for energy
            return self.energy_efficient_patrol(state)
            
        # Assess threats
        primary_threat = self.assess_primary_threat(threats)
        
        if primary_threat.range > 4000:
            # Build energy while safe
            return self.build_energy(state)
            
        # Tactical evasion based on energy state
        if self.energy_state > 0.7:
            # High energy - aggressive maneuvers
            return self.high_energy_evasion(state, primary_threat)
        elif self.energy_state > 0.3:
            # Medium energy - balanced tactics
            return self.balanced_evasion(state, primary_threat, terrain_height)
        else:
            # Low energy - defensive/escape
            return self.low_energy_escape(state, primary_threat, terrain_height)
            
    def intelligent_behavior(self, state: AircraftState,
                           threats: Optional[List[ThreatAssessment]],
                           terrain_height: Optional[float],
                           current_time: float) -> ControlCommand:
        """
        Level 4: Intelligent predictive behavior.
        """
        # Update threat history for pattern recognition
        if threats:
            self.threat_history.extend(threats)
            if len(self.threat_history) > 100:
                self.threat_history = self.threat_history[-100:]
                
        # Predict future threat positions
        future_threats = self.predict_threat_positions(threats, self.prediction_horizon)
        
        # Calculate optimal escape/engagement strategy
        strategy = self.compute_optimal_strategy(state, future_threats, terrain_height)
        
        # Execute strategy
        if strategy == 'energy_trap':
            return self.execute_energy_trap(state, future_threats[0])
        elif strategy == 'terrain_mask':
            return self.execute_terrain_masking(state, terrain_height)
        elif strategy == 'spiral_climb':
            return self.execute_spiral_climb(state)
        elif strategy == 'dive_escape':
            return self.execute_dive_escape(state)
        else:
            # Default to tactical behavior
            return self.tactical_behavior(state, threats, terrain_height, current_time)
            
    # Helper methods for tactical behaviors
    
    def update_energy_state(self, state: AircraftState):
        """Update normalized energy state"""
        # Energy = altitude + speed^2/(2g)
        specific_energy = state.position[2] + (state.velocity**2) / (2 * 9.81)
        max_energy = 5000 + (80**2) / (2 * 9.81)  # Approximate max
        self.energy_state = np.clip(specific_energy / max_energy, 0, 1)
        
    def assess_primary_threat(self, threats: List[ThreatAssessment]) -> ThreatAssessment:
        """Identify primary threat based on multiple factors"""
        if not threats:
            return None
            
        # Score each threat
        threat_scores = []
        for threat in threats:
            score = 0.0
            
            # Range factor (closer = higher threat)
            score += (5000 - threat.range) / 5000 if threat.range < 5000 else 0
            
            # Closing velocity factor
            if threat.closing_velocity > 0:
                score += threat.closing_velocity / 100
                
            # Time to intercept factor
            if threat.time_to_intercept < 30:
                score += (30 - threat.time_to_intercept) / 30
                
            threat_scores.append(score)
            
        # Return highest scoring threat
        max_idx = np.argmax(threat_scores)
        return threats[max_idx]
        
    def build_energy(self, state: AircraftState) -> ControlCommand:
        """Build energy by climbing"""
        # Gentle climb to gain altitude
        target_altitude = min(state.position[2] + 500, 5000)
        target_pos = state.position.copy()
        target_pos[2] = target_altitude
        
        self.base_controller.set_mode(BehaviorMode.WAYPOINT)
        self.base_controller.set_waypoints([target_pos])
        
        return self.base_controller.compute_commands(state)
        
    def energy_efficient_patrol(self, state: AircraftState) -> ControlCommand:
        """Patrol at best L/D speed and altitude"""
        # Maintain optimal cruise speed and altitude
        optimal_altitude = 3000  # Typical best altitude
        
        if abs(state.position[2] - optimal_altitude) > 100:
            # Adjust altitude
            target_pos = state.position.copy()
            target_pos[0] += 1000  # Move forward while climbing
            target_pos[2] = optimal_altitude
            self.base_controller.set_waypoints([target_pos])
            
        return self.base_controller.compute_commands(state)
        
    def high_energy_evasion(self, state: AircraftState,
                           threat: ThreatAssessment) -> ControlCommand:
        """Aggressive evasion with energy to spare"""
        # Immelmann turn or split-S based on threat position
        threat_below = threat.position[2] < state.position[2]
        
        if threat_below:
            # Immelmann - pull up and roll
            commands = ControlCommand(
                bank_angle=self.base_controller.max_bank * 0.9,
                throttle=1.0
            )
        else:
            # Split-S - roll and pull down
            commands = ControlCommand(
                bank_angle=self.base_controller.max_bank,
                throttle=0.3
            )
            
        return commands
        
    def balanced_evasion(self, state: AircraftState,
                        threat: ThreatAssessment,
                        terrain_height: Optional[float]) -> ControlCommand:
        """Balanced tactical evasion"""
        # Barrel roll evasion
        self.base_controller.set_mode(BehaviorMode.EVADE)
        self.base_controller.evasion_aggressiveness = 0.7
        
        # Use terrain if available
        if terrain_height and state.position[2] - terrain_height < 500:
            # Close to terrain, use it for masking
            commands = self.base_controller.compute_commands(state, threat=threat.position)
            commands.throttle = 0.8  # Maintain energy
        else:
            commands = self.base_controller.compute_commands(state, threat=threat.position)
            
        return commands
        
    def low_energy_escape(self, state: AircraftState,
                         threat: ThreatAssessment,
                         terrain_height: Optional[float]) -> ControlCommand:
        """Low energy defensive escape"""
        # Dive to gain speed and extend
        escape_vector = state.position - threat.position
        escape_vector[2] = -abs(escape_vector[2])  # Force downward
        
        escape_pos = state.position + escape_vector * 2
        
        # Don't hit terrain
        if terrain_height:
            escape_pos[2] = max(escape_pos[2], terrain_height + 100)
            
        self.base_controller.set_waypoints([escape_pos])
        return self.base_controller.compute_commands(state)
        
    def predict_threat_positions(self, threats: Optional[List[ThreatAssessment]], 
                                horizon: float) -> List[ThreatAssessment]:
        """Predict future threat positions"""
        if not threats:
            return []
            
        future_threats = []
        for threat in threats:
            future_pos = threat.position + threat.velocity * horizon
            future_threat = ThreatAssessment(
                threat_id=threat.threat_id,
                position=future_pos,
                velocity=threat.velocity,
                range=np.linalg.norm(future_pos),
                closing_velocity=threat.closing_velocity,
                time_to_intercept=max(0, threat.time_to_intercept - horizon),
                threat_level=threat.threat_level
            )
            future_threats.append(future_threat)
            
        return future_threats
        
    def compute_optimal_strategy(self, state: AircraftState,
                                future_threats: List[ThreatAssessment],
                                terrain_height: Optional[float]) -> str:
        """Compute optimal strategy based on predictions"""
        if not future_threats:
            return 'patrol'
            
        primary_threat = future_threats[0]
        
        # Decision tree based on situation
        if self.energy_state > 0.8 and primary_threat.range < 2000:
            return 'energy_trap'
        elif terrain_height and state.position[2] - terrain_height < 1000:
            return 'terrain_mask'
        elif self.energy_state < 0.3:
            return 'dive_escape'
        elif primary_threat.position[2] < state.position[2]:
            return 'spiral_climb'
        else:
            return 'tactical'
            
    def execute_energy_trap(self, state: AircraftState,
                           threat: ThreatAssessment) -> ControlCommand:
        """Lure threat into energy-depleting maneuver"""
        # Climb steeply to force threat to follow
        commands = ControlCommand(
            bank_angle=0.0,
            throttle=1.0
        )
        # Pull up hard
        return commands
        
    def execute_terrain_masking(self, state: AircraftState,
                               terrain_height: float) -> ControlCommand:
        """Use terrain for concealment"""
        # Fly low following terrain
        target_agl = 200  # 200m above ground
        target_altitude = terrain_height + target_agl
        
        # Adjust altitude while maintaining speed
        altitude_error = target_altitude - state.position[2]
        
        commands = ControlCommand(
            bank_angle=0.0,  # Straight for now
            throttle=0.7 + altitude_error * 0.001
        )
        
        return commands
        
    def execute_spiral_climb(self, state: AircraftState) -> ControlCommand:
        """Climbing spiral to gain altitude advantage"""
        commands = ControlCommand(
            bank_angle=self.base_controller.max_bank * 0.7,
            throttle=1.0
        )
        return commands
        
    def execute_dive_escape(self, state: AircraftState) -> ControlCommand:
        """Diving escape to gain speed"""
        commands = ControlCommand(
            bank_angle=0.0,
            throttle=0.0  # Idle for maximum dive speed
        )
        return commands