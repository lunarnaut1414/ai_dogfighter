# src/guidance_core/state_machine.py
"""
Hierarchical state machine for interceptor guidance phases.
Manages transitions between mission phases based on tactical situation.
"""

import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time


class MissionPhase(Enum):
    """Mission phase states for the interceptor"""
    STARTUP = auto()      # System initialization
    SEARCH = auto()       # Area coverage patterns
    TRACK = auto()        # Multi-target tracking
    INTERCEPT = auto()    # Active engagement
    EVADE = auto()        # Defensive maneuvers
    RTB = auto()          # Return to base
    EMERGENCY = auto()    # Failsafe modes
    LOITER = auto()       # Hold pattern


class InterceptSubPhase(Enum):
    """Sub-phases for intercept engagement"""
    PURSUIT = auto()      # Long range (>500m)
    TERMINAL = auto()     # Mid range (100-500m)
    ENGAGE = auto()       # Close range (<100m)
    BREAKAWAY = auto()    # Post-intercept maneuver
    REATTACK = auto()     # Re-engagement if needed


class TransitionPriority(Enum):
    """Transition priority levels"""
    SAFETY_CRITICAL = 1   # Immediate transition
    MISSION_CRITICAL = 2  # <100ms transition
    TACTICAL = 3          # <500ms transition
    STRATEGIC = 4         # <1000ms transition
    OPTIMIZATION = 5      # Best effort


@dataclass
class StateTransition:
    """Defines a state transition"""
    from_state: MissionPhase
    to_state: MissionPhase
    condition: callable  # Function that returns bool
    priority: TransitionPriority
    callback: Optional[callable] = None


@dataclass
class MissionContext:
    """Context information for state decisions"""
    own_state: Dict[str, Any]  # Position, velocity, fuel, etc.
    targets: List[Dict[str, Any]]  # Detected targets
    threats: List[Dict[str, Any]]  # Active threats
    mission_params: Dict[str, Any]  # ROE, objectives, etc.
    environment: Dict[str, Any]  # Weather, terrain, etc.
    time_in_state: float
    total_mission_time: float
    fuel_remaining: float
    ammunition_remaining: int
    health_status: Dict[str, Any]


class GuidanceStateMachine:
    """
    Hierarchical state machine for autonomous interceptor guidance.
    Manages mission phases and tactical state transitions.
    """
    
    def __init__(self, initial_state: MissionPhase = MissionPhase.STARTUP):
        """
        Initialize the state machine.
        
        Args:
            initial_state: Starting mission phase
        """
        self.current_state = initial_state
        self.previous_state = None
        self.intercept_subphase = None
        self.state_entry_time = time.time()
        self.state_history = [(self.state_entry_time, initial_state)]
        
        # Transition rules
        self.transitions = []
        self._define_transitions()
        
        # State callbacks
        self.on_enter_callbacks = {}
        self.on_exit_callbacks = {}
        self.on_update_callbacks = {}
        self._register_callbacks()
        
        # Mission parameters
        self.mission_params = {
            'search_pattern': 'expanding_square',
            'engagement_range_max': 2000.0,
            'engagement_range_min': 50.0,
            'fuel_rtb_threshold': 0.2,  # 20% fuel remaining
            'fuel_emergency_threshold': 0.1,  # 10% fuel
            'max_targets': 5,
            'reattack_enabled': True,
            'evasion_enabled': True
        }
        
        # Performance metrics
        self.transition_count = 0
        self.phase_durations = {phase: [] for phase in MissionPhase}
        
    def _define_transitions(self):
        """Define all possible state transitions and their conditions"""
        
        # STARTUP transitions
        self.add_transition(
            MissionPhase.STARTUP, MissionPhase.SEARCH,
            self._startup_complete, TransitionPriority.MISSION_CRITICAL
        )
        self.add_transition(
            MissionPhase.STARTUP, MissionPhase.EMERGENCY,
            self._critical_failure, TransitionPriority.SAFETY_CRITICAL
        )
        
        # SEARCH transitions
        self.add_transition(
            MissionPhase.SEARCH, MissionPhase.TRACK,
            self._targets_detected, TransitionPriority.TACTICAL
        )
        self.add_transition(
            MissionPhase.SEARCH, MissionPhase.RTB,
            self._fuel_low, TransitionPriority.MISSION_CRITICAL
        )
        self.add_transition(
            MissionPhase.SEARCH, MissionPhase.EMERGENCY,
            self._critical_failure, TransitionPriority.SAFETY_CRITICAL
        )
        
        # TRACK transitions
        self.add_transition(
            MissionPhase.TRACK, MissionPhase.INTERCEPT,
            self._intercept_authorized, TransitionPriority.TACTICAL
        )
        self.add_transition(
            MissionPhase.TRACK, MissionPhase.SEARCH,
            self._targets_lost, TransitionPriority.TACTICAL
        )
        self.add_transition(
            MissionPhase.TRACK, MissionPhase.EVADE,
            self._threat_detected, TransitionPriority.MISSION_CRITICAL
        )
        self.add_transition(
            MissionPhase.TRACK, MissionPhase.RTB,
            self._fuel_low, TransitionPriority.MISSION_CRITICAL
        )
        
        # INTERCEPT transitions
        self.add_transition(
            MissionPhase.INTERCEPT, MissionPhase.TRACK,
            self._intercept_complete, TransitionPriority.TACTICAL
        )
        self.add_transition(
            MissionPhase.INTERCEPT, MissionPhase.EVADE,
            self._threat_critical, TransitionPriority.SAFETY_CRITICAL
        )
        self.add_transition(
            MissionPhase.INTERCEPT, MissionPhase.RTB,
            self._fuel_critical, TransitionPriority.MISSION_CRITICAL
        )
        self.add_transition(
            MissionPhase.INTERCEPT, MissionPhase.SEARCH,
            self._target_destroyed_and_clear, TransitionPriority.STRATEGIC
        )
        
        # EVADE transitions
        self.add_transition(
            MissionPhase.EVADE, MissionPhase.TRACK,
            self._threat_evaded, TransitionPriority.TACTICAL
        )
        self.add_transition(
            MissionPhase.EVADE, MissionPhase.RTB,
            self._evasion_complete_and_fuel_low, TransitionPriority.MISSION_CRITICAL
        )
        self.add_transition(
            MissionPhase.EVADE, MissionPhase.EMERGENCY,
            self._damage_critical, TransitionPriority.SAFETY_CRITICAL
        )
        
        # RTB transitions
        self.add_transition(
            MissionPhase.RTB, MissionPhase.LOITER,
            self._arrived_at_base, TransitionPriority.STRATEGIC
        )
        self.add_transition(
            MissionPhase.RTB, MissionPhase.EMERGENCY,
            self._fuel_exhausted, TransitionPriority.SAFETY_CRITICAL
        )
        
        # EMERGENCY transitions (limited options)
        self.add_transition(
            MissionPhase.EMERGENCY, MissionPhase.RTB,
            self._emergency_recovered, TransitionPriority.MISSION_CRITICAL
        )
        
        # LOITER transitions
        self.add_transition(
            MissionPhase.LOITER, MissionPhase.SEARCH,
            self._new_mission_assigned, TransitionPriority.STRATEGIC
        )
        
    def add_transition(self, from_state: MissionPhase, to_state: MissionPhase,
                      condition: callable, priority: TransitionPriority,
                      callback: Optional[callable] = None):
        """Add a state transition rule"""
        transition = StateTransition(from_state, to_state, condition, priority, callback)
        self.transitions.append(transition)
        
    def _register_callbacks(self):
        """Register state-specific callbacks"""
        # On enter callbacks
        self.on_enter_callbacks[MissionPhase.SEARCH] = self._enter_search
        self.on_enter_callbacks[MissionPhase.TRACK] = self._enter_track
        self.on_enter_callbacks[MissionPhase.INTERCEPT] = self._enter_intercept
        self.on_enter_callbacks[MissionPhase.EVADE] = self._enter_evade
        self.on_enter_callbacks[MissionPhase.RTB] = self._enter_rtb
        self.on_enter_callbacks[MissionPhase.EMERGENCY] = self._enter_emergency
        
        # On update callbacks (called each cycle)
        self.on_update_callbacks[MissionPhase.INTERCEPT] = self._update_intercept_subphase
        
    def update(self, context: MissionContext) -> Tuple[MissionPhase, Optional[Dict]]:
        """
        Update state machine and check for transitions.
        
        Args:
            context: Current mission context
            
        Returns:
            Tuple of (current_state, transition_info or None)
        """
        context.time_in_state = time.time() - self.state_entry_time
        
        # Check transitions in priority order
        valid_transitions = [t for t in self.transitions 
                           if t.from_state == self.current_state]
        valid_transitions.sort(key=lambda t: t.priority.value)
        
        for transition in valid_transitions:
            if transition.condition(context):
                # Execute transition
                transition_info = self._execute_transition(transition, context)
                return self.current_state, transition_info
                
        # No transition, update current state
        if self.current_state in self.on_update_callbacks:
            self.on_update_callbacks[self.current_state](context)
            
        return self.current_state, None
        
    def _execute_transition(self, transition: StateTransition, 
                          context: MissionContext) -> Dict:
        """Execute a state transition"""
        # Record state duration
        duration = time.time() - self.state_entry_time
        self.phase_durations[self.current_state].append(duration)
        
        # Exit callbacks
        if self.current_state in self.on_exit_callbacks:
            self.on_exit_callbacks[self.current_state](context)
            
        # Update state
        self.previous_state = self.current_state
        self.current_state = transition.to_state
        self.state_entry_time = time.time()
        self.state_history.append((self.state_entry_time, self.current_state))
        self.transition_count += 1
        
        # Enter callbacks
        if self.current_state in self.on_enter_callbacks:
            self.on_enter_callbacks[self.current_state](context)
            
        # Transition callback
        if transition.callback:
            transition.callback(context)
            
        return {
            'from': self.previous_state,
            'to': self.current_state,
            'priority': transition.priority,
            'time': self.state_entry_time,
            'reason': transition.condition.__name__
        }
        
    # Transition condition functions
    def _startup_complete(self, context: MissionContext) -> bool:
        """Check if startup procedures are complete"""
        return context.time_in_state > 2.0 and context.health_status.get('all_systems_go', False)
        
    def _critical_failure(self, context: MissionContext) -> bool:
        """Check for critical system failure"""
        return context.health_status.get('critical_failure', False)
        
    def _targets_detected(self, context: MissionContext) -> bool:
        """Check if targets have been detected"""
        return len(context.targets) > 0
        
    def _targets_lost(self, context: MissionContext) -> bool:
        """Check if all targets have been lost"""
        return len(context.targets) == 0 and context.time_in_state > 5.0
        
    def _intercept_authorized(self, context: MissionContext) -> bool:
        """Check if intercept is authorized"""
        if not context.targets:
            return False
        
        # Check closest target
        closest_target = min(context.targets, key=lambda t: t['range'])
        in_range = closest_target['range'] <= self.mission_params['engagement_range_max']
        authorized = context.mission_params.get('weapons_free', True)
        
        return in_range and authorized
        
    def _threat_detected(self, context: MissionContext) -> bool:
        """Check if an active threat requires evasion"""
        if not self.mission_params['evasion_enabled']:
            return False
        return len(context.threats) > 0 and any(t['threat_level'] > 0.7 for t in context.threats)
        
    def _threat_critical(self, context: MissionContext) -> bool:
        """Check for critical threat during intercept"""
        return len(context.threats) > 0 and any(t['threat_level'] > 0.9 for t in context.threats)
        
    def _fuel_low(self, context: MissionContext) -> bool:
        """Check if fuel is low enough to RTB"""
        return context.fuel_remaining < self.mission_params['fuel_rtb_threshold']
        
    def _fuel_critical(self, context: MissionContext) -> bool:
        """Check if fuel is critically low"""
        return context.fuel_remaining < self.mission_params['fuel_emergency_threshold']
        
    def _fuel_exhausted(self, context: MissionContext) -> bool:
        """Check if fuel is exhausted"""
        return context.fuel_remaining < 0.05
        
    def _intercept_complete(self, context: MissionContext) -> bool:
        """Check if intercept is complete"""
        if self.intercept_subphase == InterceptSubPhase.BREAKAWAY:
            return context.time_in_state > 3.0  # 3 seconds after breakaway
        return False
        
    def _target_destroyed_and_clear(self, context: MissionContext) -> bool:
        """Check if target destroyed and area clear"""
        return self._intercept_complete(context) and len(context.targets) == 0
        
    def _threat_evaded(self, context: MissionContext) -> bool:
        """Check if threats have been evaded"""
        return len(context.threats) == 0 or all(t['threat_level'] < 0.3 for t in context.threats)
        
    def _evasion_complete_and_fuel_low(self, context: MissionContext) -> bool:
        """Check if evasion complete and need to RTB"""
        return self._threat_evaded(context) and self._fuel_low(context)
        
    def _damage_critical(self, context: MissionContext) -> bool:
        """Check for critical damage"""
        return context.health_status.get('damage_level', 0) > 0.7
        
    def _arrived_at_base(self, context: MissionContext) -> bool:
        """Check if arrived at base"""
        if 'base_position' not in context.mission_params:
            return False
        
        own_pos = np.array(context.own_state['position'])
        base_pos = np.array(context.mission_params['base_position'])
        distance = np.linalg.norm(own_pos - base_pos)
        
        return distance < 500.0  # Within 500m of base
        
    def _emergency_recovered(self, context: MissionContext) -> bool:
        """Check if recovered from emergency"""
        return not self._critical_failure(context) and context.time_in_state > 10.0
        
    def _new_mission_assigned(self, context: MissionContext) -> bool:
        """Check for new mission assignment"""
        return context.mission_params.get('new_mission_ready', False)
        
    # State enter callbacks
    def _enter_search(self, context: MissionContext):
        """Initialize search pattern"""
        self.intercept_subphase = None
        print(f"[STATE] Entering SEARCH mode - Pattern: {self.mission_params['search_pattern']}")
        
    def _enter_track(self, context: MissionContext):
        """Initialize tracking"""
        self.intercept_subphase = None
        print(f"[STATE] Entering TRACK mode - Targets: {len(context.targets)}")
        
    def _enter_intercept(self, context: MissionContext):
        """Initialize intercept engagement"""
        self.intercept_subphase = InterceptSubPhase.PURSUIT
        if context.targets:
            closest = min(context.targets, key=lambda t: t['range'])
            print(f"[STATE] Entering INTERCEPT mode - Target at {closest['range']:.0f}m")
        
    def _enter_evade(self, context: MissionContext):
        """Initialize evasive maneuvers"""
        self.intercept_subphase = None
        print(f"[STATE] Entering EVADE mode - Threats: {len(context.threats)}")
        
    def _enter_rtb(self, context: MissionContext):
        """Initialize return to base"""
        self.intercept_subphase = None
        print(f"[STATE] Entering RTB mode - Fuel: {context.fuel_remaining:.1%}")
        
    def _enter_emergency(self, context: MissionContext):
        """Initialize emergency procedures"""
        self.intercept_subphase = None
        print("[STATE] Entering EMERGENCY mode!")
        
    def _update_intercept_subphase(self, context: MissionContext):
        """Update intercept sub-phase based on range"""
        if not context.targets:
            return
            
        closest = min(context.targets, key=lambda t: t['range'])
        range_to_target = closest['range']
        
        # Transition through intercept sub-phases
        if self.intercept_subphase == InterceptSubPhase.PURSUIT:
            if range_to_target < 500:
                self.intercept_subphase = InterceptSubPhase.TERMINAL
                print(f"[INTERCEPT] Entering TERMINAL phase - Range: {range_to_target:.0f}m")
                
        elif self.intercept_subphase == InterceptSubPhase.TERMINAL:
            if range_to_target < 100:
                self.intercept_subphase = InterceptSubPhase.ENGAGE
                print(f"[INTERCEPT] Entering ENGAGE phase - Range: {range_to_target:.0f}m")
                
        elif self.intercept_subphase == InterceptSubPhase.ENGAGE:
            if range_to_target < 20 or closest.get('intercepted', False):
                self.intercept_subphase = InterceptSubPhase.BREAKAWAY
                print(f"[INTERCEPT] INTERCEPT! Entering BREAKAWAY phase")
                
        elif self.intercept_subphase == InterceptSubPhase.BREAKAWAY:
            if context.time_in_state > 2.0 and self.mission_params['reattack_enabled']:
                if len(context.targets) > 1:  # More targets available
                    self.intercept_subphase = InterceptSubPhase.REATTACK
                    print(f"[INTERCEPT] Entering REATTACK phase")
                    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'current_state': self.current_state,
            'previous_state': self.previous_state,
            'intercept_subphase': self.intercept_subphase,
            'time_in_state': time.time() - self.state_entry_time,
            'transition_count': self.transition_count,
            'state_history': self.state_history[-10:]  # Last 10 states
        }
        
    def get_guidance_mode(self) -> str:
        """Get recommended guidance mode based on current state"""
        mode_map = {
            MissionPhase.STARTUP: 'hold',
            MissionPhase.SEARCH: 'search_pattern',
            MissionPhase.TRACK: 'track',
            MissionPhase.INTERCEPT: self._get_intercept_mode(),
            MissionPhase.EVADE: 'evasive',
            MissionPhase.RTB: 'waypoint',
            MissionPhase.EMERGENCY: 'emergency_descent',
            MissionPhase.LOITER: 'orbit'
        }
        return mode_map.get(self.current_state, 'hold')
        
    def _get_intercept_mode(self) -> str:
        """Get specific intercept guidance mode"""
        if self.intercept_subphase == InterceptSubPhase.PURSUIT:
            return 'proportional_navigation'
        elif self.intercept_subphase == InterceptSubPhase.TERMINAL:
            return 'augmented_pn'
        elif self.intercept_subphase == InterceptSubPhase.ENGAGE:
            return 'optimal_guidance'
        elif self.intercept_subphase == InterceptSubPhase.BREAKAWAY:
            return 'breakaway'
        elif self.intercept_subphase == InterceptSubPhase.REATTACK:
            return 'proportional_navigation'
        return 'proportional_navigation'