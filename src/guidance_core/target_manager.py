# src/guidance_core/target_manager.py
"""
Target management system for multi-target tracking and prioritization.
Implements Multi-Criteria Decision Making (MCDM) for target selection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ThreatLevel(Enum):
    """Target threat level classification"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class TargetType(Enum):
    """Target type classification"""
    FIGHTER = "fighter"
    BOMBER = "bomber"
    MISSILE = "missile"
    DRONE = "drone"
    HELICOPTER = "helicopter"
    TRANSPORT = "transport"
    UNKNOWN = "unknown"


@dataclass
class TargetTrack:
    """Tracked target information"""
    id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_type: TargetType = TargetType.UNKNOWN
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    last_update_time: float = 0.0
    track_quality: float = 1.0  # 0-1, confidence in track
    time_since_detection: float = 0.0
    predicted_intercept_time: Optional[float] = None
    predicted_intercept_point: Optional[np.ndarray] = None
    engagement_history: List[Dict] = field(default_factory=list)
    priority_score: float = 0.0
    is_engaged: bool = False
    assigned_interceptor: Optional[str] = None


@dataclass
class EngagementZone:
    """Defines engagement envelope"""
    min_range: float = 50.0  # meters
    max_range: float = 5000.0  # meters
    min_altitude: float = 100.0  # meters
    max_altitude: float = 10000.0  # meters
    min_closing_speed: float = -10.0  # m/s (opening)
    max_closing_speed: float = 200.0  # m/s
    max_off_boresight_angle: float = 60.0  # degrees


class TargetManager:
    """
    Manages multiple target tracks and prioritization.
    Implements MCDM for optimal target selection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize target manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.tracks: Dict[str, TargetTrack] = {}
        self.engagement_zone = EngagementZone()
        self.current_time = 0.0
        
        # MCDM weights for target prioritization
        self.priority_weights = {
            'range': 0.25,  # Closer is higher priority
            'closing_speed': 0.20,  # Faster closing is higher priority
            'threat_level': 0.20,  # Higher threat is higher priority
            'angle_off': 0.15,  # Smaller angle is higher priority
            'altitude': 0.10,  # Lower altitude might be higher priority
            'track_quality': 0.10  # Better track is higher priority
        }
        
        # Track management parameters
        self.track_timeout = config.get('track_timeout', 5.0) if config else 5.0
        self.max_tracks = config.get('max_tracks', 50) if config else 50
        self.correlation_threshold = config.get('correlation_threshold', 50.0) if config else 50.0
        
        # Engagement rules
        self.max_simultaneous_engagements = config.get('max_engagements', 1) if config else 1
        self.reattack_enabled = config.get('reattack_enabled', True) if config else True
        self.min_reattack_delay = 3.0  # seconds
        
        # Statistics
        self.total_tracks_initiated = 0
        self.total_tracks_dropped = 0
        self.total_engagements = 0
        
    def update_tracks(self, sensor_contacts: List[Dict], own_state: Dict, 
                     current_time: float) -> List[TargetTrack]:
        """
        Update target tracks from sensor contacts.
        
        Args:
            sensor_contacts: List of sensor detections
            own_state: Own aircraft state
            current_time: Current simulation time
            
        Returns:
            List of active target tracks
        """
        self.current_time = current_time
        
        # Correlate new contacts with existing tracks
        for contact in sensor_contacts:
            self._process_contact(contact, current_time)
            
        # Update track states and remove stale tracks
        self._maintain_tracks(own_state, current_time)
        
        # Calculate priorities for all tracks
        self._update_priorities(own_state)
        
        return list(self.tracks.values())
        
    def _process_contact(self, contact: Dict, current_time: float):
        """Process a single sensor contact"""
        contact_pos = np.array(contact['position'])
        contact_vel = np.array(contact.get('velocity', [0, 0, 0]))
        contact_id = contact.get('id', f"unknown_{len(self.tracks)}")
        
        # Try to correlate with existing track
        best_track = None
        best_distance = self.correlation_threshold
        
        for track_id, track in self.tracks.items():
            # Predict track position
            dt = current_time - track.last_update_time
            predicted_pos = track.position + track.velocity * dt
            
            # Check correlation distance
            distance = np.linalg.norm(contact_pos - predicted_pos)
            if distance < best_distance:
                best_distance = distance
                best_track = track
                
        if best_track is not None:
            # Update existing track
            dt = current_time - best_track.last_update_time
            if dt > 0:
                # Estimate acceleration
                new_accel = (contact_vel - best_track.velocity) / dt
                # Low-pass filter acceleration
                alpha = 0.3
                best_track.acceleration = alpha * new_accel + (1 - alpha) * best_track.acceleration
                
            best_track.position = contact_pos
            best_track.velocity = contact_vel
            best_track.last_update_time = current_time
            best_track.track_quality = min(1.0, best_track.track_quality + 0.1)
            best_track.time_since_detection = current_time - self.current_time
            
            # Update type if provided
            if 'type' in contact:
                best_track.target_type = TargetType(contact['type'])
            if 'threat_level' in contact:
                best_track.threat_level = ThreatLevel(contact['threat_level'])
                
        else:
            # Create new track if under limit
            if len(self.tracks) < self.max_tracks:
                new_track = TargetTrack(
                    id=contact_id,
                    position=contact_pos,
                    velocity=contact_vel,
                    last_update_time=current_time,
                    time_since_detection=0.0,
                    track_quality=0.5  # Start with medium confidence
                )
                
                if 'type' in contact:
                    new_track.target_type = TargetType(contact['type'])
                if 'threat_level' in contact:
                    new_track.threat_level = ThreatLevel(contact['threat_level'])
                    
                self.tracks[contact_id] = new_track
                self.total_tracks_initiated += 1
                
    def _maintain_tracks(self, own_state: Dict, current_time: float):
        """Maintain track list, removing stale tracks"""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            # Check track age
            track_age = current_time - track.last_update_time
            
            if track_age > self.track_timeout:
                tracks_to_remove.append(track_id)
            else:
                # Degrade track quality over time
                track.track_quality *= 0.99
                
                # Update time since detection
                track.time_since_detection = track_age
                
                # Predict intercept geometry
                self._update_intercept_prediction(track, own_state)
                
        # Remove stale tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.total_tracks_dropped += 1
            
    def _update_intercept_prediction(self, track: TargetTrack, own_state: Dict):
        """Update intercept time and point prediction"""
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        
        rel_pos = track.position - own_pos
        rel_vel = track.velocity - own_vel
        
        # Simple constant velocity intercept prediction
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos)
        
        if abs(a) > 1e-6:
            # Solve for time to closest point of approach
            t_cpa = -b / (2 * a)
            
            if t_cpa > 0:
                track.predicted_intercept_time = t_cpa
                track.predicted_intercept_point = track.position + track.velocity * t_cpa
            else:
                track.predicted_intercept_time = None
                track.predicted_intercept_point = None
        else:
            track.predicted_intercept_time = None
            track.predicted_intercept_point = None
            
    def _update_priorities(self, own_state: Dict):
        """Update priority scores for all tracks using MCDM"""
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        own_heading = np.arctan2(own_vel[1], own_vel[0]) if np.linalg.norm(own_vel[:2]) > 1 else 0
        
        # Normalize factors for MCDM
        ranges = []
        closing_speeds = []
        angles_off = []
        
        for track in self.tracks.values():
            rel_pos = track.position - own_pos
            rel_vel = track.velocity - own_vel
            range_to_target = np.linalg.norm(rel_pos)
            
            ranges.append(range_to_target)
            closing_speeds.append(-np.dot(rel_vel, rel_pos) / max(range_to_target, 1.0))
            
            # Angle off boresight
            bearing_to_target = np.arctan2(rel_pos[1], rel_pos[0])
            angle_off = abs(bearing_to_target - own_heading)
            angle_off = min(angle_off, 2*np.pi - angle_off)  # Wrap angle
            angles_off.append(angle_off)
            
        # Normalize to 0-1 scale
        max_range = max(ranges) if ranges else 1.0
        max_closing = max(closing_speeds) if closing_speeds else 1.0
        min_closing = min(closing_speeds) if closing_speeds else 0.0
        max_angle = max(angles_off) if angles_off else np.pi
        
        # Calculate priority scores
        for i, track in enumerate(self.tracks.values()):
            scores = {}
            
            # Range score (inverse - closer is better)
            scores['range'] = 1.0 - (ranges[i] / max_range) if max_range > 0 else 0.0
            
            # Closing speed score (normalized)
            if max_closing > min_closing:
                scores['closing_speed'] = (closing_speeds[i] - min_closing) / (max_closing - min_closing)
            else:
                scores['closing_speed'] = 0.5
                
            # Threat level score
            scores['threat_level'] = track.threat_level.value / 5.0
            
            # Angle off score (inverse - smaller is better)
            scores['angle_off'] = 1.0 - (angles_off[i] / max_angle) if max_angle > 0 else 0.0
            
            # Altitude score (can be configured for different missions)
            altitude_diff = abs(track.position[2] - own_pos[2])
            scores['altitude'] = 1.0 - min(altitude_diff / 5000.0, 1.0)  # Normalize to 5000m
            
            # Track quality score
            scores['track_quality'] = track.track_quality
            
            # Calculate weighted sum
            track.priority_score = sum(scores[factor] * self.priority_weights[factor] 
                                     for factor in self.priority_weights)
            
    def get_highest_priority_target(self, exclude_engaged: bool = True) -> Optional[TargetTrack]:
        """
        Get the highest priority target for engagement.
        
        Args:
            exclude_engaged: Whether to exclude already engaged targets
            
        Returns:
            Highest priority target or None
        """
        valid_tracks = []
        
        for track in self.tracks.values():
            # Check if in engagement zone
            if not self._is_in_engagement_zone(track):
                continue
                
            # Skip engaged targets if requested
            if exclude_engaged and track.is_engaged:
                continue
                
            # Skip low quality tracks
            if track.track_quality < 0.3:
                continue
                
            valid_tracks.append(track)
            
        if not valid_tracks:
            return None
            
        # Return highest priority
        return max(valid_tracks, key=lambda t: t.priority_score)
        
    def get_targets_by_priority(self, max_targets: int = 10, 
                               exclude_engaged: bool = True) -> List[TargetTrack]:
        """
        Get list of targets sorted by priority.
        
        Args:
            max_targets: Maximum number of targets to return
            exclude_engaged: Whether to exclude engaged targets
            
        Returns:
            List of targets sorted by priority (highest first)
        """
        valid_tracks = []
        
        for track in self.tracks.values():
            if not self._is_in_engagement_zone(track):
                continue
                
            if exclude_engaged and track.is_engaged:
                continue
                
            if track.track_quality < 0.3:
                continue
                
            valid_tracks.append(track)
            
        # Sort by priority and return top N
        valid_tracks.sort(key=lambda t: t.priority_score, reverse=True)
        return valid_tracks[:max_targets]
        
    def _is_in_engagement_zone(self, track: TargetTrack) -> bool:
        """Check if target is within engagement zone"""
        # This would normally check against own position
        # Simplified for now - assumes we have access to own state
        
        # Check altitude limits
        if track.position[2] < self.engagement_zone.min_altitude:
            return False
        if track.position[2] > self.engagement_zone.max_altitude:
            return False
            
        # Range will be checked by caller with own_state
        return True
        
    def assign_target(self, target_id: str, interceptor_id: str) -> bool:
        """
        Assign a target to an interceptor.
        
        Args:
            target_id: Target track ID
            interceptor_id: Interceptor ID
            
        Returns:
            True if assignment successful
        """
        if target_id not in self.tracks:
            return False
            
        track = self.tracks[target_id]
        
        # Check if already engaged
        if track.is_engaged and track.assigned_interceptor != interceptor_id:
            return False
            
        track.is_engaged = True
        track.assigned_interceptor = interceptor_id
        
        # Record engagement
        track.engagement_history.append({
            'interceptor': interceptor_id,
            'time': self.current_time,
            'status': 'assigned'
        })
        
        self.total_engagements += 1
        return True
        
    def release_target(self, target_id: str, interceptor_id: str, 
                      status: str = 'released') -> bool:
        """
        Release a target from engagement.
        
        Args:
            target_id: Target track ID
            interceptor_id: Interceptor ID
            status: Engagement status (released, intercepted, missed)
            
        Returns:
            True if release successful
        """
        if target_id not in self.tracks:
            return False
            
        track = self.tracks[target_id]
        
        if track.assigned_interceptor != interceptor_id:
            return False
            
        track.is_engaged = False
        track.assigned_interceptor = None
        
        # Record in history
        track.engagement_history.append({
            'interceptor': interceptor_id,
            'time': self.current_time,
            'status': status
        })
        
        return True
        
    def get_engagement_recommendation(self, own_state: Dict, 
                                     capabilities: Dict) -> Dict[str, Any]:
        """
        Get comprehensive engagement recommendation.
        
        Args:
            own_state: Own aircraft state
            capabilities: Own capabilities (weapons, fuel, etc.)
            
        Returns:
            Engagement recommendation with analysis
        """
        recommendation = {
            'primary_target': None,
            'secondary_targets': [],
            'engagement_feasible': False,
            'limiting_factor': None,
            'tactical_assessment': '',
            'recommended_action': 'continue_search'
        }
        
        # Get priority targets
        targets = self.get_targets_by_priority(max_targets=5, exclude_engaged=True)
        
        if not targets:
            recommendation['tactical_assessment'] = "No valid targets in engagement zone"
            return recommendation
            
        primary = targets[0]
        recommendation['primary_target'] = primary
        recommendation['secondary_targets'] = targets[1:]
        
        # Assess engagement feasibility
        own_pos = np.array(own_state['position'])
        rel_pos = primary.position - own_pos
        range_to_target = np.linalg.norm(rel_pos)
        
        # Check range
        if range_to_target > self.engagement_zone.max_range:
            recommendation['limiting_factor'] = 'range_too_far'
            recommendation['tactical_assessment'] = f"Target beyond max range ({range_to_target:.0f}m)"
            recommendation['recommended_action'] = 'pursue'
            return recommendation
            
        if range_to_target < self.engagement_zone.min_range:
            recommendation['limiting_factor'] = 'range_too_close'
            recommendation['tactical_assessment'] = f"Target too close ({range_to_target:.0f}m)"
            recommendation['recommended_action'] = 'breakaway'
            return recommendation
            
        # Check closing speed
        own_vel = np.array(own_state['velocity'])
        rel_vel = primary.velocity - own_vel
        closing_speed = -np.dot(rel_vel, rel_pos) / range_to_target
        
        if closing_speed < self.engagement_zone.min_closing_speed:
            recommendation['limiting_factor'] = 'opening_too_fast'
            recommendation['tactical_assessment'] = "Target opening, cannot intercept"
            recommendation['recommended_action'] = 'find_new_target'
            return recommendation
            
        # Check energy state
        fuel_remaining = capabilities.get('fuel_fraction', 1.0)
        if fuel_remaining < 0.3:
            recommendation['limiting_factor'] = 'low_fuel'
            recommendation['tactical_assessment'] = "Fuel too low for engagement"
            recommendation['recommended_action'] = 'rtb'
            return recommendation
            
        # Engagement is feasible
        recommendation['engagement_feasible'] = True
        recommendation['recommended_action'] = 'engage'
        
        # Provide tactical assessment
        if primary.threat_level == ThreatLevel.CRITICAL:
            recommendation['tactical_assessment'] = "CRITICAL threat - immediate engagement required"
        elif primary.threat_level == ThreatLevel.HIGH:
            recommendation['tactical_assessment'] = "High priority target - engage when ready"
        elif closing_speed > 100:
            recommendation['tactical_assessment'] = f"High closure rate ({closing_speed:.0f}m/s) - engage quickly"
        elif len(targets) > 3:
            recommendation['tactical_assessment'] = f"Multiple targets ({len(targets)}) - prioritize closest"
        else:
            recommendation['tactical_assessment'] = "Standard engagement geometry"
            
        return recommendation
        
    def predict_intercept_sequence(self, own_state: Dict, 
                                  time_horizon: float = 60.0) -> List[Dict]:
        """
        Predict optimal intercept sequence for multiple targets.
        
        Args:
            own_state: Own aircraft state
            time_horizon: Planning horizon in seconds
            
        Returns:
            Sequence of intercept predictions
        """
        sequence = []
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        current_pos = own_pos.copy()
        current_time = 0.0
        
        # Get available targets
        available_targets = self.get_targets_by_priority(max_targets=10, exclude_engaged=True)
        engaged_targets = set()
        
        while current_time < time_horizon and available_targets:
            best_target = None
            best_time = float('inf')
            best_intercept_pos = None
            
            # Evaluate each target
            for target in available_targets:
                if target.id in engaged_targets:
                    continue
                    
                # Predict intercept time (simplified)
                rel_pos = target.position - current_pos
                range_to_target = np.linalg.norm(rel_pos)
                
                # Estimate time to intercept
                avg_speed = np.linalg.norm(own_vel)
                if avg_speed > 0:
                    time_to_intercept = range_to_target / avg_speed
                else:
                    continue
                    
                # Predict intercept position
                intercept_pos = target.position + target.velocity * time_to_intercept
                
                # Check if within time horizon
                if current_time + time_to_intercept < time_horizon:
                    if time_to_intercept < best_time:
                        best_target = target
                        best_time = time_to_intercept
                        best_intercept_pos = intercept_pos
                        
            if best_target:
                # Add to sequence
                sequence.append({
                    'target': best_target,
                    'intercept_time': current_time + best_time,
                    'intercept_position': best_intercept_pos,
                    'target_priority': best_target.priority_score
                })
                
                # Update current position and time
                current_pos = best_intercept_pos
                current_time += best_time + 5.0  # Add reattack delay
                engaged_targets.add(best_target.id)
            else:
                break
                
        return sequence
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get target management statistics"""
        return {
            'active_tracks': len(self.tracks),
            'engaged_tracks': sum(1 for t in self.tracks.values() if t.is_engaged),
            'total_initiated': self.total_tracks_initiated,
            'total_dropped': self.total_tracks_dropped,
            'total_engagements': self.total_engagements,
            'avg_track_quality': np.mean([t.track_quality for t in self.tracks.values()]) if self.tracks else 0,
            'threat_breakdown': self._get_threat_breakdown()
        }
        
    def _get_threat_breakdown(self) -> Dict[str, int]:
        """Get breakdown of tracks by threat level"""
        breakdown = {level.name: 0 for level in ThreatLevel}
        for track in self.tracks.values():
            breakdown[track.threat_level.name] += 1
        return breakdown