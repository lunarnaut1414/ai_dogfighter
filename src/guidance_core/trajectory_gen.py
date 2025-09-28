# src/guidance_core/trajectory_gen.py
"""
Trajectory generation and path planning algorithms.
Includes search patterns, waypoint generation, and optimal path planning.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class SearchPattern(Enum):
    """Types of search patterns"""
    EXPANDING_SQUARE = "expanding_square"
    PARALLEL_TRACK = "parallel_track"
    SECTOR_SEARCH = "sector_search"
    RANDOM_SEARCH = "random_search"
    SPIRAL = "spiral"
    RACETRACK = "racetrack"


class PathType(Enum):
    """Types of path segments"""
    STRAIGHT = "straight"
    TURN = "turn"
    CLIMB = "climb"
    DESCENT = "descent"
    LOITER = "loiter"


@dataclass
class Waypoint:
    """Waypoint definition"""
    position: np.ndarray  # [x, y, z] in meters
    speed: Optional[float] = None  # Desired speed in m/s
    heading: Optional[float] = None  # Desired heading in radians
    arrival_time: Optional[float] = None  # Desired arrival time
    loiter_time: Optional[float] = None  # Time to loiter at waypoint
    waypoint_type: str = "fly_by"  # fly_by, fly_over, or loiter


@dataclass
class TrajectorySegment:
    """Segment of a trajectory"""
    start_point: np.ndarray
    end_point: np.ndarray
    segment_type: PathType
    duration: float
    length: float
    control_points: Optional[List[np.ndarray]] = None  # For curved paths


class TrajectoryGenerator:
    """
    Generates trajectories for various mission phases.
    Handles search patterns, intercept paths, and evasive maneuvers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trajectory generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Search pattern parameters
        self.search_speed = config.get('search_speed', 50.0) if config else 50.0
        self.search_altitude = config.get('search_altitude', 2000.0) if config else 2000.0
        self.search_spacing = config.get('search_spacing', 1000.0) if config else 1000.0
        
        # Path constraints
        self.max_bank_angle = config.get('max_bank_angle', 60.0) if config else 60.0
        self.max_climb_rate = config.get('max_climb_rate', 10.0) if config else 10.0
        self.min_turn_radius = config.get('min_turn_radius', 100.0) if config else 100.0
        
        # Safety margins
        self.terrain_clearance = config.get('terrain_clearance', 200.0) if config else 200.0
        self.obstacle_clearance = config.get('obstacle_clearance', 500.0) if config else 500.0
        
    def generate_search_pattern(self, pattern_type: SearchPattern, 
                              center: np.ndarray, 
                              size: float,
                              current_position: np.ndarray) -> List[Waypoint]:
        """
        Generate waypoints for a search pattern.
        
        Args:
            pattern_type: Type of search pattern
            center: Center of search area [x, y, z]
            size: Size of search area in meters
            current_position: Current aircraft position
            
        Returns:
            List of waypoints defining the pattern
        """
        if pattern_type == SearchPattern.EXPANDING_SQUARE:
            return self._generate_expanding_square(center, size, current_position)
        elif pattern_type == SearchPattern.PARALLEL_TRACK:
            return self._generate_parallel_track(center, size, current_position)
        elif pattern_type == SearchPattern.SECTOR_SEARCH:
            return self._generate_sector_search(center, size, current_position)
        elif pattern_type == SearchPattern.SPIRAL:
            return self._generate_spiral(center, size, current_position)
        elif pattern_type == SearchPattern.RACETRACK:
            return self._generate_racetrack(center, size, current_position)
        else:  # Random search
            return self._generate_random_search(center, size, current_position)
            
    def _generate_expanding_square(self, center: np.ndarray, size: float, 
                                  current_pos: np.ndarray) -> List[Waypoint]:
        """Generate expanding square search pattern"""
        waypoints = []
        altitude = self.search_altitude
        
        # Start from nearest corner
        leg_length = self.search_spacing
        num_legs = int(size / self.search_spacing)
        
        # Current search position
        x, y = center[0], center[1]
        direction = 0  # 0=East, 1=North, 2=West, 3=South
        
        for i in range(num_legs * 4):
            # Calculate next waypoint
            if direction == 0:  # East
                x += leg_length
            elif direction == 1:  # North
                y += leg_length
            elif direction == 2:  # West
                x -= leg_length
            else:  # South
                y -= leg_length
                
            waypoints.append(Waypoint(
                position=np.array([x, y, altitude]),
                speed=self.search_speed,
                waypoint_type="fly_by"
            ))
            
            # Turn 90 degrees left
            direction = (direction + 1) % 4
            
            # Increase leg length every two legs
            if i % 2 == 1:
                leg_length += self.search_spacing
                
        return waypoints
        
    def _generate_parallel_track(self, center: np.ndarray, size: float,
                                current_pos: np.ndarray) -> List[Waypoint]:
        """Generate parallel track search pattern"""
        waypoints = []
        altitude = self.search_altitude
        
        # Calculate number of tracks
        num_tracks = int(size / self.search_spacing)
        track_length = size
        
        # Start position
        start_x = center[0] - size/2
        start_y = center[1] - size/2
        
        for i in range(num_tracks):
            # Eastbound leg
            if i % 2 == 0:
                waypoints.append(Waypoint(
                    position=np.array([start_x, start_y + i*self.search_spacing, altitude]),
                    speed=self.search_speed
                ))
                waypoints.append(Waypoint(
                    position=np.array([start_x + track_length, start_y + i*self.search_spacing, altitude]),
                    speed=self.search_speed
                ))
            else:
                # Westbound leg
                waypoints.append(Waypoint(
                    position=np.array([start_x + track_length, start_y + i*self.search_spacing, altitude]),
                    speed=self.search_speed
                ))
                waypoints.append(Waypoint(
                    position=np.array([start_x, start_y + i*self.search_spacing, altitude]),
                    speed=self.search_speed
                ))
                
        return waypoints
        
    def _generate_sector_search(self, center: np.ndarray, size: float,
                               current_pos: np.ndarray) -> List[Waypoint]:
        """Generate sector search pattern"""
        waypoints = []
        altitude = self.search_altitude
        
        # Number of radial lines
        num_radials = 8
        angle_step = 2 * np.pi / num_radials
        
        for i in range(num_radials):
            angle = i * angle_step
            
            # Outbound leg
            end_x = center[0] + size * np.cos(angle)
            end_y = center[1] + size * np.sin(angle)
            
            waypoints.append(Waypoint(
                position=np.array([center[0], center[1], altitude]),
                speed=self.search_speed
            ))
            waypoints.append(Waypoint(
                position=np.array([end_x, end_y, altitude]),
                speed=self.search_speed
            ))
            
        return waypoints
        
    def _generate_spiral(self, center: np.ndarray, size: float,
                        current_pos: np.ndarray) -> List[Waypoint]:
        """Generate spiral search pattern"""
        waypoints = []
        altitude = self.search_altitude
        
        # Archimedean spiral parameters
        a = 0  # Initial radius
        b = self.search_spacing / (2 * np.pi)  # Spacing between turns
        
        max_angle = (size / b)
        angle_step = np.pi / 4  # 45 degree steps
        
        angle = 0
        while angle < max_angle:
            r = a + b * angle
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            
            waypoints.append(Waypoint(
                position=np.array([x, y, altitude]),
                speed=self.search_speed,
                waypoint_type="fly_by"
            ))
            
            angle += angle_step
            
        return waypoints
        
    def _generate_racetrack(self, center: np.ndarray, size: float,
                           current_pos: np.ndarray) -> List[Waypoint]:
        """Generate racetrack holding pattern"""
        waypoints = []
        altitude = self.search_altitude
        
        # Racetrack dimensions
        length = size * 0.8
        width = size * 0.3
        
        # Four corners of racetrack
        waypoints.append(Waypoint(
            position=np.array([center[0] - length/2, center[1] - width/2, altitude]),
            speed=self.search_speed
        ))
        waypoints.append(Waypoint(
            position=np.array([center[0] + length/2, center[1] - width/2, altitude]),
            speed=self.search_speed
        ))
        waypoints.append(Waypoint(
            position=np.array([center[0] + length/2, center[1] + width/2, altitude]),
            speed=self.search_speed
        ))
        waypoints.append(Waypoint(
            position=np.array([center[0] - length/2, center[1] + width/2, altitude]),
            speed=self.search_speed
        ))
        
        # Close the loop
        waypoints.append(waypoints[0])
        
        return waypoints
        
    def _generate_random_search(self, center: np.ndarray, size: float,
                               current_pos: np.ndarray) -> List[Waypoint]:
        """Generate random search pattern"""
        waypoints = []
        altitude = self.search_altitude
        num_points = 20
        
        # Random seed for repeatability
        np.random.seed(42)
        
        for _ in range(num_points):
            # Random point within search area
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, size/2)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            waypoints.append(Waypoint(
                position=np.array([x, y, altitude]),
                speed=self.search_speed
            ))
            
        return waypoints
        
    def generate_intercept_trajectory(self, own_state: Dict, 
                                     target_state: Dict,
                                     intercept_time: float) -> TrajectorySegment:
        """
        Generate optimal intercept trajectory.
        
        Args:
            own_state: Own aircraft state
            target_state: Target state
            intercept_time: Predicted time to intercept
            
        Returns:
            Trajectory segment for intercept
        """
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        target_pos = np.array(target_state['position'])
        target_vel = np.array(target_state.get('velocity', [0, 0, 0]))
        
        # Predict intercept point
        intercept_point = target_pos + target_vel * intercept_time
        
        # Calculate required average velocity
        required_vel = (intercept_point - own_pos) / intercept_time
        
        # Create trajectory segment
        segment = TrajectorySegment(
            start_point=own_pos,
            end_point=intercept_point,
            segment_type=PathType.STRAIGHT,
            duration=intercept_time,
            length=np.linalg.norm(intercept_point - own_pos)
        )
        
        # Add control points for curved intercept if needed
        if intercept_time > 5.0:  # Long intercept, use curved path
            # Calculate lead point
            lead_distance = np.linalg.norm(own_vel) * 2.0  # 2 seconds ahead
            lead_direction = target_vel / np.linalg.norm(target_vel) if np.linalg.norm(target_vel) > 1 else np.array([1, 0, 0])
            lead_point = target_pos + lead_direction * lead_distance
            
            # Bezier control points
            segment.control_points = [
                own_pos,
                own_pos + own_vel * intercept_time * 0.3,
                lead_point,
                intercept_point
            ]
            
        return segment
        
    def generate_evasive_maneuver(self, own_state: Dict, 
                                 threat_bearing: float,
                                 threat_range: float) -> List[Waypoint]:
        """
        Generate evasive maneuver trajectory.
        
        Args:
            own_state: Own aircraft state
            threat_bearing: Bearing to threat in radians
            threat_range: Range to threat in meters
            
        Returns:
            Waypoints for evasive maneuver
        """
        own_pos = np.array(own_state['position'])
        own_vel = np.array(own_state['velocity'])
        own_speed = np.linalg.norm(own_vel)
        
        waypoints = []
        
        # Barrel roll evasion pattern
        if threat_range < 500:
            # Immediate break turn
            # Turn 90 degrees away from threat
            escape_bearing = threat_bearing + np.pi/2
            
            # First waypoint - hard break
            break_distance = 200
            wp1 = np.array([
                own_pos[0] + break_distance * np.cos(escape_bearing),
                own_pos[1] + break_distance * np.sin(escape_bearing),
                own_pos[2] - 100  # Descend
            ])
            waypoints.append(Waypoint(wp1, speed=own_speed * 1.2))
            
            # Second waypoint - reverse turn
            wp2 = np.array([
                own_pos[0] + break_distance * np.cos(escape_bearing + np.pi),
                own_pos[1] + break_distance * np.sin(escape_bearing + np.pi),
                own_pos[2] + 100  # Climb
            ])
            waypoints.append(Waypoint(wp2, speed=own_speed))
            
        else:
            # Weaving pattern for longer range
            weave_amplitude = 500
            weave_length = 1000
            
            for i in range(3):
                lateral_offset = weave_amplitude * (1 if i % 2 == 0 else -1)
                forward_offset = weave_length * (i + 1)
                
                # Calculate waypoint position
                heading = np.arctan2(own_vel[1], own_vel[0])
                wp = np.array([
                    own_pos[0] + forward_offset * np.cos(heading) + lateral_offset * np.cos(heading + np.pi/2),
                    own_pos[1] + forward_offset * np.sin(heading) + lateral_offset * np.sin(heading + np.pi/2),
                    own_pos[2] + (100 if i % 2 == 0 else -100)  # Altitude variation
                ])
                
                waypoints.append(Waypoint(wp, speed=own_speed))
                
        return waypoints
        
    def generate_return_path(self, current_pos: np.ndarray, 
                           base_pos: np.ndarray,
                           obstacles: Optional[List[Dict]] = None) -> List[Waypoint]:
        """
        Generate return-to-base trajectory avoiding obstacles.
        
        Args:
            current_pos: Current position
            base_pos: Base position
            obstacles: List of obstacles to avoid
            
        Returns:
            Waypoints for return path
        """
        waypoints = []
        
        if obstacles is None or len(obstacles) == 0:
            # Direct path
            waypoints.append(Waypoint(
                position=base_pos,
                speed=self.search_speed,
                waypoint_type="fly_over"
            ))
        else:
            # Simple obstacle avoidance - add intermediate waypoints
            # This is simplified - real implementation would use A* or RRT*
            
            direct_vector = base_pos - current_pos
            direct_distance = np.linalg.norm(direct_vector)
            direct_heading = np.arctan2(direct_vector[1], direct_vector[0])
            
            # Check for obstacles along path
            needs_avoidance = False
            for obstacle in obstacles:
                obs_pos = np.array(obstacle['position'])
                obs_radius = obstacle.get('radius', 500)
                
                # Distance from obstacle to line
                t = np.dot(obs_pos - current_pos, direct_vector) / (direct_distance * direct_distance)
                t = np.clip(t, 0, 1)
                closest_point = current_pos + t * direct_vector
                distance_to_path = np.linalg.norm(obs_pos - closest_point)
                
                if distance_to_path < obs_radius + self.obstacle_clearance:
                    needs_avoidance = True
                    
                    # Add avoidance waypoint
                    avoidance_angle = direct_heading + np.pi/3  # 60 degrees offset
                    avoidance_distance = obs_radius + self.obstacle_clearance
                    
                    wp = np.array([
                        obs_pos[0] + avoidance_distance * np.cos(avoidance_angle),
                        obs_pos[1] + avoidance_distance * np.sin(avoidance_angle),
                        current_pos[2]
                    ])
                    waypoints.append(Waypoint(wp, speed=self.search_speed))
                    
            # Final waypoint at base
            waypoints.append(Waypoint(
                position=base_pos,
                speed=self.search_speed * 0.8,  # Slower for landing
                waypoint_type="fly_over"
            ))
            
        return waypoints
        
    def smooth_trajectory(self, waypoints: List[Waypoint], 
                        max_acceleration: float = 20.0) -> List[Waypoint]:
        """
        Smooth trajectory to respect dynamic constraints.
        
        Args:
            waypoints: Original waypoints
            max_acceleration: Maximum acceleration in m/s²
            
        Returns:
            Smoothed waypoints
        """
        if len(waypoints) < 2:
            return waypoints
            
        smoothed = [waypoints[0]]
        
        for i in range(1, len(waypoints) - 1):
            prev_wp = smoothed[-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]
            
            # Calculate turn angle
            v1 = curr_wp.position - prev_wp.position
            v2 = next_wp.position - curr_wp.position
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                
                # Angle between segments
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                angle = np.arccos(cos_angle)
                
                # If sharp turn, add intermediate waypoint
                if angle > np.pi/4:  # More than 45 degrees
                    # Calculate turn radius based on speed
                    speed = curr_wp.speed or self.search_speed
                    turn_radius = speed * speed / max_acceleration
                    
                    # Add fly-by waypoint before turn
                    offset = min(turn_radius, np.linalg.norm(v1) * 0.3)
                    fly_by_pos = curr_wp.position - v1_norm * offset
                    smoothed.append(Waypoint(
                        position=fly_by_pos,
                        speed=speed,
                        waypoint_type="fly_by"
                    ))
                    
            smoothed.append(curr_wp)
            
        smoothed.append(waypoints[-1])
        return smoothed
        
    def calculate_fuel_consumption(self, trajectory: List[Waypoint], 
                                 aircraft_params: Dict) -> float:
        """
        Estimate fuel consumption for trajectory.
        
        Args:
            trajectory: List of waypoints
            aircraft_params: Aircraft performance parameters
            
        Returns:
            Estimated fuel consumption in kg
        """
        if len(trajectory) < 2:
            return 0.0
            
        total_fuel = 0.0
        sfc = aircraft_params.get('specific_fuel_consumption', 0.0001)  # kg/N/s
        thrust_cruise = aircraft_params.get('thrust_cruise', 300)  # N
        
        for i in range(len(trajectory) - 1):
            # Distance between waypoints
            distance = np.linalg.norm(trajectory[i+1].position - trajectory[i].position)
            
            # Time to traverse (assuming constant speed)
            speed = trajectory[i].speed or self.search_speed
            time_segment = distance / speed
            
            # Fuel consumption
            fuel_segment = sfc * thrust_cruise * time_segment
            total_fuel += fuel_segment
            
        return total_fuel