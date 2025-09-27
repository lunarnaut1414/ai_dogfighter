"""
Scenario visualization and replay system.
Provides real-time and post-run visualization of scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class ScenarioVisualizer:
    """
    Visualization system for scenario replay and analysis.
    """
    
    def __init__(self, scenario_runner=None, recording_file=None):
        """
        Initialize visualizer.
        
        Args:
            scenario_runner: ScenarioRunner instance for live visualization
            recording_file: Path to recorded scenario for replay
        """
        self.runner = scenario_runner
        self.recording = None
        
        if recording_file:
            self.load_recording(recording_file)
            
        # Visualization settings
        self.figure_size = (16, 10)
        self.update_interval = 50  # ms
        self.trail_length = 100  # Number of points in trail
        
        # Color scheme
        self.colors = {
            'interceptor': 'blue',
            'target': 'red',
            'friendly': 'green',
            'unknown': 'gray',
            'trail': 'alpha',
            'sensor_fov': 'yellow',
            'threat_zone': 'red',
            'safe_zone': 'green'
        }
        
    def load_recording(self, recording_file: str):
        """Load recorded scenario data"""
        with open(recording_file, 'r') as f:
            self.recording = json.load(f)
        print(f"Loaded recording: {self.recording['scenario']}")
        print(f"Duration: {self.recording['metrics']['duration']:.1f}s")
        print(f"Frames: {len(self.recording['frames'])}")
        
    def create_live_display(self):
        """Create live display during scenario execution"""
        if not self.runner:
            raise ValueError("No scenario runner provided for live display")
            
        fig = plt.figure(figsize=self.figure_size)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main 2D view (large)
        self.ax_2d = fig.add_subplot(gs[:2, :2])
        self.ax_2d.set_xlabel('East (m)')
        self.ax_2d.set_ylabel('North (m)')
        self.ax_2d.set_title('Tactical Situation Display')
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.set_aspect('equal')
        
        # 3D view
        self.ax_3d = fig.add_subplot(gs[0, 2], projection='3d')
        self.ax_3d.set_xlabel('East')
        self.ax_3d.set_ylabel('North')
        self.ax_3d.set_zlabel('Alt')
        self.ax_3d.set_title('3D View')
        
        # Altitude profile
        self.ax_alt = fig.add_subplot(gs[1, 2])
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.set_title('Altitude Profile')
        self.ax_alt.grid(True, alpha=0.3)
        
        # Range plot
        self.ax_range = fig.add_subplot(gs[2, 0])
        self.ax_range.set_xlabel('Time (s)')
        self.ax_range.set_ylabel('Range (m)')
        self.ax_range.set_title('Engagement Range')
        self.ax_range.grid(True, alpha=0.3)
        
        # Speed plot
        self.ax_speed = fig.add_subplot(gs[2, 1])
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        self.ax_speed.set_title('Aircraft Speeds')
        self.ax_speed.grid(True, alpha=0.3)
        
        # Status panel
        self.ax_status = fig.add_subplot(gs[2, 2])
        self.ax_status.axis('off')
        self.ax_status.set_title('Mission Status')
        
        # Initialize plot elements
        self._init_plot_elements()
        
        # Set up animation
        self.animation = FuncAnimation(
            fig, self._update_live, interval=self.update_interval, blit=False
        )
        
        plt.suptitle(f"Scenario: {self.runner.config.get('name', 'Live')}", fontsize=14)
        
        return fig
        
    def _init_plot_elements(self):
        """Initialize plot elements for animation"""
        # 2D elements
        self.interceptor_marker, = self.ax_2d.plot([], [], 'b^', markersize=12, label='Interceptor')
        self.interceptor_trail, = self.ax_2d.plot([], [], 'b-', alpha=0.3, linewidth=1)
        
        self.target_markers = []
        self.target_trails = []
        
        # Sensor FOV cone
        self.sensor_cone = None
        
        # 3D elements
        self.interceptor_3d, = self.ax_3d.plot([], [], [], 'b^', markersize=8)
        self.target_3d_markers = []
        
        # Time series data
        self.time_data = []
        self.altitude_data = {'interceptor': [], 'targets': {}}
        self.range_data = {}
        self.speed_data = {'interceptor': [], 'targets': {}}
        
        # Altitude lines
        self.alt_line_interceptor, = self.ax_alt.plot([], [], 'b-', label='Interceptor')
        self.alt_lines_targets = {}
        
        # Range lines
        self.range_lines = {}
        
        # Speed lines  
        self.speed_line_interceptor, = self.ax_speed.plot([], [], 'b-', label='Interceptor')
        self.speed_lines_targets = {}
        
        # Status text
        self.status_text = self.ax_status.text(0.05, 0.95, '', transform=self.ax_status.transAxes,
                                              verticalalignment='top', fontsize=10, family='monospace')
        
    def _update_live(self, frame):
        """Update live visualization"""
        if not self.runner or not self.runner.asset_manager:
            return
            
        # Get current states
        interceptor_state = self.runner.asset_manager.get_asset_state(self.runner.interceptor_id)
        if not interceptor_state:
            return
            
        current_time = self.runner.asset_manager.time
        
        # Update time data
        self.time_data.append(current_time)
        if len(self.time_data) > self.trail_length:
            self.time_data.pop(0)
            
        # Update interceptor
        interceptor_pos = interceptor_state.position
        self.interceptor_marker.set_data([interceptor_pos[0]], [interceptor_pos[1]])
        
        # Update interceptor trail
        if not hasattr(self, 'interceptor_trail_data'):
            self.interceptor_trail_data = []
        self.interceptor_trail_data.append(interceptor_pos[:2])
        if len(self.interceptor_trail_data) > self.trail_length:
            self.interceptor_trail_data.pop(0)
        if len(self.interceptor_trail_data) > 1:
            trail = np.array(self.interceptor_trail_data)
            self.interceptor_trail.set_data(trail[:, 0], trail[:, 1])
            
        # Update 3D interceptor
        self.interceptor_3d.set_data([interceptor_pos[0]], [interceptor_pos[1]])
        self.interceptor_3d.set_3d_properties([interceptor_pos[2]])
        
        # Update altitude data
        self.altitude_data['interceptor'].append(interceptor_pos[2])
        if len(self.altitude_data['interceptor']) > self.trail_length:
            self.altitude_data['interceptor'].pop(0)
        self.alt_line_interceptor.set_data(self.time_data, self.altitude_data['interceptor'])
        
        # Update speed data
        self.speed_data['interceptor'].append(interceptor_state.velocity)
        if len(self.speed_data['interceptor']) > self.trail_length:
            self.speed_data['interceptor'].pop(0)
        self.speed_line_interceptor.set_data(self.time_data, self.speed_data['interceptor'])
        
        # Update targets
        for i, target_id in enumerate(self.runner.target_ids):
            target_state = self.runner.asset_manager.get_asset_state(target_id)
            if not target_state:
                continue
                
            target_pos = target_state.position
            
            # Create markers if needed
            if i >= len(self.target_markers):
                marker, = self.ax_2d.plot([], [], 'rv', markersize=10, label=f'Target {target_id}')
                trail, = self.ax_2d.plot([], [], 'r-', alpha=0.3, linewidth=1)
                self.target_markers.append(marker)
                self.target_trails.append(trail)
                
                marker_3d, = self.ax_3d.plot([], [], [], 'rv', markersize=6)
                self.target_3d_markers.append(marker_3d)
                
                self.altitude_data['targets'][target_id] = []
                self.speed_data['targets'][target_id] = []
                
                alt_line, = self.ax_alt.plot([], [], 'r-', alpha=0.5, label=target_id)
                self.alt_lines_targets[target_id] = alt_line
                
                speed_line, = self.ax_speed.plot([], [], 'r-', alpha=0.5, label=target_id)
                self.speed_lines_targets[target_id] = speed_line
                
                self.range_data[target_id] = []
                range_line, = self.ax_range.plot([], [], '-', label=target_id)
                self.range_lines[target_id] = range_line
                
            # Update target position
            self.target_markers[i].set_data([target_pos[0]], [target_pos[1]])
            
            # Update target trail
            if not hasattr(self, 'target_trail_data'):
                self.target_trail_data = {}
            if target_id not in self.target_trail_data:
                self.target_trail_data[target_id] = []
            self.target_trail_data[target_id].append(target_pos[:2])
            if len(self.target_trail_data[target_id]) > self.trail_length:
                self.target_trail_data[target_id].pop(0)
            if len(self.target_trail_data[target_id]) > 1:
                trail = np.array(self.target_trail_data[target_id])
                self.target_trails[i].set_data(trail[:, 0], trail[:, 1])
                
            # Update 3D
            self.target_3d_markers[i].set_data([target_pos[0]], [target_pos[1]])
            self.target_3d_markers[i].set_3d_properties([target_pos[2]])
            
            # Update altitude
            self.altitude_data['targets'][target_id].append(target_pos[2])
            if len(self.altitude_data['targets'][target_id]) > self.trail_length:
                self.altitude_data['targets'][target_id].pop(0)
            self.alt_lines_targets[target_id].set_data(self.time_data, self.altitude_data['targets'][target_id])
            
            # Update speed
            self.speed_data['targets'][target_id].append(target_state.velocity)
            if len(self.speed_data['targets'][target_id]) > self.trail_length:
                self.speed_data['targets'][target_id].pop(0)
            self.speed_lines_targets[target_id].set_data(self.time_data, self.speed_data['targets'][target_id])
            
            # Update range
            rel_state = self.runner.asset_manager.get_relative_state(
                self.runner.interceptor_id, target_id
            )
            if rel_state:
                self.range_data[target_id].append(rel_state['range'])
                if len(self.range_data[target_id]) > self.trail_length:
                    self.range_data[target_id].pop(0)
                self.range_lines[target_id].set_data(self.time_data, self.range_data[target_id])
                
        # Update status text
        status_lines = [
            f"Time: {current_time:.1f}s",
            f"Interceptor:",
            f"  Pos: ({interceptor_pos[0]:.0f}, {interceptor_pos[1]:.0f}, {interceptor_pos[2]:.0f})",
            f"  Speed: {interceptor_state.velocity:.1f} m/s",
            f"  Fuel: {interceptor_state.fuel_remaining:.1f}",
            "",
            "Objectives:"
        ]
        
        for obj in self.runner.objectives:
            status = "✓" if obj.completed else "✗" if obj.failed else "○"
            status_lines.append(f"  {status} {obj.description[:30]}")
            status_lines.append(f"      Progress: {obj.progress*100:.1f}%")
            
        self.status_text.set_text('\n'.join(status_lines))
        
        # Auto-scale axes
        self._autoscale_axes()
        
    def _autoscale_axes(self):
        """Auto-scale axes to show all data"""
        # 2D view - find bounds
        all_x = []
        all_y = []
        
        if hasattr(self, 'interceptor_trail_data') and self.interceptor_trail_data:
            trail = np.array(self.interceptor_trail_data)
            all_x.extend(trail[:, 0])
            all_y.extend(trail[:, 1])
            
        if hasattr(self, 'target_trail_data'):
            for trail_data in self.target_trail_data.values():
                if trail_data:
                    trail = np.array(trail_data)
                    all_x.extend(trail[:, 0])
                    all_y.extend(trail[:, 1])
                    
        if all_x and all_y:
            x_margin = (max(all_x) - min(all_x)) * 0.1 + 100
            y_margin = (max(all_y) - min(all_y)) * 0.1 + 100
            self.ax_2d.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            self.ax_2d.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
            
        # Time series plots
        if self.time_data:
            time_min = min(self.time_data)
            time_max = max(self.time_data)
            time_margin = (time_max - time_min) * 0.05 + 0.1
            
            self.ax_alt.set_xlim(time_min - time_margin, time_max + time_margin)
            self.ax_range.set_xlim(time_min - time_margin, time_max + time_margin)
            self.ax_speed.set_xlim(time_min - time_margin, time_max + time_margin)
            
            # Altitude limits
            all_alts = self.altitude_data['interceptor'].copy()
            for target_alts in self.altitude_data['targets'].values():
                all_alts.extend(target_alts)
            if all_alts:
                self.ax_alt.set_ylim(0, max(all_alts) * 1.1)
                
            # Range limits
            all_ranges = []
            for ranges in self.range_data.values():
                all_ranges.extend(ranges)
            if all_ranges:
                self.ax_range.set_ylim(0, max(all_ranges) * 1.1)
                
            # Speed limits
            all_speeds = self.speed_data['interceptor'].copy()
            for target_speeds in self.speed_data['targets'].values():
                all_speeds.extend(target_speeds)
            if all_speeds:
                self.ax_speed.set_ylim(0, max(all_speeds) * 1.1)
                
    def create_replay_animation(self, speed=1.0):
        """
        Create animation from recorded data.
        
        Args:
            speed: Replay speed multiplier
            
        Returns:
            Figure and animation objects
        """
        if not self.recording:
            raise ValueError("No recording loaded for replay")
            
        fig = plt.figure(figsize=self.figure_size)
        
        # Similar layout to live display
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main 2D view
        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.set_xlabel('East (m)')
        ax_main.set_ylabel('North (m)')
        ax_main.set_title('Mission Replay')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        
        # 3D view
        ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
        ax_3d.set_xlabel('East')
        ax_3d.set_ylabel('North')
        ax_3d.set_zlabel('Alt')
        ax_3d.set_title('3D View')
        
        # Progress bar
        ax_progress = fig.add_subplot(gs[1, 1])
        ax_progress.set_xlabel('Time (s)')
        ax_progress.set_ylabel('Objectives')
        ax_progress.set_title('Mission Progress')
        
        # Extract data for animation
        frames = self.recording['frames']
        
        # Initialize elements
        interceptor_marker, = ax_main.plot([], [], 'b^', markersize=12)
        interceptor_trail, = ax_main.plot([], [], 'b-', alpha=0.3)
        
        target_markers = {}
        target_trails = {}
        
        # Time text
        time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                verticalalignment='top', fontsize=12)
        
        def update(frame_idx):
            if frame_idx >= len(frames):
                return
                
            frame = frames[frame_idx]
            
            # Update time
            time_text.set_text(f"Time: {frame['time']:.1f}s")
            
            # Update interceptor
            int_pos = frame['interceptor']['position']
            interceptor_marker.set_data([int_pos[0]], [int_pos[1]])
            
            # Update interceptor trail
            trail_start = max(0, frame_idx - 50)
            trail_positions = [frames[i]['interceptor']['position'] 
                             for i in range(trail_start, frame_idx + 1)]
            if trail_positions:
                trail = np.array(trail_positions)
                interceptor_trail.set_data(trail[:, 0], trail[:, 1])
                
            # Update targets
            for target_id, target_data in frame['targets'].items():
                if target_id not in target_markers:
                    marker, = ax_main.plot([], [], 'rv', markersize=10)
                    trail, = ax_main.plot([], [], 'r-', alpha=0.3)
                    target_markers[target_id] = marker
                    target_trails[target_id] = trail
                    
                pos = target_data['position']
                target_markers[target_id].set_data([pos[0]], [pos[1]])
                
                # Trail
                trail_positions = []
                for i in range(trail_start, frame_idx + 1):
                    if target_id in frames[i]['targets']:
                        trail_positions.append(frames[i]['targets'][target_id]['position'])
                if trail_positions:
                    trail = np.array(trail_positions)
                    target_trails[target_id].set_data(trail[:, 0], trail[:, 1])
                    
            return [interceptor_marker, interceptor_trail, time_text] + list(target_markers.values())
            
        # Create animation
        interval = int(1000 / (30 * speed))  # 30 FPS base rate
        animation = FuncAnimation(fig, update, frames=len(frames),
                                interval=interval, blit=False, repeat=True)
        
        plt.suptitle(f"Replay: {self.recording['scenario']}", fontsize=14)
        
        return fig, animation
        
    def generate_summary_plots(self, save_path=None):
        """
        Generate summary plots for a completed scenario.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        if not self.runner and not self.recording:
            raise ValueError("No data available for summary")
            
        fig = plt.figure(figsize=(16, 12))
        
        # Create comprehensive summary layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Complete trajectory overview
        ax_traj = fig.add_subplot(gs[0, :2])
        self._plot_complete_trajectories(ax_traj)
        
        # Altitude vs time
        ax_alt = fig.add_subplot(gs[1, 0])
        self._plot_altitude_history(ax_alt)
        
        # Speed vs time
        ax_speed = fig.add_subplot(gs[1, 1])
        self._plot_speed_history(ax_speed)
        
        # Range vs time
        ax_range = fig.add_subplot(gs[1, 2])
        self._plot_range_history(ax_range)
        
        # Energy state
        ax_energy = fig.add_subplot(gs[2, 0])
        self._plot_energy_state(ax_energy)
        
        # Objectives timeline
        ax_obj = fig.add_subplot(gs[2, 1])
        self._plot_objectives_timeline(ax_obj)
        
        # Statistics panel
        ax_stats = fig.add_subplot(gs[:, 2])
        self._plot_statistics_panel(ax_stats)
        
        # Overall title
        scenario_name = self.runner.config.get('name') if self.runner else self.recording['scenario']
        plt.suptitle(f"Scenario Summary: {scenario_name}", fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary saved to: {save_path}")
            
        return fig
        
    def _plot_complete_trajectories(self, ax):
        """Plot complete flight paths"""
        ax.set_title('Complete Flight Trajectories')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot terrain contours if available
        if self.runner and hasattr(self.runner, 'battlespace'):
            battlespace = self.runner.battlespace
            terrain_step = 20
            terrain = battlespace.terrain.elevation[::terrain_step, ::terrain_step]
            x = np.linspace(0, battlespace.width, terrain.shape[1])
            y = np.linspace(0, battlespace.height, terrain.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.contour(X, Y, terrain, levels=10, colors='gray', alpha=0.2, linewidths=0.5)
            
        # Add legend
        ax.legend(loc='best')
        
    def _plot_altitude_history(self, ax):
        """Plot altitude history"""
        ax.set_title('Altitude History')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
    def _plot_speed_history(self, ax):
        """Plot speed history"""
        ax.set_title('Speed History')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
    def _plot_range_history(self, ax):
        """Plot engagement range history"""
        ax.set_title('Engagement Ranges')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Range (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
    def _plot_energy_state(self, ax):
        """Plot energy state evolution"""
        ax.set_title('Energy State')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Specific Energy (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
    def _plot_objectives_timeline(self, ax):
        """Plot objectives completion timeline"""
        ax.set_title('Objectives Timeline')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Objective')
        ax.grid(True, alpha=0.3)
        
    def _plot_statistics_panel(self, ax):
        """Display key statistics"""
        ax.axis('off')
        ax.set_title('Mission Statistics', fontweight='bold', fontsize=12)
        
        # Compile statistics text
        stats_text = []
        
        if self.runner:
            metrics = self.runner.metrics
            stats_text.extend([
                f"Duration: {self.runner.asset_manager.time:.1f}s",
                f"Intercepts: {len(metrics.intercept_events)}",
                f"Success Rate: {metrics.success_rate*100:.1f}%",
                f"",
                f"Performance:",
                f"  Update Rate: {1000/metrics.mean_update_time:.1f} Hz",
                f"  Max Update: {metrics.max_update_time*1000:.2f}ms",
                f"",
                f"Objectives Complete: {metrics.objectives_completed}",
                f"Objectives Failed: {metrics.objectives_failed}",
            ])
            
            if metrics.intercept_events:
                stats_text.append("")
                stats_text.append("Intercept Events:")
                for event in metrics.intercept_events[:5]:  # Show first 5
                    stats_text.append(f"  {event.target_id}: {event.range:.1f}m @ {event.time:.1f}s")
                    
        elif self.recording:
            metrics = self.recording['metrics']
            stats_text.extend([
                f"Duration: {metrics['duration']:.1f}s",
                f"Total Steps: {metrics['steps']}",
                f"Intercepts: {metrics['intercepts']}",
                f"Success Rate: {metrics['success_rate']*100:.1f}%",
            ])
            
        # Display text
        text = '\n'.join(stats_text)
        ax.text(0.1, 0.95, text, transform=ax.transAxes, verticalalignment='top',
               fontsize=10, family='monospace')
        
    def show(self):
        """Display all figures"""
        plt.show()