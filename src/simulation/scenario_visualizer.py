"""
ScenarioVisualizer: Visualization system for scenario execution and replay.
Provides real-time 3D visualization and post-run analysis capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time


class ScenarioVisualizer:
    """
    Real-time and replay visualization for scenarios.
    """
    
    def __init__(self, 
                 scenario_runner: Optional[Any] = None,
                 recording_file: Optional[str] = None,
                 figsize: Tuple[float, float] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            scenario_runner: Live scenario runner instance
            recording_file: Path to recorded scenario data
            figsize: Figure size for visualization
        """
        self.scenario_runner = scenario_runner
        self.recording_file = recording_file
        self.figsize = figsize
        
        # Load recorded data if provided
        self.recorded_data = None
        if recording_file:
            self._load_recording()
            
        # Visualization components
        self.fig = None
        self.axes = {}
        self.plots = {}
        self.annotations = {}
        
        # Animation
        self.animation = None
        self.current_frame = 0
        self.playing = True
        
        # Trail history
        self.position_history = {}
        self.trail_length = 100  # Number of points to keep in trail
        
    def _load_recording(self):
        """Load recorded scenario data"""
        with open(self.recording_file, 'r') as f:
            self.recorded_data = json.load(f)
            
    def create_live_display(self):
        """Create live visualization display"""
        if not self.scenario_runner:
            raise ValueError("No scenario runner provided for live display")
            
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle("Live Scenario Execution", fontsize=14, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 3D trajectory view
        self.axes['3d'] = self.fig.add_subplot(gs[:, 0:2], projection='3d')
        self._setup_3d_axis(self.axes['3d'])
        
        # 2D top-down view
        self.axes['2d'] = self.fig.add_subplot(gs[0, 2])
        self._setup_2d_axis(self.axes['2d'])
        
        # Status panel
        self.axes['status'] = self.fig.add_subplot(gs[1, 2])
        self._setup_status_panel(self.axes['status'])
        
        # Initialize plots
        self._initialize_plots()
        
    def create_replay_display(self):
        """Create replay visualization display"""
        if not self.recorded_data:
            raise ValueError("No recorded data loaded")
            
        self.fig = plt.figure(figsize=self.figsize)
        scenario_name = self.recorded_data['config'].get('name', 'Scenario Replay')
        self.fig.suptitle(f"Replay: {scenario_name}", fontsize=14, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 3D trajectory view
        self.axes['3d'] = self.fig.add_subplot(gs[:, 0:2], projection='3d')
        self._setup_3d_axis(self.axes['3d'])
        
        # 2D top-down view  
        self.axes['2d'] = self.fig.add_subplot(gs[0, 2])
        self._setup_2d_axis(self.axes['2d'])
        
        # Metrics panel
        self.axes['metrics'] = self.fig.add_subplot(gs[1, 2])
        self._setup_metrics_panel(self.axes['metrics'])
        
        # Initialize plots with recorded data
        self._initialize_replay_plots()
        
    def _setup_3d_axis(self, ax):
        """Setup 3D axis properties"""
        ax.set_xlabel('East (m)', fontsize=10)
        ax.set_ylabel('North (m)', fontsize=10)
        ax.set_zlabel('Altitude (m)', fontsize=10)
        ax.set_title('3D Trajectories', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable limits (will be adjusted based on data)
        ax.set_xlim([0, 50000])
        ax.set_ylim([0, 50000])
        ax.set_zlim([0, 10000])
        
    def _setup_2d_axis(self, ax):
        """Setup 2D top-down view axis"""
        ax.set_xlabel('East (m)', fontsize=10)
        ax.set_ylabel('North (m)', fontsize=10)
        ax.set_title('Top-Down View', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
    def _setup_status_panel(self, ax):
        """Setup status display panel"""
        ax.set_title('Scenario Status', fontsize=12)
        ax.axis('off')
        
    def _setup_metrics_panel(self, ax):
        """Setup metrics display panel"""
        ax.set_title('Performance Metrics', fontsize=12)
        ax.axis('off')
        
    def _initialize_plots(self):
        """Initialize plot elements for live display"""
        # 3D plots
        ax3d = self.axes['3d']
        
        # Interceptor plot
        self.plots['interceptor_3d'], = ax3d.plot([], [], [], 'b^-', 
                                                  markersize=10, label='Interceptor')
        self.plots['interceptor_trail_3d'], = ax3d.plot([], [], [], 'b-', 
                                                        alpha=0.3, linewidth=1)
        
        # Target plots
        self.plots['targets_3d'] = {}
        
        # 2D plots
        ax2d = self.axes['2d']
        
        self.plots['interceptor_2d'], = ax2d.plot([], [], 'b^', 
                                                  markersize=12, label='Interceptor')
        self.plots['interceptor_trail_2d'], = ax2d.plot([], [], 'b-', 
                                                        alpha=0.3, linewidth=1)
        
        self.plots['targets_2d'] = {}
        
        # Detection circles
        self.plots['detection_circle'] = plt.Circle((0, 0), 0, fill=False, 
                                                   edgecolor='cyan', 
                                                   linestyle='--', alpha=0.5)
        ax2d.add_patch(self.plots['detection_circle'])
        
    def _initialize_replay_plots(self):
        """Initialize plots for replay mode"""
        if not self.recorded_data:
            return
            
        frames = self.recorded_data['frames']
        if not frames:
            return
            
        # Get all asset IDs from first frame
        first_frame = frames[0]
        asset_ids = list(first_frame['assets'].keys())
        
        # Determine interceptor and targets
        self.interceptor_id = None
        self.target_ids = []
        
        for asset_id in asset_ids:
            if 'interceptor' in asset_id.lower():
                self.interceptor_id = asset_id
            else:
                self.target_ids.append(asset_id)
                
        # Initialize 3D plots
        ax3d = self.axes['3d']
        
        if self.interceptor_id:
            self.plots['interceptor_3d'], = ax3d.plot([], [], [], 'b^-',
                                                      markersize=10, label='Interceptor')
            self.plots['interceptor_trail_3d'], = ax3d.plot([], [], [], 'b-',
                                                            alpha=0.3, linewidth=1)
            
        self.plots['targets_3d'] = {}
        for i, target_id in enumerate(self.target_ids):
            color = ['r', 'orange', 'yellow', 'green', 'purple'][i % 5]
            self.plots['targets_3d'][target_id], = ax3d.plot([], [], [], f'{color}o-',
                                                             markersize=8, 
                                                             label=f'Target {i+1}')
            
        # Initialize 2D plots
        ax2d = self.axes['2d']
        
        if self.interceptor_id:
            self.plots['interceptor_2d'], = ax2d.plot([], [], 'b^',
                                                      markersize=12, label='Interceptor')
            self.plots['interceptor_trail_2d'], = ax2d.plot([], [], 'b-',
                                                            alpha=0.3, linewidth=1)
            
        self.plots['targets_2d'] = {}
        for i, target_id in enumerate(self.target_ids):
            color = ['r', 'orange', 'yellow', 'green', 'purple'][i % 5]
            self.plots['targets_2d'][target_id], = ax2d.plot([], [], f'{color}o',
                                                             markersize=8,
                                                             label=f'Target {i+1}')
            
    def update_live(self, frame):
        """Update live visualization"""
        if not self.scenario_runner or self.scenario_runner.state.value != "running":
            return self.plots.values()
            
        # Get current state from scenario runner
        assets = self.scenario_runner.asset_manager.get_all_assets()
        
        # Update interceptor
        interceptor_id = self.scenario_runner.interceptor_id
        if interceptor_id in assets:
            self._update_aircraft_plot(interceptor_id, assets[interceptor_id], 
                                      'interceptor', 'blue')
            
        # Update targets
        for target_id in self.scenario_runner.target_ids:
            if target_id in assets:
                self._update_aircraft_plot(target_id, assets[target_id],
                                          'target', 'red')
                
        # Update status
        self._update_status_panel()
        
        return list(self.plots.values())
        
    def update_replay(self, frame_idx):
        """Update replay visualization"""
        if not self.recorded_data:
            return []
            
        frames = self.recorded_data['frames']
        if frame_idx >= len(frames):
            return list(self.plots.values())
            
        frame_data = frames[frame_idx]
        
        # Update time display
        time_text = f"Time: {frame_data['time']:.1f}s"
        self.axes['3d'].set_title(f'3D Trajectories - {time_text}', fontsize=12)
        
        # Update interceptor
        if self.interceptor_id and self.interceptor_id in frame_data['assets']:
            asset_data = frame_data['assets'][self.interceptor_id]
            self._update_replay_aircraft(self.interceptor_id, asset_data, 'interceptor')
            
        # Update targets
        for target_id in self.target_ids:
            if target_id in frame_data['assets']:
                asset_data = frame_data['assets'][target_id]
                self._update_replay_aircraft(target_id, asset_data, 'target')
                
        # Update metrics
        self._update_metrics_panel(frame_data)
        
        return list(self.plots.values())
        
    def _update_aircraft_plot(self, aircraft_id: str, asset_info: Dict, 
                             aircraft_type: str, color: str):
        """Update aircraft plot in live mode"""
        position = asset_info['state']['position']
        
        # Update position history
        if aircraft_id not in self.position_history:
            self.position_history[aircraft_id] = []
            
        self.position_history[aircraft_id].append(position.copy())
        
        # Limit trail length
        if len(self.position_history[aircraft_id]) > self.trail_length:
            self.position_history[aircraft_id].pop(0)
            
        # Update plots based on type
        if aircraft_type == 'interceptor':
            # 3D plot
            self.plots['interceptor_3d'].set_data([position[0]], [position[1]])
            self.plots['interceptor_3d'].set_3d_properties([position[2]])
            
            # Trail
            if len(self.position_history[aircraft_id]) > 1:
                trail = np.array(self.position_history[aircraft_id])
                self.plots['interceptor_trail_3d'].set_data(trail[:, 0], trail[:, 1])
                self.plots['interceptor_trail_3d'].set_3d_properties(trail[:, 2])
                
            # 2D plot
            self.plots['interceptor_2d'].set_data([position[0]], [position[1]])
            
            if len(self.position_history[aircraft_id]) > 1:
                trail = np.array(self.position_history[aircraft_id])
                self.plots['interceptor_trail_2d'].set_data(trail[:, 0], trail[:, 1])
                
        else:  # target
            # Create plots if they don't exist
            if aircraft_id not in self.plots['targets_3d']:
                ax3d = self.axes['3d']
                self.plots['targets_3d'][aircraft_id], = ax3d.plot([], [], [], 
                                                                   f'{color}o-',
                                                                   markersize=8)
                ax2d = self.axes['2d']
                self.plots['targets_2d'][aircraft_id], = ax2d.plot([], [], 
                                                                   f'{color}o',
                                                                   markersize=8)
                
            # Update position
            self.plots['targets_3d'][aircraft_id].set_data([position[0]], [position[1]])
            self.plots['targets_3d'][aircraft_id].set_3d_properties([position[2]])
            self.plots['targets_2d'][aircraft_id].set_data([position[0]], [position[1]])
            
    def _update_replay_aircraft(self, aircraft_id: str, asset_data: Dict,
                               aircraft_type: str):
        """Update aircraft plot in replay mode"""
        position = asset_data['position']
        
        # Update position history
        if aircraft_id not in self.position_history:
            self.position_history[aircraft_id] = []
            
        self.position_history[aircraft_id].append(position)
        
        # Limit trail length
        if len(self.position_history[aircraft_id]) > self.trail_length:
            self.position_history[aircraft_id].pop(0)
            
        # Update plots
        if aircraft_type == 'interceptor':
            self.plots['interceptor_3d'].set_data([position[0]], [position[1]])
            self.plots['interceptor_3d'].set_3d_properties([position[2]])
            
            if len(self.position_history[aircraft_id]) > 1:
                trail = np.array(self.position_history[aircraft_id])
                self.plots['interceptor_trail_3d'].set_data(trail[:, 0], trail[:, 1])
                self.plots['interceptor_trail_3d'].set_3d_properties(trail[:, 2])
                
            self.plots['interceptor_2d'].set_data([position[0]], [position[1]])
            
            if len(self.position_history[aircraft_id]) > 1:
                trail = np.array(self.position_history[aircraft_id])
                self.plots['interceptor_trail_2d'].set_data(trail[:, 0], trail[:, 1])
                
        else:  # target
            if aircraft_id in self.plots['targets_3d']:
                self.plots['targets_3d'][aircraft_id].set_data([position[0]], [position[1]])
                self.plots['targets_3d'][aircraft_id].set_3d_properties([position[2]])
                self.plots['targets_2d'][aircraft_id].set_data([position[0]], [position[1]])
                
    def _update_status_panel(self):
        """Update status panel in live mode"""
        ax = self.axes['status']
        ax.clear()
        ax.axis('off')
        
        if not self.scenario_runner:
            return
            
        # Gather status info
        interceptor = self.scenario_runner.asset_manager.assets[
            self.scenario_runner.interceptor_id
        ].aircraft
        
        status_text = [
            f"Time: {self.scenario_runner.time:.1f}s",
            f"",
            f"Interceptor Status:",
            f"  Altitude: {interceptor.state.position[2]:.0f}m",
            f"  Speed: {interceptor.state.velocity:.1f}m/s",
            f"  Fuel: {interceptor.state.fuel_remaining/interceptor.fuel_capacity:.1%}",
            f"",
            f"Targets Remaining: {len(self.scenario_runner.target_ids)}",
            f"Intercepts: {self.scenario_runner.metrics.intercepts}",
            f"",
            f"Update Rate: {1000/self.scenario_runner.metrics.mean_update_time_ms:.1f} Hz"
        ]
        
        # Display text
        y_pos = 0.95
        for line in status_text:
            ax.text(0.05, y_pos, line, fontsize=10, 
                   fontweight='bold' if line and not line.startswith(' ') else 'normal',
                   transform=ax.transAxes)
            y_pos -= 0.08
            
    def _update_metrics_panel(self, frame_data: Dict):
        """Update metrics panel in replay mode"""
        ax = self.axes['metrics']
        ax.clear()
        ax.axis('off')
        
        # Extract metrics
        objectives = frame_data.get('objectives', {})
        events = frame_data.get('events', [])
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        # Display metrics
        metrics_text = [
            f"Frame: {self.current_frame}/{len(self.recorded_data['frames'])}",
            f"Time: {frame_data['time']:.1f}s",
            f"",
            f"Events:",
        ]
        
        for event_type, count in event_counts.items():
            metrics_text.append(f"  {event_type}: {count}")
            
        if objectives:
            metrics_text.append("")
            metrics_text.append("Objectives:")
            if 'summary' in objectives:
                summary = objectives['summary']
                metrics_text.append(f"  Completed: {summary.get('completed', 0)}")
                metrics_text.append(f"  Pending: {summary.get('pending', 0)}")
                
        # Display text
        y_pos = 0.95
        for line in metrics_text:
            ax.text(0.05, y_pos, line, fontsize=10,
                   fontweight='bold' if line and not line.startswith(' ') else 'normal',
                   transform=ax.transAxes)
            y_pos -= 0.08
            
    def animate_live(self, interval: int = 50):
        """Start live animation"""
        if not self.scenario_runner:
            raise ValueError("No scenario runner for live animation")
            
        self.animation = FuncAnimation(
            self.fig,
            self.update_live,
            interval=interval,
            blit=False,
            repeat=True
        )
        
    def animate_replay(self, interval: int = 50):
        """Start replay animation"""
        if not self.recorded_data:
            raise ValueError("No recorded data for replay")
            
        num_frames = len(self.recorded_data['frames'])
        
        def update(frame_idx):
            self.current_frame = frame_idx
            return self.update_replay(frame_idx)
            
        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=True
        )
        
    def show(self):
        """Display visualization"""
        if self.fig:
            plt.show()
            
    def save_animation(self, filename: str, fps: int = 20):
        """Save animation to file"""
        if self.animation:
            self.animation.save(filename, fps=fps)
            print(f"Animation saved to {filename}")
            
    def export_metrics(self, filename: str):
        """Export metrics to CSV"""
        if not self.recorded_data:
            print("No recorded data to export")
            return
            
        import csv
        
        metrics = self.recorded_data.get('metrics', {})
        frames = self.recorded_data.get('frames', [])
        
        # Compile time-series data
        data_rows = []
        for frame in frames:
            row = {
                'time': frame['time'],
                'num_targets': len([a for a in frame['assets'] if 'target' in a.lower()]),
                'num_events': len(frame.get('events', []))
            }
            
            # Add asset-specific data
            for asset_id, asset_data in frame['assets'].items():
                prefix = asset_id.replace(' ', '_')
                row[f'{prefix}_x'] = asset_data['position'][0]
                row[f'{prefix}_y'] = asset_data['position'][1]
                row[f'{prefix}_z'] = asset_data['position'][2]
                row[f'{prefix}_fuel'] = asset_data.get('fuel', 0)
                
            data_rows.append(row)
            
        # Write CSV
        if data_rows:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_rows[0].keys())
                writer.writeheader()
                writer.writerows(data_rows)
                
            print(f"Metrics exported to {filename}")