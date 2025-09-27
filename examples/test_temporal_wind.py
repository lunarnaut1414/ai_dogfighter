"""
Example demonstrating temporal wind field simulation over 3 hours.
Visualizes wind evolution at 1-minute intervals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import yaml

from src.battlespace import Battlespace


def create_battlespace():
    """Create and initialize battlespace."""
    config_file = "configs/battlespace/default_battlespace.yaml"
    print(f"Loading battlespace from {config_file}")
    
    battlespace = Battlespace(config_file=config_file)
    battlespace.generate()
    
    # Set initial time of day (noon)
    battlespace.weather.time_of_day = 12.0
    
    return battlespace


def simulate_wind_evolution(battlespace, duration_hours=3, dt_minutes=1):
    """
    Simulate wind field evolution over time.
    
    Args:
        battlespace: Battlespace object
        duration_hours: Simulation duration in hours
        dt_minutes: Time step in minutes
        
    Returns:
        List of wind field snapshots
    """
    duration_seconds = duration_hours * 3600
    dt_seconds = dt_minutes * 60
    num_steps = int(duration_seconds / dt_seconds) + 1
    
    snapshots = []
    times = []
    
    print(f"\nSimulating {duration_hours} hours of wind evolution...")
    print(f"Time step: {dt_minutes} minutes")
    print(f"Total steps: {num_steps}")
    
    for step in range(num_steps):
        current_time = step * dt_seconds
        
        # Update weather
        if step > 0:
            battlespace.weather.update(dt_seconds)
        
        # Get wind summary
        summary = battlespace.weather.get_wind_summary()
        
        # Sample wind field at surface level
        # Downsample for visualization
        sample_step = 5
        ny_sample = battlespace.weather.ny // sample_step
        nx_sample = battlespace.weather.nx // sample_step
        
        wind_u_surface = battlespace.weather.wind_u[0, ::sample_step, ::sample_step]
        wind_v_surface = battlespace.weather.wind_v[0, ::sample_step, ::sample_step]
        wind_speed = np.sqrt(wind_u_surface**2 + wind_v_surface**2)
        
        # Store snapshot
        snapshot = {
            'time': current_time,
            'time_minutes': current_time / 60,
            'hour_of_day': summary['hour_of_day'],
            'wind_u': wind_u_surface.copy(),
            'wind_v': wind_v_surface.copy(),
            'wind_speed': wind_speed,
            'summary': summary
        }
        snapshots.append(snapshot)
        times.append(current_time / 60)  # Convert to minutes
        
        # Print progress
        if step % 10 == 0:
            print(f"  Step {step}/{num_steps}: t={current_time/60:.0f} min, "
                  f"Hour={summary['hour_of_day']:.1f}, "
                  f"Mean wind={summary['mean_surface_wind']:.1f} m/s")
    
    return snapshots, times


def plot_wind_evolution(battlespace, snapshots, times):
    """
    Create static plots showing wind evolution at key times.
    """
    # Select 6 snapshots evenly distributed
    indices = np.linspace(0, len(snapshots)-1, 6, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Create grid for plotting
    sample_step = 5
    ny_sample = len(range(0, battlespace.weather.ny, sample_step))
    nx_sample = len(range(0, battlespace.weather.nx, sample_step))
    x = np.linspace(0, battlespace.width, nx_sample)
    y = np.linspace(0, battlespace.height, ny_sample)
    X, Y = np.meshgrid(x, y)
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        snapshot = snapshots[ax_idx]
        
        # Plot wind speed as background
        im = ax.contourf(X, Y, snapshot['wind_speed'], 
                        levels=20, cmap='YlOrRd', alpha=0.7)
        
        # Plot wind vectors (further downsampled for clarity)
        vector_step = 2
        ax.quiver(X[::vector_step, ::vector_step], 
                 Y[::vector_step, ::vector_step],
                 snapshot['wind_u'][::vector_step, ::vector_step],
                 snapshot['wind_v'][::vector_step, ::vector_step],
                 scale=200, alpha=0.8, width=0.002)
        
        ax.set_title(f"t = {snapshot['time_minutes']:.0f} min\n"
                    f"Hour: {snapshot['hour_of_day']:.1f}",
                    fontsize=10)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label='Wind Speed (m/s)', 
                 orientation='horizontal', fraction=0.05, pad=0.1)
    
    plt.suptitle('Wind Field Evolution Over 3 Hours', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_wind_statistics(snapshots, times):
    """
    Plot wind statistics over time.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Extract time series data
    hours_of_day = [s['hour_of_day'] for s in snapshots]
    mean_winds = [s['summary']['mean_surface_wind'] for s in snapshots]
    max_winds = [s['summary']['max_surface_wind'] for s in snapshots]
    
    # Wind speeds at different altitudes
    altitudes = ['0m', '1000m', '5000m', '10000m']
    wind_speeds = {alt: [] for alt in altitudes}
    wind_directions = {alt: [] for alt in altitudes}
    vertical_winds = {alt: [] for alt in altitudes}
    
    for snapshot in snapshots:
        for alt in altitudes:
            if alt in snapshot['summary']['wind_samples']:
                wind_speeds[alt].append(snapshot['summary']['wind_samples'][alt]['speed'])
                wind_directions[alt].append(snapshot['summary']['wind_samples'][alt]['direction'])
                vertical_winds[alt].append(snapshot['summary']['wind_samples'][alt]['vertical'])
    
    # Plot 1: Mean and Max surface wind
    ax = axes[0, 0]
    ax.plot(times, mean_winds, label='Mean', color='blue')
    ax.plot(times, max_winds, label='Max', color='red')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Surface Wind Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Hour of day
    ax = axes[0, 1]
    ax.plot(times, hours_of_day, color='orange')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Time of Day')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Wind speed by altitude
    ax = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for alt, color in zip(altitudes, colors):
        if alt in wind_speeds:
            ax.plot(times[:len(wind_speeds[alt])], wind_speeds[alt], 
                   label=alt, color=color)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed at Different Altitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Wind direction at surface
    ax = axes[1, 1]
    if '0m' in wind_directions:
        ax.plot(times[:len(wind_directions['0m'])], wind_directions['0m'], 
               color='darkblue')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Wind Direction (degrees)')
    ax.set_title('Surface Wind Direction')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Vertical wind component
    ax = axes[2, 0]
    for alt, color in zip(['0m', '1000m'], ['blue', 'green']):
        if alt in vertical_winds:
            ax.plot(times[:len(vertical_winds[alt])], vertical_winds[alt], 
                   label=alt, color=color, alpha=0.7)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Vertical Wind (m/s)')
    ax.set_title('Vertical Wind Component')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Wind rose (final snapshot)
    ax = axes[2, 1]
    final_snapshot = snapshots[-1]
    wind_u_flat = final_snapshot['wind_u'].flatten()
    wind_v_flat = final_snapshot['wind_v'].flatten()
    
    # Create wind rose bins
    theta = np.arctan2(wind_v_flat, wind_u_flat)
    r = np.sqrt(wind_u_flat**2 + wind_v_flat**2)
    
    ax = plt.subplot(3, 2, 6, projection='polar')
    ax.hist(theta, bins=16, weights=r, alpha=0.7, color='skyblue', edgecolor='darkblue')
    ax.set_title('Final Wind Rose (Surface)', pad=20)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_zero_location('N')  # North at top
    
    plt.suptitle('Wind Field Statistics Over 3 Hours', fontsize=14)
    plt.tight_layout()
    plt.show()


def create_animation(battlespace, snapshots):
    """
    Create animated visualization of wind evolution.
    """
    print("\nCreating animation...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create grid
    sample_step = 5
    ny_sample = len(range(0, battlespace.weather.ny, sample_step))
    nx_sample = len(range(0, battlespace.weather.nx, sample_step))
    x = np.linspace(0, battlespace.width, nx_sample)
    y = np.linspace(0, battlespace.height, ny_sample)
    X, Y = np.meshgrid(x, y)
    
    # Initialize plots
    levels = np.linspace(0, 20, 21)
    im1 = ax1.contourf(X, Y, snapshots[0]['wind_speed'], 
                      levels=levels, cmap='YlOrRd', alpha=0.7)
    
    # Quiver plot
    vector_step = 2
    q1 = ax1.quiver(X[::vector_step, ::vector_step], 
                   Y[::vector_step, ::vector_step],
                   snapshots[0]['wind_u'][::vector_step, ::vector_step],
                   snapshots[0]['wind_v'][::vector_step, ::vector_step],
                   scale=200, alpha=0.8, width=0.002)
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_aspect('equal')
    ax1.set_title('Wind Field')
    
    # Time series plot
    times = [s['time_minutes'] for s in snapshots]
    mean_winds = [s['summary']['mean_surface_wind'] for s in snapshots]
    line, = ax2.plot([], [], 'b-', label='Mean Surface Wind')
    point, = ax2.plot([], [], 'ro', markersize=8)
    
    ax2.set_xlim(0, max(times))
    ax2.set_ylim(0, max(mean_winds) * 1.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_title('Mean Surface Wind Speed')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add colorbar
    fig.colorbar(im1, ax=ax1, label='Wind Speed (m/s)')
    
    # Add text for time display
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    def update(frame):
        snapshot = snapshots[frame]
        
        # Clear and redraw contour
        ax1.clear()
        ax1.contourf(X, Y, snapshot['wind_speed'], 
                    levels=levels, cmap='YlOrRd', alpha=0.7)
        ax1.quiver(X[::vector_step, ::vector_step], 
                  Y[::vector_step, ::vector_step],
                  snapshot['wind_u'][::vector_step, ::vector_step],
                  snapshot['wind_v'][::vector_step, ::vector_step],
                  scale=200, alpha=0.8, width=0.002)
        
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_aspect('equal')
        ax1.set_title(f'Wind Field (Hour: {snapshot["hour_of_day"]:.1f})')
        
        # Update time series
        line.set_data(times[:frame+1], mean_winds[:frame+1])
        point.set_data([times[frame]], [mean_winds[frame]])
        
        # Update time text
        time_text.set_text(f'Time: {snapshot["time_minutes"]:.0f} minutes')
        
        return [line, point]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(snapshots), 
                        interval=100, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("Temporal Wind Field Simulation")
    print("=" * 60)
    
    # Create battlespace
    battlespace = create_battlespace()
    
    # Run simulation
    snapshots, times = simulate_wind_evolution(
        battlespace, 
        duration_hours=3,
        dt_minutes=1
    )
    
    print(f"\nSimulation complete. Generated {len(snapshots)} snapshots.")
    
    # Plot results
    print("\nGenerating visualizations...")
    
    # Static plots
    plot_wind_evolution(battlespace, snapshots, times)
    plot_wind_statistics(snapshots, times)
    
    # Optional: Create animation
    response = input("\nCreate animated visualization? (y/n): ")
    if response.lower() == 'y':
        anim = create_animation(battlespace, snapshots)
        
        # Optional: Save animation
        save = input("Save animation as MP4? (y/n): ")
        if save.lower() == 'y':
            print("Saving animation...")
            anim.save('wind_evolution.mp4', fps=10, writer='ffmpeg')
            print("Animation saved as wind_evolution.mp4")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()