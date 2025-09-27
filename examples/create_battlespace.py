"""
Example script demonstrating battlespace creation and visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import yaml

from src.battlespace import Battlespace


def create_battlespace_from_config(config_file: str) -> Battlespace:
    """
    Create battlespace from configuration file.
    
    Args:
        config_file: Path to YAML configuration
        
    Returns:
        Initialized battlespace
    """
    print(f"Loading configuration from {config_file}")
    battlespace = Battlespace(config_file=config_file)
    
    print("Generating battlespace...")
    battlespace.generate()
    
    # Print statistics
    info = battlespace.get_info()
    print("\nBattlespace Info:")
    print(f"  Dimensions: {info['width']}m x {info['height']}m x {info['altitude_ceiling']}m")
    print(f"  Grid Size: {info['grid_size']}")
    print(f"  Terrain Range: {info['terrain_range'][0]:.1f}m to {info['terrain_range'][1]:.1f}m")
    print(f"  Structures: {info['num_structures']}")
    print(f"  No-Fly Zones: {info['num_no_fly_zones']}")
    
    return battlespace


def visualize_terrain_2d(battlespace: Battlespace):
    """
    Create 2D visualization of terrain with contours.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Terrain elevation heatmap
    ax = axes[0, 0]
    terrain = battlespace.terrain.elevation
    im = ax.imshow(terrain, origin='lower', cmap='terrain', 
                   extent=[0, battlespace.width, 0, battlespace.height])
    ax.set_title('Terrain Elevation')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Terrain contours
    ax = axes[0, 1]
    x = np.linspace(0, battlespace.width, battlespace.terrain.nx)
    y = np.linspace(0, battlespace.height, battlespace.terrain.ny)
    X, Y = np.meshgrid(x, y)
    contour = ax.contour(X, Y, terrain, levels=15, colors='black', alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.contourf(X, Y, terrain, levels=15, cmap='terrain', alpha=0.8)
    ax.set_title('Terrain Contours')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    
    # Terrain types
    ax = axes[1, 0]
    terrain_types = battlespace.terrain.terrain_type
    im = ax.imshow(terrain_types, origin='lower', cmap='tab10',
                   extent=[0, battlespace.width, 0, battlespace.height])
    ax.set_title('Terrain Types')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    
    # Add legend for terrain types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Water'),
        Patch(facecolor='green', label='Grass'),
        Patch(facecolor='brown', label='Dirt'),
        Patch(facecolor='gray', label='Rock'),
        Patch(facecolor='white', label='Snow')
    ]
    ax.legend(handles=legend_elements[:5], loc='upper right')
    
    # Slope map
    ax = axes[1, 1]
    # Calculate slope magnitude
    gy, gx = np.gradient(terrain, battlespace.grid_resolution)
    slope = np.sqrt(gx**2 + gy**2)
    slope_angle = np.degrees(np.arctan(slope))
    
    im = ax.imshow(slope_angle, origin='lower', cmap='hot',
                   extent=[0, battlespace.width, 0, battlespace.height])
    ax.set_title('Terrain Slope')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    plt.colorbar(im, ax=ax, label='Slope (degrees)')
    
    plt.tight_layout()
    plt.show()


def visualize_terrain_3d(battlespace: Battlespace):
    """
    Create 3D visualization of terrain.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for performance
    step = 5
    terrain = battlespace.terrain.elevation[::step, ::step]
    nx, ny = terrain.shape[1], terrain.shape[0]
    
    x = np.linspace(0, battlespace.width, nx)
    y = np.linspace(0, battlespace.height, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, terrain, cmap='terrain',
                           linewidth=0, antialiased=True, alpha=0.9)
    
    # Add contour lines at base
    ax.contour(X, Y, terrain, zdir='z', offset=terrain.min(), 
               levels=10, cmap='terrain', alpha=0.5)
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Terrain Visualization')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')
    
    plt.show()


def test_queries(battlespace: Battlespace):
    """
    Test various battlespace queries.
    """
    print("\nTesting Battlespace Queries:")
    print("-" * 40)
    
    # Test positions
    test_positions = [
        np.array([25000, 25000, 1000]),  # Center
        np.array([10000, 10000, 500]),   # Low altitude
        np.array([40000, 40000, 5000]),  # High altitude
    ]
    
    for i, pos in enumerate(test_positions):
        print(f"\nPosition {i+1}: [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")
        
        # Terrain elevation
        elevation = battlespace.get_elevation(pos[0], pos[1])
        print(f"  Terrain elevation: {elevation:.1f}m")
        
        # Collision check
        collision = battlespace.check_collision(pos)
        print(f"  Collision: {collision}")
        
        # Wind
        wind = battlespace.get_wind(pos)
        print(f"  Wind: [{wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}] m/s")
        
        # Turbulence
        turb = battlespace.weather.get_turbulence_at(pos)
        print(f"  Turbulence: {turb:.2f}")
        
        # Air density
        density = battlespace.weather.get_density_at(pos[2])
        print(f"  Air density: {density:.3f} kg/m³")
        
        # Valid position
        valid = battlespace.is_valid_position(pos)
        print(f"  Valid position: {valid}")
        
        # Minimum safe altitude
        msa = battlespace.get_minimum_safe_altitude(pos[0], pos[1])
        print(f"  Minimum safe altitude: {msa:.1f}m")
    
    # Test line of sight
    print("\nLine of Sight Tests:")
    pos1 = np.array([10000, 10000, 2000])
    pos2 = np.array([40000, 40000, 2000])
    los = battlespace.get_line_of_sight(pos1, pos2)
    print(f"  LOS between corners at 2000m: {los}")


def main():
    """
    Main example execution.
    """
    # Path to config file
    config_file = "configs/battlespace/default_battlespace.yaml"
    
    # Create battlespace
    battlespace = create_battlespace_from_config(config_file)
    
    # Run tests
    test_queries(battlespace)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_terrain_2d(battlespace)
    visualize_terrain_3d(battlespace)
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()