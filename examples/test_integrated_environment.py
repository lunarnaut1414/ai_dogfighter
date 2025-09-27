"""
Demonstration of integrated terrain-weather effects and tactical features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.battlespace import Battlespace


def create_enhanced_battlespace():
    """Create battlespace with integrated features."""
    config_file = "configs/battlespace/default_battlespace.yaml"
    print("Creating enhanced battlespace with terrain-weather integration...")
    
    battlespace = Battlespace(
        config_file=config_file,
        enable_integration=True
    )
    
    print("Generating integrated environment...")
    battlespace.generate(seed=42)
    
    return battlespace


def visualize_integrated_effects(battlespace):
    """Visualize terrain with wind effects."""
    fig = plt.figure(figsize=(16, 10))
    
    # 3D terrain with wind vectors
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Downsample for visualization
    step = 10
    terrain = battlespace.terrain.elevation[::step, ::step]
    nx, ny = terrain.shape[1], terrain.shape[0]
    
    x = np.linspace(0, battlespace.width, nx)
    y = np.linspace(0, battlespace.height, ny)
    X, Y = np.meshgrid(x, y)
    
    # Plot terrain surface
    surf = ax1.plot_surface(X, Y, terrain, cmap='terrain',
                           alpha=0.8, linewidth=0, antialiased=True)
    
    # Add wind vectors at different altitudes
    wind_step = 2  # Use step relative to already downsampled terrain
    for altitude_idx, altitude in enumerate([500, 1500, 3000]):
        # Calculate actual grid points
        wind_nx = len(range(0, nx, wind_step))
        wind_ny = len(range(0, ny, wind_step))
        
        wx = np.zeros((wind_ny, wind_nx))
        wy = np.zeros((wind_ny, wind_nx))
        wz = np.zeros((wind_ny, wind_nx))
        
        for j_idx, j in enumerate(range(0, ny, wind_step)):
            for i_idx, i in enumerate(range(0, nx, wind_step)):
                if i < len(x) and j < len(y):
                    pos = np.array([x[i], y[j], altitude])
                    wind = battlespace.get_wind(pos)
                    wx[j_idx, i_idx] = wind[0]
                    wy[j_idx, i_idx] = wind[1]
                    wz[j_idx, i_idx] = wind[2]
        
        # Create meshgrid for wind positions
        X_wind = X[::wind_step, ::wind_step]
        Y_wind = Y[::wind_step, ::wind_step]
        Z_wind = terrain[::wind_step, ::wind_step] + altitude
        
        # Ensure all arrays have same shape
        min_shape = min(X_wind.shape[0], wx.shape[0]), min(X_wind.shape[1], wx.shape[1])
        
        # Plot wind arrows
        ax1.quiver(X_wind[:min_shape[0], :min_shape[1]], 
                  Y_wind[:min_shape[0], :min_shape[1]],
                  Z_wind[:min_shape[0], :min_shape[1]],
                  wx[:min_shape[0], :min_shape[1]], 
                  wy[:min_shape[0], :min_shape[1]], 
                  wz[:min_shape[0], :min_shape[1]],
                  length=500, normalize=True, alpha=0.6,
                  color=['red', 'green', 'blue'][altitude_idx])
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('Terrain with Wind at Multiple Altitudes')
    ax1.view_init(elev=20, azim=45)
    
    # Vertical wind (updrafts/downdrafts)
    ax2 = fig.add_subplot(2, 3, 2)
    vertical_wind = battlespace.weather.wind_w[0]  # Surface level
    im = ax2.contourf(vertical_wind, levels=20, cmap='RdBu_r', 
                     extent=[0, battlespace.width, 0, battlespace.height])
    ax2.set_title('Vertical Wind Component (Surface)')
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    plt.colorbar(im, ax=ax2, label='W (m/s)')
    
    # Turbulence map
    ax3 = fig.add_subplot(2, 3, 3)
    turbulence = battlespace.weather.turbulence[2]  # 1000m altitude
    im = ax3.contourf(turbulence, levels=20, cmap='YlOrRd',
                     extent=[0, battlespace.width, 0, battlespace.height])
    ax3.set_title('Turbulence Intensity (1000m)')
    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    plt.colorbar(im, ax=ax3, label='Intensity')
    
    # Radar shadow map
    ax4 = fig.add_subplot(2, 3, 4)
    if battlespace._radar_shadow_map is not None:
        im = ax4.imshow(battlespace._radar_shadow_map, origin='lower',
                       cmap='RdYlGn_r', alpha=0.7,
                       extent=[0, battlespace.width, 0, battlespace.height])
        
        # Overlay terrain contours
        terrain_step = 5
        terrain_subset = battlespace.terrain.elevation[::terrain_step, ::terrain_step]
        ny_cont, nx_cont = terrain_subset.shape
        x_cont = np.linspace(0, battlespace.width, nx_cont)
        y_cont = np.linspace(0, battlespace.height, ny_cont)
        X_cont, Y_cont = np.meshgrid(x_cont, y_cont)
        ax4.contour(X_cont, Y_cont, terrain_subset,
                   levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    ax4.set_title('Radar Shadow Map (Red=Hidden)')
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    
    # Thermal potential map
    ax5 = fig.add_subplot(2, 3, 5)
    if battlespace._thermal_map is not None:
        im = ax5.imshow(battlespace._thermal_map, origin='lower',
                       cmap='hot', extent=[0, battlespace.width, 0, battlespace.height])
        plt.colorbar(im, ax=ax5, label='Thermal Potential')
    ax5.set_title('Thermal Activity Potential')
    ax5.set_xlabel('East (m)')
    ax5.set_ylabel('North (m)')
    
    # Combined tactical view
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Show terrain as background
    im = ax6.contourf(battlespace.terrain.elevation, levels=20, cmap='terrain',
                     alpha=0.5, extent=[0, battlespace.width, 0, battlespace.height])
    
    # Mark high ground positions
    if 'high_ground' in battlespace.tactical_positions:
        for pos in battlespace.tactical_positions['high_ground'][:20]:  # Limit to 20
            ax6.plot(pos[0], pos[1], 'r^', markersize=8, label='High Ground' 
                    if pos is battlespace.tactical_positions['high_ground'][0] else '')
    
    ax6.set_title('Tactical Positions')
    ax6.set_xlabel('East (m)')
    ax6.set_ylabel('North (m)')
    ax6.legend()
    
    plt.suptitle('Integrated Terrain-Weather Effects', fontsize=14)
    plt.tight_layout()
    plt.show()


def test_aircraft_effects(battlespace):
    """Test environmental effects on aircraft at various positions."""
    print("\n" + "="*60)
    print("Testing Aircraft Environmental Effects")
    print("="*60)
    
    # Test positions
    test_scenarios = [
        {
            'name': 'Valley Flying',
            'position': np.array([25000, 25000, 200]),  # Low in center
            'velocity': np.array([100, 0, 0])  # Flying east at 100 m/s
        },
        {
            'name': 'Ridge Soaring',
            'position': np.array([30000, 30000, 2500]),  # Higher altitude
            'velocity': np.array([50, 50, 0])  # Diagonal flight
        },
        {
            'name': 'High Altitude',
            'position': np.array([20000, 20000, 8000]),  # High up
            'velocity': np.array([150, 0, 0])  # Fast flight
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        
        effects = battlespace.get_aircraft_environment_effects(
            scenario['position'], 
            scenario['velocity']
        )
        
        print(f"Position: {scenario['position']}")
        print(f"Velocity: {scenario['velocity']} m/s")
        print(f"\nEnvironmental Effects:")
        print(f"  AGL Altitude: {effects['agl_altitude']:.1f} m")
        print(f"  Wind: [{effects['wind_vector'][0]:.1f}, "
              f"{effects['wind_vector'][1]:.1f}, {effects['wind_vector'][2]:.1f}] m/s")
        print(f"  Relative Airspeed: {effects['relative_airspeed']:.1f} m/s")
        print(f"  Turbulence: {effects['turbulence_intensity']:.3f}")
        print(f"  Air Density: {effects['air_density']:.3f} kg/m³")
        print(f"  Density Ratio: {effects['density_ratio']:.3f}")
        
        if 'ground_effect_factor' in effects and effects['ground_effect_factor'] > 1.0:
            print(f"  Ground Effect: {effects['ground_effect_factor']:.2f}x lift")
        
        if 'in_mountain_wave' in effects:
            print(f"  In Mountain Wave: {effects['in_mountain_wave']}")
            print(f"  Vertical Wind: {effects['vertical_wind']:.2f} m/s")
        
        if 'radar_visible' in effects:
            print(f"  Radar Visible: {effects['radar_visible']}")


def test_tactical_routing(battlespace):
    """Test tactical path planning."""
    print("\n" + "="*60)
    print("Testing Tactical Routing")
    print("="*60)
    
    # Define mission
    start_pos = np.array([5000, 5000, 1500])
    target_pos = np.array([45000, 45000, 2000])
    threat_pos = np.array([25000, 25000, 5000])  # Threat in center
    
    print(f"Start: {start_pos}")
    print(f"Target: {target_pos}")
    print(f"Threat: {threat_pos}")
    
    # Find direct path
    print("\nDirect Path:")
    direct_path = battlespace.find_optimal_intercept_path(start_pos, target_pos)
    print(f"  Waypoints: {len(direct_path)}")
    
    # Check exposure
    exposed_points = 0
    for wp in direct_path:
        if battlespace.get_line_of_sight(threat_pos, wp):
            exposed_points += 1
    print(f"  Exposed to threat: {exposed_points}/{len(direct_path)} waypoints")
    
    # Find terrain masking route
    print("\nTerrain Masking Route:")
    masked_path = battlespace.find_terrain_masking_route(start_pos, target_pos, threat_pos)
    print(f"  Waypoints: {len(masked_path)}")
    
    # Check exposure
    exposed_points = 0
    for wp in masked_path:
        if battlespace.get_line_of_sight(threat_pos, wp):
            exposed_points += 1
    print(f"  Exposed to threat: {exposed_points}/{len(masked_path)} waypoints")
    
    # Evaluate tactical advantages at different positions
    print("\nTactical Position Analysis:")
    for name, pos in [('Start', start_pos), ('Mid-point', (start_pos + target_pos)/2), ('Target', target_pos)]:
        advantages = battlespace.get_tactical_advantages(pos)
        print(f"\n{name} Position Advantages:")
        print(f"  Height Advantage: {advantages['height_advantage']:.2f}")
        print(f"  Energy State: {advantages['energy_state']:.2f}")
        print(f"  Escape Routes: {advantages['escape_routes']}/8")
        if 'terrain_masking_nearby' in advantages:
            print(f"  Terrain Masking: {advantages['terrain_masking_nearby']:.2%}")
    
    # Visualize paths
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(battlespace.terrain.elevation, levels=20, cmap='terrain',
                alpha=0.5, extent=[0, battlespace.width, 0, battlespace.height])
    
    # Plot paths
    direct_path_array = np.array(direct_path)
    masked_path_array = np.array(masked_path)
    
    ax1.plot(direct_path_array[:, 0], direct_path_array[:, 1], 
            'r-', linewidth=2, label='Direct Path')
    ax1.plot(masked_path_array[:, 0], masked_path_array[:, 1], 
            'b-', linewidth=2, label='Terrain Masking')
    
    # Mark positions
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax1.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
    ax1.plot(threat_pos[0], threat_pos[1], 'rx', markersize=15, label='Threat')
    
    # Draw threat range
    threat_range = plt.Circle((threat_pos[0], threat_pos[1]), 15000, 
                             fill=False, color='red', linestyle='--', alpha=0.5)
    ax1.add_patch(threat_range)
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Tactical Routing Comparison')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot terrain
    step = 20
    terrain = battlespace.terrain.elevation[::step, ::step]
    nx, ny = terrain.shape[1], terrain.shape[0]
    x = np.linspace(0, battlespace.width, nx)
    y = np.linspace(0, battlespace.height, ny)
    X, Y = np.meshgrid(x, y)
    
    ax2.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.3, linewidth=0)
    
    # Plot 3D paths
    ax2.plot(direct_path_array[:, 0], direct_path_array[:, 1], direct_path_array[:, 2],
            'r-', linewidth=3, label='Direct')
    ax2.plot(masked_path_array[:, 0], masked_path_array[:, 1], masked_path_array[:, 2],
            'b-', linewidth=3, label='Masked')
    
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_zlabel('Altitude (m)')
    ax2.set_title('3D Path Visualization')
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration."""
    print("="*60)
    print("Integrated Battlespace Environment Demo")
    print("="*60)
    
    # Create enhanced battlespace
    battlespace = create_enhanced_battlespace()
    
    # Get info
    info = battlespace.get_info()
    print(f"\nBattlespace Info:")
    print(f"  Dimensions: {info['width']}m x {info['height']}m x {info['altitude_ceiling']}m")
    print(f"  Terrain Range: {info['terrain_range'][0]:.1f}m to {info['terrain_range'][1]:.1f}m")
    
    # Visualize integrated effects
    visualize_integrated_effects(battlespace)
    
    # Test aircraft effects
    test_aircraft_effects(battlespace)
    
    # Test tactical routing
    test_tactical_routing(battlespace)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()