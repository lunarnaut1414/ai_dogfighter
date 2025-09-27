# 3DOF Fixed-Wing Aircraft Simulation Development Guide

## Overview
This guide outlines the development of a 3DOF (Three Degree of Freedom) fixed-wing aircraft simulation system that integrates with the existing battlespace environment. The 3DOF model tracks position (x, y, z) while simplifying rotational dynamics into instantaneous heading and flight path angle changes.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Phase 1: 3DOF Dynamics Model](#phase-1-3dof-dynamics-model)
3. [Phase 2: Asset Manager Implementation](#phase-2-asset-manager-implementation)
4. [Phase 3: Flight Controller & Behaviors](#phase-3-flight-controller--behaviors)
5. [Phase 4: Scenario System](#phase-4-scenario-system)
6. [Phase 5: Integration Points](#phase-5-integration-points)
7. [Phase 6: Testing Strategy](#phase-6-testing-strategy)
8. [Implementation Priority](#implementation-priority)
9. [Key Design Decisions](#key-design-decisions)
10. [Next Steps](#next-steps)

## Architecture Overview

### Project Structure
```
interceptor_guidance/
├── configs/
│   ├── aircraft/
│   │   ├── interceptor_drone.yaml     # Your drone configuration
│   │   ├── target_basic.yaml          # Basic target aircraft
│   │   └── target_fighter.yaml        # Advanced target
│   └── scenarios/
│       ├── single_target.yaml         # 1v1 engagement
│       ├── multi_target.yaml          # 1vN engagement
│       └── swarm_defense.yaml         # Complex scenario
├── src/
│   ├── assets/                        # NEW: Aircraft simulation
│   │   ├── __init__.py
│   │   ├── asset_manager.py           # Central state management
│   │   ├── aircraft_3dof.py           # 3DOF dynamics model
│   │   ├── flight_controller.py       # Autopilot/behaviors
│   │   └── sensor_model.py            # Detection/tracking
│   └── simulation/
│       ├── environment.py             # MODIFY: Integrate assets
│       └── scenarios.py               # MODIFY: Use asset manager
```

### System Components
- **Asset Manager**: Central registry and state propagation for all aircraft
- **3DOF Aircraft Model**: Simplified fixed-wing dynamics
- **Flight Controller**: Autopilot and behavior implementation
- **Sensor Model**: Detection and tracking with realistic limitations
- **Scenario System**: Configuration-driven simulation setup

## Phase 1: 3DOF Dynamics Model

### 1.1 State Vector Definition
For 3DOF fixed-wing aircraft, we track:
- **Position**: [x, y, z] in battlespace coordinates (meters)
- **Velocity**: Magnitude V (true airspeed in m/s)
- **Flight Path Angles**: 
  - γ (gamma): Climb angle (radians)
  - ψ (psi): Heading angle (radians)
- **Bank Angle**: φ (phi): Bank angle for coordinated turns (radians)

### 1.2 Equations of Motion
The 3DOF point-mass dynamics with flight path angles:

```
Position Dynamics:
dx/dt = V * cos(γ) * cos(ψ)
dy/dt = V * cos(γ) * sin(ψ)  
dz/dt = V * sin(γ)

Velocity Dynamics:
dV/dt = (T - D)/m - g*sin(γ)

Flight Path Dynamics:
dγ/dt = (L*cos(φ) - W*cos(γ))/(m*V)
dψ/dt = (L*sin(φ))/(m*V*cos(γ))
```

Where:
- **T** = Thrust force (N)
- **D** = Drag force (N)
- **L** = Lift force (N)
- **W** = Weight = m*g (N)
- **m** = Mass (kg)
- **g** = Gravity (9.81 m/s²)
- **φ** = Bank angle (control input)

### 1.3 Force Models

#### Lift Force
```
L = 0.5 * ρ * V² * S * CL
CL = CL_alpha * α  (simplified, α = angle of attack)
```

#### Drag Force
```
D = 0.5 * ρ * V² * S * CD
CD = CD0 + k * CL²  (parabolic drag polar)
```

#### Thrust Model
```
T = throttle * T_max
throttle ∈ [0, 1]
```

### 1.4 Configuration Schema
```yaml
# configs/aircraft/interceptor_drone.yaml
aircraft:
  name: "Interceptor MQ-X"
  type: "fixed_wing_3dof"
  
  # Mass Properties
  mass: 150.0  # kg
  
  # Aerodynamic Properties
  aerodynamics:
    reference_area: 2.5  # m^2 (wing area)
    cd0: 0.025          # Parasitic drag coefficient
    k: 0.04             # Induced drag factor
    cl_alpha: 5.0       # Lift curve slope (per radian)
    cl_max: 1.4         # Maximum lift coefficient
    
  # Propulsion
  propulsion:
    thrust_max: 500.0   # N
    thrust_min: 0.0     # N
    sfc: 0.0001        # Specific fuel consumption (kg/N/s)
    
  # Performance Envelope
  performance:
    v_min: 20.0        # Stall speed (m/s)
    v_max: 80.0        # Max speed (m/s)  
    v_cruise: 50.0     # Cruise speed (m/s)
    climb_rate_max: 10.0  # m/s
    turn_rate_max: 0.5    # rad/s
    load_factor_max: 4.0  # g's
    service_ceiling: 10000.0  # meters
    
  # Control Limits
  control:
    bank_angle_max: 60.0  # degrees
    bank_rate_max: 90.0   # degrees/second
    pitch_angle_max: 20.0  # degrees
    throttle_rate: 0.5     # 1/s (0 to 1 in 2 seconds)
    
  # Initial Fuel/Battery
  fuel:
    capacity: 20.0      # kg or kWh
    initial: 20.0       # kg or kWh
    
  # Sensor Suite
  sensors:
    radar:
      max_range: 10000.0  # meters
      fov_azimuth: 120.0  # degrees
      fov_elevation: 30.0  # degrees
    infrared:
      max_range: 5000.0   # meters
      fov: 60.0           # degrees
```

## Phase 2: Asset Manager Implementation

### 2.1 Core Responsibilities
The Asset Manager serves as the central authority for all aircraft in the simulation:

1. **State Management**
   - Maintain registry of all aircraft
   - Store current and historical states
   - Handle asset creation/destruction

2. **Time Propagation**
   - Synchronous updates at fixed timestep
   - Apply dynamics models
   - Enforce flight envelope constraints

3. **Environmental Integration**
   - Query battlespace for conditions
   - Apply wind and turbulence
   - Check terrain collisions

4. **Spatial Operations**
   - Efficient proximity queries
   - Collision detection
   - Line-of-sight calculations

### 2.2 Integration with Battlespace

```python
class AssetManager:
    """
    Central manager for all aircraft assets in the simulation.
    Integrates with battlespace for environmental effects.
    """
    
    def __init__(self, battlespace: Battlespace, dt: float = 0.02):
        self.battlespace = battlespace
        self.dt = dt  # 50 Hz update rate
        self.assets = {}  # Dict[str, Aircraft3DOF]
        self.time = 0.0
        self.asset_history = {}  # Store trajectories
        
    def spawn_aircraft(self, config: dict, asset_id: str) -> Aircraft3DOF:
        """Create and register a new aircraft"""
        # Load aircraft configuration
        # Set initial state
        # Register in spatial index
        # Return aircraft instance
        
    def update(self):
        """Propagate all assets one timestep"""
        for asset_id, asset in self.assets.items():
            # Get environmental conditions at aircraft position
            wind = self.battlespace.get_wind(asset.position)
            density = self.battlespace.weather.get_density_at(asset.position[2])
            turbulence = self.battlespace.weather.get_turbulence_at(asset.position)
            
            # Check terrain
            terrain_height = self.battlespace.get_elevation(
                asset.position[0], asset.position[1]
            )
            
            # Get all environmental effects
            env_effects = self.battlespace.get_aircraft_environment_effects(
                asset.position, asset.get_velocity_vector()
            )
            
            # Propagate dynamics with environmental conditions
            asset.update(self.dt, env_effects)
            
            # Check for terrain collision
            if asset.position[2] <= terrain_height + 10:
                asset.handle_terrain_collision()
            
            # Store history for analysis
            self._record_state(asset_id, asset.state)
            
        self.time += self.dt
    
    def get_assets_in_range(self, position: np.ndarray, range: float) -> List[str]:
        """Find all assets within range of position"""
        # Use spatial indexing for efficiency
        
    def get_asset_state(self, asset_id: str) -> AircraftState:
        """Get current state of an asset"""
        
    def get_relative_state(self, from_id: str, to_id: str) -> RelativeState:
        """Get relative position, velocity, angles between assets"""
```

### 2.3 Efficient Spatial Queries

Leverage spatial indexing for performance with many assets:

```python
class SpatialIndex:
    """R-tree based spatial index for aircraft"""
    
    def __init__(self):
        from rtree import index
        self.idx = index.Index(properties=self._get_properties())
        
    def insert(self, asset_id: str, position: np.ndarray):
        """Add/update asset position"""
        
    def query_range(self, position: np.ndarray, range: float) -> List[str]:
        """Find all assets within range"""
        
    def nearest_k(self, position: np.ndarray, k: int) -> List[str]:
        """Find k nearest assets"""
```

## Phase 3: Flight Controller & Behaviors

### 3.1 Basic Autopilot Architecture

```python
class FlightController:
    """
    Basic flight control for autonomous aircraft.
    Provides different behavior modes for targets.
    """
    
    def __init__(self, aircraft_config: dict):
        self.config = aircraft_config
        self.mode = 'waypoint'  # Current behavior mode
        self.waypoints = []
        self.current_waypoint_idx = 0
        
        # Control gains (simplified)
        self.k_heading = 1.0  # Heading control gain
        self.k_altitude = 0.1  # Altitude control gain
        self.k_speed = 0.05   # Speed control gain
    
    def compute_commands(self, state: AircraftState, 
                        target: Optional[np.ndarray] = None) -> ControlCommand:
        """
        Compute bank angle and throttle commands based on mode.
        """
        if self.mode == 'waypoint':
            return self.waypoint_guidance(state, self.waypoints[self.current_waypoint_idx])
        elif self.mode == 'orbit':
            return self.orbit_guidance(state, target)
        elif self.mode == 'evade':
            return self.evasive_guidance(state, target)
        elif self.mode == 'pursuit':
            return self.pursuit_guidance(state, target)
            
    def waypoint_guidance(self, state: AircraftState, 
                         waypoint: np.ndarray) -> ControlCommand:
        """Navigate to waypoint"""
        # Compute desired heading to waypoint
        # Compute desired altitude
        # Compute speed command
        # Return bank angle and throttle
        
    def orbit_guidance(self, state: AircraftState, 
                      center: np.ndarray) -> ControlCommand:
        """Orbit around a point"""
        
    def evasive_guidance(self, state: AircraftState, 
                        threat: np.ndarray) -> ControlCommand:
        """Evade from threat"""
```

### 3.2 Target Behavior Progression

Progressive complexity levels for target aircraft:

#### Level 0: Static/Predictable
- **Straight Line**: Constant velocity, heading, altitude
- **Orbit**: Circle at fixed radius
- **Figure-8**: Predictable pattern

#### Level 1: Waypoint Navigation
- **Sequential Waypoints**: Navigate through points
- **Looping Patrol**: Repeat waypoint sequence
- **Altitude Changes**: Climb/descent between waypoints

#### Level 2: Reactive Behaviors
- **Simple Evasion**: Break turn when threatened
- **Speed Changes**: Accelerate when pursued
- **Altitude Escape**: Climb/dive to evade

#### Level 3: Tactical Behaviors
- **Energy Management**: Trade altitude for speed
- **Terrain Masking**: Use terrain for cover
- **Coordinated Evasion**: Barrel rolls, split-S

#### Level 4: Intelligent Behaviors
- **Predictive Evasion**: Anticipate intercept
- **Optimal Escape**: Calculate best escape route
- **Cooperative Tactics**: Multiple target coordination

### 3.3 Behavior State Machine

```yaml
# Target behavior configuration
behavior:
  type: "state_machine"
  initial_state: "patrol"
  
  states:
    patrol:
      type: "waypoint"
      waypoints: [[10000, 10000, 2000], [20000, 20000, 2000]]
      speed: 40.0
      transitions:
        threat_detected: "evade"
        fuel_low: "rtb"
        
    evade:
      type: "evasive"
      maneuver: "break_turn"
      g_limit: 3.0
      transitions:
        threat_clear: "patrol"
        fuel_critical: "rtb"
        
    rtb:
      type: "waypoint"
      waypoints: [[5000, 5000, 1000]]
      speed: 50.0
      transitions:
        landed: "complete"
```

## Phase 4: Scenario System

### 4.1 Scenario Configuration Schema

```yaml
# configs/scenarios/single_target.yaml
scenario:
  name: "Basic Intercept Training"
  description: "Single target interception in clear weather"
  battlespace: "default_battlespace.yaml"
  
  # Time and Weather
  time_of_day: 12.0  # Noon
  weather_preset: "clear"  # clear, cloudy, stormy
  wind_override:
    base_vector: [5, 0, 0]  # 5 m/s from west
  
  # Interceptor (controlled by guidance algorithm)
  interceptor:
    aircraft: "interceptor_drone.yaml"
    initial_state:
      position: [5000, 5000, 2000]
      velocity: 50.0
      heading: 0.0  # North (radians)
      fuel_fraction: 1.0
    guidance_mode: "autonomous"  # autonomous or manual
      
  # Target Aircraft
  targets:
    - id: "bandit_1"
      aircraft: "target_basic.yaml"
      behavior: "waypoint"
      threat_level: "hostile"
      initial_state:
        position: [25000, 25000, 2500]
        velocity: 40.0
        heading: 3.14159  # South
      waypoints:
        - [25000, 20000, 2500]
        - [20000, 15000, 2000]
        - [15000, 10000, 2000]
      evasion_trigger:
        range: 2000  # Start evading when interceptor within 2km
        
  # Success Criteria
  objectives:
    primary:
      - type: "intercept"
        target_id: "bandit_1"
        range: 50  # meters
        time_limit: 300  # seconds
    secondary:
      - type: "fuel_remaining"
        min_fraction: 0.2
      - type: "no_terrain_collision"
        
  # Termination Conditions
  termination:
    conditions:
      - interceptor_fuel_empty
      - all_targets_neutralized
      - time_limit_exceeded
      - terrain_collision
    time_limit: 600  # seconds
    
  # Data Recording
  recording:
    enabled: true
    frequency: 10  # Hz
    include:
      - aircraft_states
      - guidance_commands
      - sensor_tracks
      - environmental_conditions
```

### 4.2 Multi-Target Scenario

```yaml
# configs/scenarios/multi_target.yaml
scenario:
  name: "Multi-Target Engagement"
  
  targets:
    - id: "primary_threat"
      aircraft: "target_fighter.yaml"
      behavior: "aggressive"
      threat_level: "hostile"
      initial_state:
        position: [30000, 30000, 3000]
        
    - id: "secondary_threat"
      aircraft: "target_basic.yaml"
      behavior: "evasive"
      threat_level: "hostile"
      initial_state:
        position: [20000, 35000, 2000]
        
    - id: "decoy"
      aircraft: "target_basic.yaml"
      behavior: "patrol"
      threat_level: "unknown"
      initial_state:
        position: [25000, 20000, 2500]
        
  objectives:
    primary:
      - type: "prioritize_threats"
        engagement_order: ["primary_threat", "secondary_threat"]
      - type: "time_to_intercept"
        max_time: 240
```

### 4.3 Scenario Execution Engine

```python
class ScenarioRunner:
    """
    Executes scenarios and manages simulation flow.
    """
    
    def __init__(self, scenario_config: dict, guidance_algorithm):
        self.config = scenario_config
        self.guidance = guidance_algorithm
        
        # Initialize battlespace
        self.battlespace = Battlespace(
            config_file=scenario_config['battlespace']
        )
        self.battlespace.generate()
        
        # Create asset manager
        self.asset_manager = AssetManager(self.battlespace)
        
        # Metrics tracking
        self.metrics = SimulationMetrics()
        
    def setup(self):
        """Initialize scenario"""
        # Spawn interceptor
        self.interceptor_id = self.asset_manager.spawn_aircraft(
            self.config['interceptor'],
            "interceptor"
        )
        
        # Spawn targets
        self.target_ids = []
        for target_config in self.config['targets']:
            target_id = self.asset_manager.spawn_aircraft(
                target_config,
                target_config['id']
            )
            self.target_ids.append(target_id)
            
    def run(self) -> SimulationResults:
        """Execute scenario"""
        while not self.check_termination():
            # Get sensor observations
            sensor_data = self.get_sensor_observations()
            
            # Compute guidance commands
            guidance_cmd = self.guidance.compute(
                self.asset_manager.get_asset_state(self.interceptor_id),
                sensor_data,
                self.battlespace
            )
            
            # Apply commands
            self.asset_manager.apply_commands(
                self.interceptor_id,
                guidance_cmd
            )
            
            # Update all assets
            self.asset_manager.update()
            
            # Record metrics
            self.metrics.record(self.asset_manager.time)
            
            # Check objectives
            self.evaluate_objectives()
            
        return self.metrics.generate_report()
```

## Phase 5: Integration Points

### 5.1 Guidance Algorithm Interface

#### Input Data Structure
```python
@dataclass
class GuidanceInput:
    """Input data provided to guidance algorithm each cycle"""
    
    # Own ship state
    own_state: AircraftState
    
    # Sensor tracks (may include uncertainty)
    target_tracks: List[TargetTrack]
    
    # Environmental awareness
    environment: EnvironmentInfo
    
    # Mission parameters
    mission: MissionParameters
    
@dataclass
class AircraftState:
    """Complete 3DOF aircraft state"""
    position: np.ndarray  # [x, y, z] meters
    velocity: float       # True airspeed (m/s)
    heading: float        # radians
    climb_angle: float    # radians
    bank_angle: float     # radians
    fuel_remaining: float # kg or fraction
    
@dataclass
class TargetTrack:
    """Target information from sensors"""
    track_id: str
    position: np.ndarray
    position_uncertainty: np.ndarray  # Covariance
    velocity: np.ndarray
    velocity_uncertainty: np.ndarray
    threat_level: str  # hostile, unknown, friendly
    time_since_update: float
    
@dataclass
class EnvironmentInfo:
    """Battlespace information"""
    wind_vector: np.ndarray
    terrain_elevation: float
    minimum_safe_altitude: float
    no_fly_zones: List[NoFlyZone]
    threat_zones: List[ThreatZone]
```

#### Output Command Structure
```python
@dataclass
class GuidanceCommand:
    """Commands from guidance algorithm to flight control"""
    
    # Navigation commands (pick one mode)
    mode: str  # 'direct', 'waypoint', 'velocity'
    
    # Direct control mode
    bank_angle_cmd: Optional[float]  # radians
    throttle_cmd: Optional[float]    # [0, 1]
    
    # Waypoint mode
    target_position: Optional[np.ndarray]
    target_velocity: Optional[float]
    
    # Velocity vector mode
    velocity_vector_cmd: Optional[np.ndarray]
    
    # Status/Intent
    guidance_phase: str  # 'search', 'approach', 'terminal', 'evade'
    time_to_intercept: Optional[float]
```

### 5.2 Sensor Model Integration

```python
class SensorModel:
    """
    Realistic sensor model with detection probability and measurement noise.
    """
    
    def __init__(self, sensor_config: dict):
        self.max_range = sensor_config['max_range']
        self.fov_azimuth = sensor_config['fov_azimuth']
        self.fov_elevation = sensor_config['fov_elevation']
        self.update_rate = sensor_config.get('update_rate', 10)  # Hz
        
    def get_detections(self, own_state: AircraftState, 
                       true_targets: List[Aircraft3DOF],
                       battlespace: Battlespace) -> List[TargetTrack]:
        """
        Generate sensor detections with realistic effects.
        """
        detections = []
        
        for target in true_targets:
            # Check range
            range_to_target = np.linalg.norm(
                target.position - own_state.position
            )
            if range_to_target > self.max_range:
                continue
                
            # Check field of view
            if not self.in_fov(own_state, target):
                continue
                
            # Check line of sight (terrain masking)
            if not battlespace.get_line_of_sight(
                own_state.position, target.position
            ):
                continue
                
            # Detection probability based on range
            p_detect = self.detection_probability(range_to_target)
            if np.random.random() > p_detect:
                continue
                
            # Add measurement noise
            measured_position = target.position + np.random.randn(3) * 10
            measured_velocity = target.velocity + np.random.randn(3) * 2
            
            detections.append(TargetTrack(
                track_id=target.id,
                position=measured_position,
                velocity=measured_velocity,
                # ... uncertainty, etc
            ))
            
        return detections
```

### 5.3 Battlespace Integration Examples

```python
# In Aircraft3DOF class

def apply_environmental_effects(self, battlespace: Battlespace):
    """
    Apply all environmental effects from battlespace.
    """
    # Get comprehensive environmental effects
    effects = battlespace.get_aircraft_environment_effects(
        self.position, 
        self.get_velocity_vector()
    )
    
    # Wind effects
    self.wind = effects['wind_vector']
    self.ground_speed = np.linalg.norm(self.velocity_vector)
    self.air_velocity = self.velocity_vector - self.wind
    self.true_airspeed = np.linalg.norm(self.air_velocity)
    
    # Density effects on aerodynamics
    self.air_density = effects['air_density']
    density_ratio = effects['density_ratio']
    
    # Turbulence
    if effects['turbulence_intensity'] > 0:
        # Add turbulence perturbations
        turb = effects['turbulence_perturbation']
        self.velocity_vector += turb * self.dt
        
    # Ground effect (increased lift near terrain)
    if effects['ground_effect_factor'] > 1.0:
        self.lift_multiplier = effects['ground_effect_factor']
        
def check_terrain_constraints(self, battlespace: Battlespace):
    """
    Ensure aircraft respects terrain and airspace.
    """
    # Minimum safe altitude
    min_alt = battlespace.get_minimum_safe_altitude(
        self.position[0], self.position[1],
        radius=500, safety_margin=50
    )
    
    if self.position[2] < min_alt:
        # Terrain avoidance response
        self.commanded_altitude = min_alt + 100
        
    # No-fly zones
    if not battlespace.airspace.is_position_valid(self.position):
        # Compute escape vector
        self.enter_escape_mode()
```

## Phase 6: Testing Strategy

### 6.1 Unit Tests

```python
# tests/test_aircraft_3dof.py

def test_energy_conservation():
    """Verify total energy is conserved in absence of thrust/drag"""
    aircraft = Aircraft3DOF(config)
    initial_energy = aircraft.kinetic_energy + aircraft.potential_energy
    
    # Disable thrust and drag
    aircraft.thrust = 0
    aircraft.drag = 0
    
    # Simulate for 10 seconds
    for _ in range(500):  # 0.02s timestep
        aircraft.update(0.02)
        
    final_energy = aircraft.kinetic_energy + aircraft.potential_energy
    assert abs(final_energy - initial_energy) < 0.01  # Numerical tolerance
    
def test_turn_radius():
    """Verify turn radius matches theory"""
    # R = V²/(g*tan(φ))
    
def test_climb_performance():
    """Verify climb rate within limits"""
    
def test_stall_detection():
    """Verify stall at low speeds"""
```

### 6.2 Integration Tests

```python
def test_multi_aircraft_simulation():
    """Test asset manager with multiple aircraft"""
    battlespace = create_test_battlespace()
    manager = AssetManager(battlespace)
    
    # Spawn 10 aircraft
    for i in range(10):
        manager.spawn_aircraft(config, f"aircraft_{i}")
        
    # Run for 60 seconds
    for _ in range(3000):
        manager.update()
        
    # Verify all aircraft updated
    # Check no collisions
    # Verify performance metrics
    
def test_environmental_integration():
    """Test wind and turbulence effects"""
    
def test_sensor_detection_chain():
    """Test sensor model with occlusion"""
```

### 6.3 Performance Benchmarks

```python
def benchmark_scaling():
    """Test performance with increasing aircraft count"""
    results = {}
    
    for n_aircraft in [10, 50, 100, 200]:
        manager = create_manager_with_aircraft(n_aircraft)
        
        start_time = time.perf_counter()
        for _ in range(1000):  # 20 seconds at 50Hz
            manager.update()
        elapsed = time.perf_counter() - start_time
        
        results[n_aircraft] = {
            'total_time': elapsed,
            'time_per_update': elapsed / 1000,
            'updates_per_second': 1000 / elapsed
        }
        
    return results
```

### 6.4 Scenario Validation

```python
def test_intercept_geometry():
    """Validate basic intercept scenarios"""
    scenarios = [
        "head_on_intercept",
        "tail_chase",
        "beam_attack",
        "high_altitude_dive"
    ]
    
    for scenario_name in scenarios:
        result = run_scenario(f"test_scenarios/{scenario_name}.yaml")
        assert result.objectives_met
        assert result.time_to_intercept < result.time_limit
```

## Implementation Priority

### Week 1: Core Dynamics
1. **Implement `aircraft_3dof.py`**
   - Basic 3DOF equations of motion
   - Force calculations (lift, drag, thrust)
   - State propagation
   - Unit tests for dynamics

2. **Create aircraft configurations**
   - Interceptor drone config
   - Basic target config
   - Validate performance numbers

### Week 2: Asset Management
3. **Build `asset_manager.py`**
   - Asset registry and spawning
   - Single aircraft update
   - Environmental integration
   - Multi-aircraft scaling

4. **Integrate with battlespace**
   - Wind effects
   - Terrain queries
   - Collision detection

### Week 3: Control & Behaviors
5. **Implement `flight_controller.py`**
   - Waypoint following
   - Basic autopilot
   - Command limiting

6. **Add target behaviors**
   - Level 0-1 behaviors
   - State machine structure

### Week 4: Integration & Testing
7. **Create scenario system**
   - Config loading
   - Scenario runner
   - Metrics collection

8. **Implement sensor model**
   - Range/FOV checking
   - Detection probability
   - Measurement noise

9. **Integration testing**
   - Full simulation loop
   - Performance optimization
   - Bug fixes

### Week 5: Refinement
10. **Advanced features**
    - Level 2+ behaviors
    - Sensor fusion
    - Performance tuning

## Key Design Decisions

### Why 3DOF?
- **Sufficient Fidelity**: Captures essential fixed-wing dynamics for trajectory analysis
- **Computational Efficiency**: Can simulate 50+ aircraft at 50Hz on Jetson
- **Easier Validation**: Simpler to tune and verify against known physics
- **Appropriate Abstraction**: Matches the control authority of guidance (heading, speed, altitude)

### Integration Philosophy
- **Reuse Existing Code**: Leverage battlespace environmental systems fully
- **Consistent Conventions**: Same coordinate system, units, and patterns
- **Modular Design**: Each component can be tested independently
- **Configuration-Driven**: Easy to create new aircraft and scenarios

### Performance Targets
- **Update Rate**: 50Hz minimum for all assets
- **Scalability**: 50+ simultaneous aircraft
- **Latency**: < 10ms for guidance loop
- **Memory**: < 100MB for 50 aircraft with 60s history

### Simplifications & Assumptions
- **Instantaneous Bank**: No roll dynamics (assumes coordinated turns)
- **Thrust Response**: Simplified throttle model (no engine lag)
- **Perfect Control**: No actuator dynamics or control surface modeling
- **Point Mass**: No moments of inertia or rotational dynamics

## Next Steps

1. **Review Configuration Schema**
   - Validate aircraft parameters are realistic
   - Ensure all needed parameters are included

2. **Implement Core 3DOF Class**
   - Start with basic dynamics
   - Add environmental effects incrementally

3. **Create Test Suite**
   - Unit tests for physics validation
   - Integration tests for battlespace interaction

4. **Build Asset Manager**
   - Start with single aircraft
   - Scale to multiple aircraft

5. **Implement Simple Scenario**
   - Single target, waypoint following
   - Test full simulation loop

6. **Add Guidance Interface**
   - Define clear API for your guidance algorithm
   - Create mock guidance for testing

7. **Progressive Enhancement**
   - Add sensor model
   - Implement target behaviors
   - Optimize performance

## Appendix A: Reference Parameters

### Typical Small UAV Parameters
- **Mass**: 50-200 kg
- **Wing Area**: 2-5 m²
- **Cruise Speed**: 30-60 m/s
- **Stall Speed**: 15-25 m/s
- **Max Thrust**: 200-1000 N
- **L/D Ratio**: 10-15
- **Service Ceiling**: 5000-15000 m
- **Endurance**: 2-8 hours

### Control Response Times
- **Bank Angle**: 30-90 deg/s
- **Throttle**: 0.2-1.0 s (idle to max)
- **Speed Change**: 2-5 m/s²

### Sensor Characteristics
- **Radar Range**: 5-20 km
- **IR Range**: 2-10 km
- **Position Error**: 10-50 m
- **Velocity Error**: 1-5 m/s
- **Update Rate**: 1-10 Hz

## Appendix B: Coordinate Conventions

Following the existing battlespace conventions:
- **Origin**: Southwest corner (0, 0, 0)
- **X-Axis**: East (positive right)
- **Y-Axis**: North (positive up on map)
- **Z-Axis**: Altitude (positive up)
- **Units**: All distances in meters
- **Angles**: All angles in radians
- **Heading**: 0 = North, π/2 = East