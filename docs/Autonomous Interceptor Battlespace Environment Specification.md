# Autonomous Interceptor Battlespace Environment Specification

## Project Overview
Building a scalable air combat simulation environment for testing autonomous interceptor drone guidance algorithms. The system focuses on fixed-wing aircraft engagements with configurable difficulty levels and procedurally generated terrain.

## Current Development Status

### ✅ Completed Design Decisions
- **Aircraft Types**: Fixed-wing interceptor vs fixed-wing targets (initially 1v1, scalable to NvM)
- **Coordinate System**: Origin at SW corner, X=East, Y=North, Z=Altitude (meters)
- **Default Battlespace**: 50km x 50km x 15km altitude ceiling
- **Terrain Generation**: Perlin noise-based procedural generation
- **Architecture**: Modular, scalable, configuration-driven design

### 🚧 In Progress
- [ ] Core terrain system implementation
- [ ] Basic battlespace class structure
- [ ] Configuration system setup

### 📋 Future Phases
- [ ] Structure/building system
- [ ] Weather and wind fields
- [ ] Multi-aircraft support
- [ ] ROS2 integration

---

## Battlespace Environment Architecture

### Core Components

#### 1. **Coordinate System**
- **Origin**: Southwest corner (0, 0, 0)
- **Axes**: X=East, Y=North, Z=Altitude (up)
- **Units**: Meters throughout
- **Default Size**: 50km x 50km x 15km altitude ceiling
- **Grid Resolution**: 100m (configurable)

#### 2. **Terrain System**
- **Generation Method**: Perlin noise (multi-octave)
- **Features**:
  - Configurable frequency, amplitude, octaves, seed
  - Terrain type classification by elevation (water, grass, rock, snow)
  - Grid-based heightfield with bilinear interpolation
  - Gradient and normal calculations for tactical analysis
  - Line-of-sight calculations for sensor masking

#### 3. **Structure System** (Future-Ready)
- **Design**: Placeholder architecture for Phase 2
- **Features**:
  - Base Structure class with position and bounding box
  - Spatial indexing for collision detection
  - Support for cities, airbases, SAM sites
  - Modular building asset system

#### 4. **Airspace Management**
- **Features**:
  - Configurable altitude ceiling
  - 3D no-fly zones
  - Threat coverage zones (SAM sites)
  - Minimum safe altitude calculations
  - Terrain masking for radar/sensors

#### 5. **Environmental Effects**
- **Features**:
  - 3D wind field with altitude variation
  - Turbulence modeling
  - Atmospheric density effects
  - Time-of-day system (future)

---

## Target Difficulty Levels

### Behavioral Complexity Progression

| Level | Name | Behavior | Key Features |
|-------|------|----------|--------------|
| 0 | Training Dummy | Stationary/Orbit | No reactions, perfect predictability |
| 1 | Straight Line | Constant velocity | No evasion, maintains altitude |
| 2 | Basic Maneuvering | Gentle turns | Predetermined patterns, speed variations |
| 3 | Evasive (Reactive) | Threat awareness | Barrel rolls, split-S, break turns |
| 4 | Advanced Evasive | Predictive evasion | Complex maneuver chains, energy management |
| 5 | Counter-Engagement | Offensive capability | BFM, position advantage, energy traps |
| 6 | Expert Adversary | Full ACM | ML-based tactics, team coordination |
| 7+ | Augmented | Superhuman | Perfect optimization, swarm tactics |

---

## File Structure

```
interceptor_guidance/
├── configs/
│   ├── battlespace/
│   │   ├── default_battlespace.yaml    # 50x50km standard
│   │   ├── small_battlespace.yaml      # 10x10km testing
│   │   ├── mountain_valley.yaml        # Specific terrain
│   │   └── urban_scenario.yaml         # Future city env
│
├── src/
│   ├── battlespace/                    # Environment module
│   │   ├── __init__.py
│   │   ├── core.py                     # Main Battlespace class
│   │   ├── terrain.py                  # TerrainLayer class
│   │   ├── structures.py               # StructureLayer class
│   │   ├── airspace.py                 # AirspaceLayer class
│   │   ├── weather.py                  # WeatherSystem class
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   ├── terrain_generator.py    # Perlin noise impl
│   │   │   ├── structure_generator.py  # Building placement
│   │   │   └── weather_generator.py    # Wind fields
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── spatial_index.py        # Quadtree/R-tree
│   │       ├── interpolation.py        # Smooth queries
│   │       └── coordinate_utils.py     # Transforms
│   │
│   ├── simulation/                     # Modified existing
│   │   ├── environment.py              # Imports Battlespace
│   │   ├── dynamics.py                 # 6DOF aircraft model
│   │   ├── renderer.py                 # Add terrain rendering
│   │   └── scenarios.py                # Use battlespace configs
│   │
│   └── visualization/                  # New viz module
│       ├── __init__.py
│       ├── battlespace_renderer.py     # 2D/3D rendering
│       ├── trajectory_plotter.py       # Path visualization
│       └── heatmap_generator.py        # Analysis overlays
│
├── tests/
│   └── unit/
│       └── battlespace/                # Test directory
│           ├── test_terrain.py
│           ├── test_structures.py
│           └── test_airspace.py
│
└── data/
    ├── terrain/                        # Cached terrain
    │   ├── heightmaps/
    │   └── masks/
    └── structures/                     # Building templates
        ├── buildings.json
        └── airbases.json
```

---

## Key Interfaces

### Battlespace Query API
```python
# Core queries for aircraft simulation
get_elevation(x, y) -> float
check_collision(position, radius) -> bool
get_wind(position) -> Vector3
is_valid_position(position) -> bool
get_line_of_sight(pos1, pos2) -> bool
get_minimum_safe_altitude(x, y, margin) -> float
```

### Configuration Schema
```yaml
battlespace:
  dimensions:
    width: 50000              # meters
    height: 50000             # meters
    altitude_ceiling: 15000   # meters
    grid_resolution: 100      # meters per cell
    
  terrain:
    generator: "perlin"
    seed: 42
    parameters:
      octaves: 6
      frequency: 0.0001
      amplitude: 2000         # max elevation
      base_elevation: 100
    
  structures:
    enabled: false            # Phase 2
    
  weather:
    wind:
      base_vector: [10, 0, 0] # m/s
      altitude_multiplier: 1.5
```

---

## Implementation Roadmap

### Phase 1: Core Terrain (Current)
- [x] Design terrain generation approach
- [ ] Implement Battlespace core class
- [ ] Implement TerrainLayer with Perlin noise
- [ ] Add elevation interpolation
- [ ] Create basic visualization
- [ ] Write unit tests

### Phase 2: Structures & Collision
- [ ] Implement Structure base class
- [ ] Add spatial indexing (Quadtree)
- [ ] Implement collision detection
- [ ] Add urban area generator
- [ ] Create airbase templates

### Phase 3: Environmental Systems
- [ ] Implement wind field generation
- [ ] Add turbulence modeling
- [ ] Create weather system
- [ ] Add atmospheric effects

### Phase 4: Aircraft Integration
- [ ] Integrate with dynamics model
- [ ] Add sensor line-of-sight
- [ ] Implement terrain following
- [ ] Add emergency landing sites

### Phase 5: Advanced Features
- [ ] Multi-aircraft support (NvM)
- [ ] Dynamic weather
- [ ] Destructible structures
- [ ] Real terrain data import

---

## Design Principles

1. **Modularity**: Each system operates independently
2. **Scalability**: Easy to extend without refactoring
3. **Performance**: Optimized for real-time queries
4. **Configurability**: YAML-driven parameters
5. **Testability**: Comprehensive unit test coverage
6. **Visualization**: Built-in debugging tools

---

## Performance Targets

- **Terrain Query**: < 0.1ms per elevation lookup
- **Collision Check**: < 0.5ms per query
- **LOS Calculation**: < 1ms per check
- **Memory Usage**: < 500MB for 50x50km terrain
- **Generation Time**: < 5s for full battlespace

---

## Next Steps

1. **Immediate**: Create `battlespace/core.py` with basic class structure
2. **Next**: Implement Perlin noise terrain generation
3. **Then**: Add visualization for debugging
4. **Finally**: Integrate with aircraft simulation

---

## Notes & Decisions

- **Terrain Only First**: Buildings/structures deferred to Phase 2
- **Fixed-Wing Focus**: Optimized for aircraft that can't hover
- **Grid-Based**: Regular grid simplifies queries vs. irregular mesh
- **Perlin Noise**: Good balance of realism and performance
- **YAML Config**: Human-readable, version-controllable settings

---

## References

- Original design documents: `autonomous_intercepter_guidance_add.md`
- Development roadmap: `docs/Autonomous Interceptor Guidance System.imd`
- Project structure: Following existing `interceptor_guidance/` layout