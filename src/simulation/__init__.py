"""
Simulation module for interceptor guidance system.
"""

from .environment import SimulationEnvironment
from .objectives import (
    Objective,
    ObjectiveType,
    ObjectiveStatus,
    ObjectiveResult,
    InterceptObjective,
    FuelEfficiencyObjective,
    TimeEfficiencyObjective,
    SurvivalObjective,
    NoCollisionObjective,
    ReachWaypointObjective,
    MaintainAltitudeObjective,
    InterceptEvent
)
from .objective_manager import ObjectiveManager
from .scenario_runner import ScenarioRunner, ScenarioState, ScenarioMetrics
from .scenario_visualizer import ScenarioVisualizer
from .dynamics import DynamicsModel, DynamicsState, DynamicsType
from .sensors import (
    SensorType,
    Detection,
    SensorConfig,
    SensorBase,
    Radar,
    InfraredSensor,
    PerfectSensor,
    SensorSuite
)

__all__ = [
    # Environment
    'SimulationEnvironment',
    
    # Objectives
    'Objective',
    'ObjectiveType', 
    'ObjectiveStatus',
    'ObjectiveResult',
    'InterceptObjective',
    'FuelEfficiencyObjective',
    'TimeEfficiencyObjective',
    'SurvivalObjective',
    'NoCollisionObjective',
    'ReachWaypointObjective',
    'MaintainAltitudeObjective',
    'InterceptEvent',
    'ObjectiveManager',
    
    # Scenario
    'ScenarioRunner',
    'ScenarioState',
    'ScenarioMetrics',
    'ScenarioVisualizer',
    
    # Dynamics
    'DynamicsModel',
    'DynamicsState',
    'DynamicsType',
    
    # Sensors
    'SensorType',
    'Detection',
    'SensorConfig',
    'SensorBase',
    'Radar',
    'InfraredSensor',
    'PerfectSensor',
    'SensorSuite'
]