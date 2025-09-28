#!/usr/bin/env python3
"""
Test the complete Phase 4 scenario system implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.scenario_runner import ScenarioRunner
from src.simulation.scenario_visualizer import ScenarioVisualizer
from src.guidance_core.guidance_laws import ProportionalNavigation
import numpy as np


class TestGuidance:
    """Simple test guidance for scenario validation"""
    
    def compute_commands(self, interceptor_state, target_states, battlespace):
        """Basic PN guidance to closest target"""
        if not target_states:
            return {
                'commanded_heading': 0,
                'commanded_altitude': 2000,
                'commanded_throttle': 0.5
            }
            
        # Find closest target
        interceptor_pos = interceptor_state['position']
        min_range = float('inf')
        
        for target_id, target in target_states.items():
            r = np.linalg.norm(np.array(target['position']) - np.array(interceptor_pos))
            if r < min_range:
                min_range = r
                closest = target
                
        # Compute heading to target
        dx = closest['position'][0] - interceptor_pos[0]
        dy = closest['position'][1] - interceptor_pos[1]
        commanded_heading = np.arctan2(dy, dx)
        
        return {
            'commanded_heading': commanded_heading,
            'commanded_altitude': closest['position'][2],
            'commanded_throttle': 0.8
        }


def main():
    print("Testing Phase 4 Scenario System...")
    
    # Test single target scenario
    runner = ScenarioRunner(
        scenario_config='configs/scenarios/single_target.yaml',
        guidance_algorithm=TestGuidance(),
        realtime=False,
        verbose=True,
        record=True
    )
    
    runner.setup()
    results = runner.run()
    
    print(f"\nScenario completed!")
    print(f"Duration: {results['duration']:.1f}s")
    print(f"Intercepts: {results['metrics']['intercepts']}")
    print(f"Update rate: {1000/results['metrics']['mean_update_time_ms']:.1f} Hz")
    

if __name__ == '__main__':
    main()