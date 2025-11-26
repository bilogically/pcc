"""
Activation Tracking System for PCC Integration Model

Logs activity of each neural component over time to enable:
- Visualization of activation patterns
- Comparison with empirical fMRI/EEG data
- Validation against known findings (e.g., encoding/retrieval flip)

Usage:
    tracker = ActivationTracker()
    tracker.log("vmPFC", "bid_computation", 0.85, {"region": "region_1", "stake": 0.95})
    ...
    tracker.summarize()
    tracker.plot_activity()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict
import json


class Phase(Enum):
    """Task phases for grouping activation"""
    PRE_STIMULUS = "pre_stimulus"
    VISUAL_PROCESSING = "visual_processing"
    CHAMPION_BIDDING = "champion_bidding"
    TALLYING = "tallying"
    COHERENCE_CHECK = "coherence_check"
    DEADLOCK_FEEDBACK = "deadlock_feedback"
    RESOLUTION = "resolution"
    SELF_INTEGRATION = "self_integration"
    POST_DECISION = "post_decision"


@dataclass
class ActivationEvent:
    """A single activation event from a neural component"""
    step: int                    # Global simulation step
    region: str                  # "vmPFC", "dlPFC", "dACC", "dPCC", "vPCC", "sgACC", etc.
    activity_type: str           # "bid", "route", "tally", "coherence_check", etc.
    magnitude: float             # 0-1 activity level
    phase: Phase                 # Which task phase this occurred in
    iteration: int = 0           # Which iteration of the decision loop (0 = first)
    details: Dict[str, Any] = field(default_factory=dict)  # Context-specific info
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "region": self.region,
            "activity_type": self.activity_type,
            "magnitude": self.magnitude,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "details": self.details
        }


class ActivationTracker:
    """
    Central tracker for all neural activation events.
    
    Provides logging, querying, summarization, and visualization.
    """
    
    def __init__(self):
        self.events: List[ActivationEvent] = []
        self.current_step: int = 0
        self.current_phase: Phase = Phase.PRE_STIMULUS
        self.current_iteration: int = 0
        self.trial_markers: List[Tuple[int, str]] = []  # (step, label)
        
    def reset(self):
        """Clear all events and reset state"""
        self.events = []
        self.current_step = 0
        self.current_phase = Phase.PRE_STIMULUS
        self.current_iteration = 0
        self.trial_markers = []
    
    def new_trial(self, label: str = ""):
        """Mark the start of a new trial"""
        self.trial_markers.append((self.current_step, label))
        self.current_iteration = 0
        self.set_phase(Phase.PRE_STIMULUS)
    
    def set_phase(self, phase: Phase):
        """Set the current task phase"""
        self.current_phase = phase
    
    def set_iteration(self, iteration: int):
        """Set the current decision loop iteration"""
        self.current_iteration = iteration
    
    def tick(self):
        """Advance the simulation step"""
        self.current_step += 1
    
    def log(self, region: str, activity_type: str, magnitude: float, 
            details: Optional[Dict] = None):
        """
        Log an activation event.
        
        Args:
            region: Neural region identifier (e.g., "vmPFC", "dPCC")
            activity_type: Type of activity (e.g., "bid", "route", "coherence_check")
            magnitude: Activity level 0-1
            details: Optional context-specific information
        """
        event = ActivationEvent(
            step=self.current_step,
            region=region,
            activity_type=activity_type,
            magnitude=magnitude,
            phase=self.current_phase,
            iteration=self.current_iteration,
            details=details or {}
        )
        self.events.append(event)
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_events(self, 
                   region: Optional[str] = None,
                   activity_type: Optional[str] = None,
                   phase: Optional[Phase] = None,
                   iteration: Optional[int] = None) -> List[ActivationEvent]:
        """Filter events by criteria"""
        filtered = self.events
        
        if region is not None:
            filtered = [e for e in filtered if e.region == region]
        if activity_type is not None:
            filtered = [e for e in filtered if e.activity_type == activity_type]
        if phase is not None:
            filtered = [e for e in filtered if e.phase == phase]
        if iteration is not None:
            filtered = [e for e in filtered if e.iteration == iteration]
        
        return filtered
    
    def get_timeseries(self, region: str) -> List[Tuple[int, float]]:
        """
        Get activation timeseries for a region.
        Returns list of (step, magnitude) tuples.
        """
        events = self.get_events(region=region)
        return [(e.step, e.magnitude) for e in events]
    
    def get_region_total(self, region: str) -> float:
        """Get total accumulated activation for a region"""
        events = self.get_events(region=region)
        return sum(e.magnitude for e in events)
    
    def get_region_mean(self, region: str) -> float:
        """Get mean activation for a region"""
        events = self.get_events(region=region)
        if not events:
            return 0.0
        return sum(e.magnitude for e in events) / len(events)
    
    def get_phase_activity(self, phase: Phase) -> Dict[str, float]:
        """Get total activity per region during a specific phase"""
        events = self.get_events(phase=phase)
        activity = defaultdict(float)
        for e in events:
            activity[e.region] += e.magnitude
        return dict(activity)
    
    def get_all_regions(self) -> List[str]:
        """Get list of all regions that have logged activity"""
        return list(set(e.region for e in self.events))
    
    # =========================================================================
    # Summarization Methods
    # =========================================================================
    
    def summarize(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of activation patterns.
        """
        regions = self.get_all_regions()
        
        summary = {
            "total_events": len(self.events),
            "total_steps": self.current_step,
            "num_iterations": self.current_iteration + 1,
            "regions": {},
            "phases": {},
            "by_iteration": {}
        }
        
        # Per-region summary
        for region in regions:
            events = self.get_events(region=region)
            summary["regions"][region] = {
                "total_activation": sum(e.magnitude for e in events),
                "mean_activation": sum(e.magnitude for e in events) / len(events) if events else 0,
                "num_events": len(events),
                "activity_types": list(set(e.activity_type for e in events))
            }
        
        # Per-phase summary
        for phase in Phase:
            summary["phases"][phase.value] = self.get_phase_activity(phase)
        
        # Per-iteration summary
        max_iter = max((e.iteration for e in self.events), default=0)
        for i in range(max_iter + 1):
            iter_events = self.get_events(iteration=i)
            iter_activity = defaultdict(float)
            for e in iter_events:
                iter_activity[e.region] += e.magnitude
            summary["by_iteration"][i] = dict(iter_activity)
        
        return summary
    
    def summarize_compact(self) -> Dict[str, float]:
        """Simple region -> total activation mapping"""
        return {region: self.get_region_total(region) 
                for region in self.get_all_regions()}
    
    def compare_regions(self, region1: str, region2: str) -> Dict[str, Any]:
        """Compare activation between two regions"""
        r1_total = self.get_region_total(region1)
        r2_total = self.get_region_total(region2)
        
        return {
            region1: r1_total,
            region2: r2_total,
            "difference": r1_total - r2_total,
            "ratio": r1_total / r2_total if r2_total > 0 else float('inf'),
            "dominant": region1 if r1_total > r2_total else region2
        }
    
    # =========================================================================
    # Encoding/Retrieval Specific
    # =========================================================================
    
    def compare_phases(self, phase1: Phase, phase2: Phase) -> Dict[str, Dict[str, float]]:
        """
        Compare regional activity between two phases.
        Useful for encoding/retrieval flip analysis.
        """
        p1_activity = self.get_phase_activity(phase1)
        p2_activity = self.get_phase_activity(phase2)
        
        all_regions = set(p1_activity.keys()) | set(p2_activity.keys())
        
        comparison = {}
        for region in all_regions:
            v1 = p1_activity.get(region, 0)
            v2 = p2_activity.get(region, 0)
            comparison[region] = {
                phase1.value: v1,
                phase2.value: v2,
                "difference": v1 - v2,
                "higher_in": phase1.value if v1 > v2 else phase2.value
            }
        
        return comparison
    
    # =========================================================================
    # Visualization (Text-based for now)
    # =========================================================================
    
    def print_summary(self):
        """Print a formatted summary to console"""
        summary = self.summarize()
        
        print("\n" + "=" * 60)
        print("ACTIVATION SUMMARY")
        print("=" * 60)
        print(f"Total events: {summary['total_events']}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Iterations: {summary['num_iterations']}")
        
        print("\n--- Per Region ---")
        for region, data in sorted(summary["regions"].items()):
            print(f"  {region:12s}: total={data['total_activation']:.3f}, "
                  f"mean={data['mean_activation']:.3f}, "
                  f"events={data['num_events']}")
        
        print("\n--- Per Phase ---")
        for phase, activity in summary["phases"].items():
            if activity:
                print(f"  {phase}:")
                for region, value in sorted(activity.items()):
                    print(f"    {region}: {value:.3f}")
        
        print("\n--- Per Iteration ---")
        for iteration, activity in summary["by_iteration"].items():
            if activity:
                total = sum(activity.values())
                print(f"  Iteration {iteration}: total={total:.3f}")
                for region, value in sorted(activity.items()):
                    bar = "█" * int(value * 20)
                    print(f"    {region:12s}: {value:.3f} {bar}")
    
    def print_timeseries(self, region: str, width: int = 50):
        """Print ASCII visualization of activation over time"""
        timeseries = self.get_timeseries(region)
        if not timeseries:
            print(f"No events for {region}")
            return
        
        print(f"\n{region} Activation Over Time:")
        print("-" * (width + 20))
        
        max_step = max(t[0] for t in timeseries)
        
        # Aggregate by step if multiple events per step
        step_activity = defaultdict(float)
        for step, mag in timeseries:
            step_activity[step] += mag
        
        for step in range(max_step + 1):
            mag = step_activity.get(step, 0)
            bar_len = int(mag * width)
            bar = "█" * bar_len + "░" * (width - bar_len)
            print(f"  Step {step:3d}: [{bar}] {mag:.2f}")
    
    def print_iteration_comparison(self):
        """Show how activity changes across iterations"""
        summary = self.summarize()
        regions = self.get_all_regions()
        
        print("\n--- Iteration Comparison ---")
        print(f"{'Region':12s}", end="")
        for i in summary["by_iteration"]:
            print(f"  Iter {i:2d}", end="")
        print()
        print("-" * (12 + 9 * len(summary["by_iteration"])))
        
        for region in sorted(regions):
            print(f"{region:12s}", end="")
            for i, activity in summary["by_iteration"].items():
                val = activity.get(region, 0)
                print(f"  {val:6.3f}", end="")
            print()
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def to_json(self) -> str:
        """Export all events as JSON"""
        return json.dumps({
            "events": [e.to_dict() for e in self.events],
            "trial_markers": self.trial_markers,
            "summary": self.summarize()
        }, indent=2)
    
    def to_csv_rows(self) -> List[Dict]:
        """Export events as list of dicts (for CSV/DataFrame)"""
        return [e.to_dict() for e in self.events]


# =============================================================================
# Global tracker instance (optional convenience)
# =============================================================================

_global_tracker: Optional[ActivationTracker] = None

def get_tracker() -> ActivationTracker:
    """Get or create global tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ActivationTracker()
    return _global_tracker

def reset_tracker():
    """Reset global tracker"""
    global _global_tracker
    _global_tracker = ActivationTracker()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Quick test
    tracker = ActivationTracker()
    
    # Simulate a simple trial
    tracker.new_trial("test_trial")
    
    tracker.set_phase(Phase.VISUAL_PROCESSING)
    tracker.log("Visual", "scene_processing", 0.8, {"objects": ["person", "danger"]})
    tracker.tick()
    
    tracker.set_phase(Phase.CHAMPION_BIDDING)
    tracker.set_iteration(0)
    tracker.log("vmPFC", "bid_computation", 0.95, {"region": "region_1"})
    tracker.log("vmPFC", "bid_computation", 0.3, {"region": "region_2"})
    tracker.log("dlPFC", "bid_computation", 0.4, {"region": "region_1"})
    tracker.log("dlPFC", "bid_computation", 0.85, {"region": "region_2"})
    tracker.tick()
    
    tracker.set_phase(Phase.TALLYING)
    tracker.log("dPCC", "tally", 0.7, {"totals": {"r1": 1.35, "r2": 1.15}})
    tracker.tick()
    
    tracker.set_phase(Phase.COHERENCE_CHECK)
    tracker.log("dACC", "coherence_check", 0.5, {"status": "deadlock"})
    tracker.tick()
    
    # Iteration 2
    tracker.set_phase(Phase.CHAMPION_BIDDING)
    tracker.set_iteration(1)
    tracker.log("vmPFC", "bid_computation", 0.98, {"region": "region_1"})
    tracker.log("vmPFC", "bid_computation", 0.25, {"region": "region_2"})
    tracker.log("dlPFC", "bid_computation", 0.35, {"region": "region_1"})
    tracker.log("dlPFC", "bid_computation", 0.80, {"region": "region_2"})
    tracker.tick()
    
    tracker.set_phase(Phase.TALLYING)
    tracker.log("dPCC", "tally", 0.75, {"totals": {"r1": 1.33, "r2": 1.05}})
    tracker.tick()
    
    tracker.set_phase(Phase.COHERENCE_CHECK)
    tracker.log("dACC", "coherence_check", 0.8, {"status": "resolved"})
    tracker.tick()
    
    tracker.set_phase(Phase.SELF_INTEGRATION)
    tracker.log("vPCC", "self_relevance", 0.75, {"score": 0.75})
    tracker.log("sgACC", "gatekeeper", 0.6, {"arousal": 0.3, "emotional_tag": 0.6})
    tracker.tick()
    
    # Print results
    tracker.print_summary()
    tracker.print_iteration_comparison()
    tracker.print_timeseries("vmPFC")
