"""
PCC Integration Model - Toy Implementation (v3: with Activation Tracking)

Models the dual PCC architecture for decision-making and self-integration.

COGNITIVE PATHWAY (dPCC circuit):
- Visual → dPCC (routes to champions)
- Champions (vmPFC, dlPFC) bid on options  
- dPCC tallies bids
- dACC checks coherence (deadlock detection)
- If resolved → motor output + self-integration

SELF-INTEGRATION PATHWAY (vPCC-sgACC circuit):
- Resolved decision → vPCC (self-relevance assessment)
- vPCC → sgACC (gatekeeper to limbic/autonomic)
- sgACC outputs to:
  - Hippocampus (encode as episodic memory)
  - Amygdala (emotional tagging)
  - Hypothalamus (autonomic: HR, cortisol)
  - Back to vPCC (update self-model)

Key anatomical facts:
- vPCC has DIRECT reciprocal connections with sgACC
- dPCC does NOT connect to sgACC - connects to premotor/MCC
- sgACC is the "gatekeeper" between cognition and emotion/autonomic

NEW IN V3: Full activation tracking for empirical validation
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from activation_tracker import ActivationTracker, Phase


class CoherenceStatus(Enum):
    RESOLVED = "resolved"
    DEADLOCK = "deadlock"


@dataclass
class VisualRegion:
    """A boxed region in the visual scene"""
    region_id: str
    objects: List[str]  # e.g., ["person_loved", "danger_state"]


@dataclass
class VisualInput:
    """What the visual system sends to PCC"""
    regions: List[VisualRegion]
    scene_structure: str = "two_boxed_regions"


@dataclass 
class ChampionBid:
    """A bid from a champion region on an option"""
    champion_id: str
    region_id: str
    stake: float  # 0.0 to 1.0 - how much they're willing to spend


@dataclass
class IntegrationState:
    """The current state being routed through PCC"""
    visual: Optional[VisualInput] = None
    bids: List[ChampionBid] = field(default_factory=list)
    coherence: Optional[CoherenceStatus] = None
    winner: Optional[str] = None
    self_integration: Optional['sgACCOutput'] = None  # Output from vPCC-sgACC pathway


# =============================================================================
# VISUAL SYSTEM
# =============================================================================

class VisualSystem:
    """
    Visual processing system.
    Parses scene, extracts objects and structure.
    """
    
    def __init__(self, tracker: Optional[ActivationTracker] = None):
        self.tracker = tracker
    
    def process_scene(self, scene: VisualInput) -> VisualInput:
        """
        Process visual scene and log activation.
        In reality this would do complex processing; here we just pass through.
        """
        if self.tracker:
            # Activation proportional to scene complexity
            complexity = sum(len(r.objects) for r in scene.regions)
            magnitude = min(1.0, complexity * 0.2)
            
            self.tracker.log(
                "Visual", 
                "scene_processing", 
                magnitude,
                {
                    "num_regions": len(scene.regions),
                    "total_objects": complexity,
                    "structure": scene.scene_structure
                }
            )
        
        return scene


# =============================================================================
# CHAMPIONS - Brain regions that bid on options
# =============================================================================

class Champion:
    """Base class for brain regions that bid on options"""
    
    def __init__(self, champion_id: str, value_map: Dict[str, float], 
                 tracker: Optional[ActivationTracker] = None):
        self.champion_id = champion_id
        self.value_map = value_map  # object_type -> base value
        self.last_bid = None
        self.competing_bid = None  # feedback from last round
        self.tracker = tracker
    
    def compute_region_value(self, region: VisualRegion) -> float:
        """Compute base value for a region based on its objects"""
        total = 0.0
        for obj in region.objects:
            total += self.value_map.get(obj, 0.0)
        return total
    
    def generate_bid(self, visual: VisualInput, deadlock_feedback: bool = False) -> List[ChampionBid]:
        """Generate bids for each region"""
        bids = []
        
        # Compute values for each region
        region_values = {}
        for region in visual.regions:
            region_values[region.region_id] = self.compute_region_value(region)
        
        # Convert to stakes (normalized, with some noise)
        max_val = max(region_values.values()) if region_values else 1.0
        if max_val == 0:
            max_val = 1.0
            
        for region_id, value in region_values.items():
            base_stake = value / max_val if max_val > 0 else 0.0
            
            # If deadlock feedback, adjust based on competition
            if deadlock_feedback and self.last_bid is not None:
                # Either raise (fight harder) or back off (concede)
                adjustment = random.uniform(-0.15, 0.2)  # slight bias toward raising
                base_stake = max(0.0, min(1.0, base_stake + adjustment))
            
            # Add small noise
            noise = random.uniform(-0.05, 0.05)
            final_stake = max(0.0, min(1.0, base_stake + noise))
            
            bids.append(ChampionBid(
                champion_id=self.champion_id,
                region_id=region_id,
                stake=final_stake
            ))
            
            # Log activation for this bid
            if self.tracker:
                self.tracker.log(
                    self.champion_id,
                    "bid_computation",
                    final_stake,
                    {
                        "region": region_id,
                        "raw_value": value,
                        "deadlock_feedback": deadlock_feedback
                    }
                )
        
        self.last_bid = bids
        return bids


class vmPFC(Champion):
    """
    Ventromedial PFC - champions personal/emotional value
    High value for: loved ones, self-relevant, emotionally salient
    """
    def __init__(self, tracker: Optional[ActivationTracker] = None):
        value_map = {
            "person_loved": 0.95,
            "person_familiar": 0.7,
            "person_stranger": 0.3,
            "danger_state": -0.2,  # danger to loved ones hurts more
            "safety_state": 0.1,
        }
        super().__init__("vmPFC", value_map, tracker)


class dlPFC(Champion):
    """
    Dorsolateral PFC - champions utilitarian/abstract moral value
    High value for: maximizing outcomes, following rules
    """
    def __init__(self, tracker: Optional[ActivationTracker] = None):
        value_map = {
            "person_loved": 0.3,      # one person is one person
            "person_familiar": 0.3,
            "person_stranger": 0.3,
            "multiple_persons": 0.8,  # more lives = more value (utilitarian)
            "danger_state": -0.1,
            "safety_state": 0.1,
        }
        super().__init__("dlPFC", value_map, tracker)


# =============================================================================
# SELF-INTEGRATION PATHWAY (vPCC-sgACC circuit)
# =============================================================================

@dataclass
class SelfModel:
    """
    The accumulated self-template stored in vPCC.
    Contains history of integrated decisions and their outcomes.
    """
    choice_history: List[Dict] = field(default_factory=list)
    autonomic_baseline: float = 0.5  # resting arousal level
    value_associations: Dict[str, float] = field(default_factory=dict)
    
    def similarity_to_past(self, choice_features: Dict) -> float:
        """How similar is this choice to past self-defining choices?"""
        if not self.choice_history:
            return 0.0
        
        # Simple: count how many past choices involved similar objects
        similarities = []
        for past in self.choice_history:
            overlap = len(set(choice_features.get('objects', [])) & 
                        set(past.get('objects', [])))
            similarities.append(overlap / max(len(choice_features.get('objects', [])), 1))
        
        return sum(similarities) / len(similarities) if similarities else 0.0


@dataclass 
class sgACCOutput:
    """What sgACC sends to downstream targets"""
    to_hippocampus: Dict  # episodic encoding signal
    to_amygdala: float    # emotional tag (-1 to 1)
    to_hypothalamus: float  # autonomic arousal signal
    to_vPCC: Dict         # self-model update


class sgACC:
    """
    Subgenual Anterior Cingulate Cortex
    
    The "gatekeeper" between cognitive and emotional/autonomic systems.
    Receives from vPCC, outputs to:
    - Hippocampus (episodic memory encoding)
    - Amygdala (emotional tagging)
    - Hypothalamus (autonomic: heart rate, cortisol)
    - Back to vPCC (self-model update)
    """
    
    def __init__(self, tracker: Optional[ActivationTracker] = None):
        self.arousal_sensitivity = 0.3
        self.tracker = tracker
    
    def process(self, decision: Dict, self_relevance: float) -> sgACCOutput:
        """
        Process a resolved decision for emotional/autonomic integration.
        
        Args:
            decision: The resolved choice {winner, bids, visual, ...}
            self_relevance: 0-1 score from vPCC
        """
        # More self-relevant decisions get stronger encoding
        encoding_strength = self_relevance
        
        # Emotional tag based on which champion won
        # If vmPFC-preferred option won, positive; if dlPFC-preferred, more neutral
        emotional_tag = 0.0
        if 'winning_champion' in decision:
            if decision['winning_champion'] == 'vmPFC':
                emotional_tag = 0.6  # warm, personal
            elif decision['winning_champion'] == 'dlPFC':
                emotional_tag = 0.2  # cooler, principled
        
        # Autonomic arousal proportional to how contested the decision was
        arousal = decision.get('conflict_level', 0.5) * self.arousal_sensitivity
        
        # Log sgACC activity
        if self.tracker:
            # sgACC activation reflects gatekeeper function
            # Higher when self-relevant and emotionally significant
            magnitude = (self_relevance + abs(emotional_tag) + arousal) / 3
            self.tracker.log(
                "sgACC",
                "gatekeeper_processing",
                magnitude,
                {
                    "self_relevance": self_relevance,
                    "emotional_tag": emotional_tag,
                    "arousal": arousal,
                    "encoding_strength": encoding_strength
                }
            )
            
            # Also log downstream targets
            self.tracker.log("Hippocampus", "encoding", encoding_strength, 
                           {"choice": decision.get('winner')})
            self.tracker.log("Amygdala", "emotional_tag", abs(emotional_tag),
                           {"valence": emotional_tag})
            self.tracker.log("Hypothalamus", "autonomic", arousal,
                           {"type": "arousal"})
        
        return sgACCOutput(
            to_hippocampus={
                'encode': True,
                'strength': encoding_strength,
                'context': decision.get('visual', {}),
                'choice': decision.get('winner')
            },
            to_amygdala=emotional_tag,
            to_hypothalamus=arousal,
            to_vPCC={
                'update_self_model': True,
                'choice_features': {
                    'objects': decision.get('choice_objects', []),
                    'valence': emotional_tag,
                    'self_relevance': self_relevance
                }
            }
        )


class vPCC:
    """
    Ventral Posterior Cingulate Cortex
    
    The self-relevance assessor and self-model holder.
    - Receives resolved decisions from dPCC/dACC pathway
    - Assesses self-relevance against stored self-template
    - Routes to sgACC for limbic/autonomic integration
    - Receives updates back from sgACC to update self-model
    
    Key anatomical fact: vPCC has DIRECT reciprocal connection with sgACC.
    This connection does NOT exist for dPCC.
    """
    
    def __init__(self, tracker: Optional[ActivationTracker] = None):
        self.self_model = SelfModel()
        self.sgacc = sgACC(tracker)
        self.tracker = tracker
    
    def assess_self_relevance(self, decision: Dict) -> float:
        """
        How self-relevant is this decision?
        
        Factors:
        - Involves self-relevant objects (loved ones, possessions)
        - Similar to past self-defining choices
        - High emotional stakes
        """
        base_relevance = 0.3  # all choices have some self-relevance
        
        # Check for self-relevant objects
        objects = decision.get('choice_objects', [])
        self_relevant_objects = {'person_loved', 'person_familiar', 'self', 'home'}
        object_relevance = len(set(objects) & self_relevant_objects) * 0.2
        
        # Check similarity to past choices
        past_similarity = self.self_model.similarity_to_past({'objects': objects})
        
        # Combine
        total = min(1.0, base_relevance + object_relevance + past_similarity * 0.3)
        
        # Log the self-relevance assessment
        if self.tracker:
            self.tracker.log(
                "vPCC",
                "self_relevance_assessment",
                total,
                {
                    "base": base_relevance,
                    "object_relevance": object_relevance,
                    "past_similarity": past_similarity,
                    "objects": objects
                }
            )
        
        return total
    
    def integrate_decision(self, decision: Dict) -> sgACCOutput:
        """
        Main entry point: take a resolved decision and integrate into self.
        
        This is called AFTER dACC says RESOLVED.
        """
        # Step 1: Assess self-relevance
        self_relevance = self.assess_self_relevance(decision)
        
        # Step 2: Route to sgACC
        sgacc_output = self.sgacc.process(decision, self_relevance)
        
        # Step 3: Update self-model with feedback from sgACC
        if sgacc_output.to_vPCC.get('update_self_model'):
            self.self_model.choice_history.append(
                sgacc_output.to_vPCC['choice_features']
            )
            
            # Log the self-model update
            if self.tracker:
                self.tracker.log(
                    "vPCC",
                    "self_model_update",
                    0.5,  # constant for update operation
                    {
                        "history_length": len(self.self_model.choice_history),
                        "new_entry": sgacc_output.to_vPCC['choice_features']
                    }
                )
        
        return sgacc_output


# =============================================================================
# dACC - Deadlock Detector (cognitive pathway)
# =============================================================================

class dACC:
    """
    Dorsal Anterior Cingulate Cortex - detects deadlocks in bidding
    
    Part of the COGNITIVE pathway (dPCC circuit).
    Simple threshold check: is there a clear winner?
    
    Note: This is distinct from sgACC which handles emotional/autonomic.
    dACC connects to dPCC and dlPFC (cognitive control).
    sgACC connects to vPCC and limbic structures.
    """
    
    def __init__(self, threshold: float = 0.15, tracker: Optional[ActivationTracker] = None):
        self.threshold = threshold
        self.tracker = tracker
    
    def check_coherence(self, region_totals: Dict[str, float]) -> Tuple[CoherenceStatus, Optional[str]]:
        """
        Check if there's a clear winner among regions.
        Returns (status, winner_region_id or None)
        """
        if not region_totals:
            return CoherenceStatus.DEADLOCK, None
        
        sorted_regions = sorted(region_totals.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_regions) < 2:
            status = CoherenceStatus.RESOLVED
            winner = sorted_regions[0][0]
            gap = sorted_regions[0][1]  # no comparison, use value
        else:
            top_region, top_value = sorted_regions[0]
            second_region, second_value = sorted_regions[1]
            gap = top_value - second_value
            
            if gap >= self.threshold:
                status = CoherenceStatus.RESOLVED
                winner = top_region
            else:
                status = CoherenceStatus.DEADLOCK
                winner = None
        
        # Log dACC activity
        if self.tracker:
            # dACC activation higher when conflict is detected (close race)
            # Inverse of gap: more conflict when gap is small
            conflict_signal = 1.0 - min(1.0, gap / 0.5)  # normalize
            
            self.tracker.log(
                "dACC",
                "coherence_check",
                conflict_signal if status == CoherenceStatus.DEADLOCK else 0.3,
                {
                    "status": status.value,
                    "gap": gap,
                    "threshold": self.threshold,
                    "winner": winner,
                    "totals": region_totals
                }
            )
        
        return status, winner


# =============================================================================
# dPCC - The Cognitive Router
# =============================================================================

class dPCC:
    """
    Dorsal Posterior Cingulate Cortex - content-agnostic router
    
    Part of the COGNITIVE pathway.
    Jobs:
    1. Receive visual input
    2. Route to champions for bidding
    3. Tally bids per region
    4. Send to dACC for coherence check
    5. If deadlock, iterate with feedback
    6. When resolved, hand off to vPCC for self-integration
    
    Key anatomical fact: dPCC connects to premotor areas and dACC/MCC.
    It does NOT have direct connection to sgACC (that's vPCC's job).
    """
    
    def __init__(self, champions: List[Champion], dacc: dACC, vpcc: vPCC, 
                 tracker: Optional[ActivationTracker] = None,
                 max_iterations: int = 10):
        self.champions = champions
        self.dacc = dacc
        self.vpcc = vpcc
        self.tracker = tracker
        self.max_iterations = max_iterations
    
    def tally_bids(self, bids: List[ChampionBid]) -> Dict[str, float]:
        """Sum all champion stakes per region"""
        totals = {}
        for bid in bids:
            if bid.region_id not in totals:
                totals[bid.region_id] = 0.0
            totals[bid.region_id] += bid.stake
        return totals
    
    def integrate(self, visual: VisualInput, verbose: bool = False) -> IntegrationState:
        """
        Main integration loop.
        Routes visual to champions, collects bids, checks for deadlock, iterates.
        When resolved, hands off to vPCC for self-integration.
        """
        state = IntegrationState(visual=visual)
        
        # Log visual input receipt
        if self.tracker:
            self.tracker.set_phase(Phase.VISUAL_PROCESSING)
            self.tracker.log(
                "dPCC",
                "receive_visual",
                0.5,
                {"num_regions": len(visual.regions)}
            )
            self.tracker.tick()
        
        deadlock_feedback = False
        
        for iteration in range(self.max_iterations):
            if self.tracker:
                self.tracker.set_iteration(iteration)
                self.tracker.set_phase(Phase.CHAMPION_BIDDING)
            
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Collect bids from all champions
            all_bids = []
            for champion in self.champions:
                bids = champion.generate_bid(visual, deadlock_feedback=deadlock_feedback)
                all_bids.extend(bids)
                
                if verbose:
                    for bid in bids:
                        print(f"  {bid.champion_id} bids {bid.stake:.3f} on {bid.region_id}")
            
            state.bids = all_bids
            
            # Tally bids per region
            if self.tracker:
                self.tracker.set_phase(Phase.TALLYING)
            
            totals = self.tally_bids(all_bids)
            
            # Log tallying
            if self.tracker:
                self.tracker.log(
                    "dPCC",
                    "tally_bids",
                    0.6,
                    {"totals": totals}
                )
                self.tracker.tick()
            
            if verbose:
                print(f"  Totals: {totals}")
            
            # Check coherence via dACC
            if self.tracker:
                self.tracker.set_phase(Phase.COHERENCE_CHECK)
            
            status, winner = self.dacc.check_coherence(totals)
            state.coherence = status
            
            if self.tracker:
                self.tracker.tick()
            
            if verbose:
                print(f"  dACC: {status.value}" + (f" -> {winner}" if winner else ""))
            
            if status == CoherenceStatus.RESOLVED:
                state.winner = winner
                
                # Determine which champion "won" (whose preferred option was chosen)
                winner_champion = self._determine_winning_champion(all_bids, winner)
                
                # Calculate conflict level (how contested was this?)
                conflict_level = self._calculate_conflict_level(totals)
                
                # Get objects involved in winning choice
                choice_objects = []
                for region in visual.regions:
                    if region.region_id == winner:
                        choice_objects = region.objects
                        break
                
                # Log resolution
                if self.tracker:
                    self.tracker.set_phase(Phase.RESOLUTION)
                    self.tracker.log(
                        "dPCC",
                        "resolution",
                        0.8,
                        {
                            "winner": winner,
                            "winning_champion": winner_champion,
                            "iterations": iteration + 1
                        }
                    )
                    self.tracker.tick()
                
                # === HAND OFF TO vPCC FOR SELF-INTEGRATION ===
                if self.tracker:
                    self.tracker.set_phase(Phase.SELF_INTEGRATION)
                
                decision_package = {
                    'winner': winner,
                    'winning_champion': winner_champion,
                    'conflict_level': conflict_level,
                    'choice_objects': choice_objects,
                    'visual': visual,
                    'iterations': iteration + 1
                }
                
                sgacc_output = self.vpcc.integrate_decision(decision_package)
                state.self_integration = sgacc_output
                
                if self.tracker:
                    self.tracker.tick()
                
                if verbose:
                    print(f"\n  >>> vPCC-sgACC integration:")
                    print(f"      Self-relevance: {self.vpcc.assess_self_relevance(decision_package):.2f}")
                    print(f"      Emotional tag: {sgacc_output.to_amygdala:.2f}")
                    print(f"      Autonomic arousal: {sgacc_output.to_hypothalamus:.2f}")
                
                return state
            
            # Deadlock - signal feedback for next round
            if self.tracker:
                self.tracker.set_phase(Phase.DEADLOCK_FEEDBACK)
                self.tracker.log(
                    "dPCC",
                    "deadlock_feedback",
                    0.4,
                    {"iteration": iteration}
                )
                self.tracker.tick()
            
            deadlock_feedback = True
        
        # Max iterations reached without resolution
        if verbose:
            print(f"\n  Max iterations reached - forcing decision")
        
        # Force pick the leader even if below threshold
        totals = self.tally_bids(state.bids)
        winner = max(totals.items(), key=lambda x: x[1])[0]
        state.winner = winner
        state.coherence = CoherenceStatus.RESOLVED
        
        return state
    
    def _determine_winning_champion(self, bids: List[ChampionBid], winner: str) -> str:
        """Which champion had the highest stake on the winning option?"""
        winner_bids = [b for b in bids if b.region_id == winner]
        if not winner_bids:
            return "unknown"
        best = max(winner_bids, key=lambda b: b.stake)
        return best.champion_id
    
    def _calculate_conflict_level(self, totals: Dict[str, float]) -> float:
        """How contested was this decision? (0 = easy, 1 = very contested)"""
        if len(totals) < 2:
            return 0.0
        sorted_vals = sorted(totals.values(), reverse=True)
        gap = sorted_vals[0] - sorted_vals[1]
        # Normalize: gap of 0 = max conflict, gap of 1+ = no conflict
        return max(0.0, 1.0 - gap)


# =============================================================================
# VISUAL SYSTEM - Generates scene input
# =============================================================================

def create_moral_dilemma_scene() -> VisualInput:
    """
    Create a classic trolley-problem-like visual scene.
    Region 1: One person you love in danger
    Region 2: Five strangers in danger
    """
    return VisualInput(
        regions=[
            VisualRegion(
                region_id="region_1",
                objects=["person_loved", "danger_state"]
            ),
            VisualRegion(
                region_id="region_2", 
                objects=["multiple_persons", "person_stranger", "danger_state"]
            )
        ],
        scene_structure="two_boxed_regions"
    )


def create_easy_scene() -> VisualInput:
    """
    An easy choice - strangers vs loved one, no moral weight difference
    """
    return VisualInput(
        regions=[
            VisualRegion(
                region_id="region_1",
                objects=["person_loved", "safety_state"]
            ),
            VisualRegion(
                region_id="region_2",
                objects=["person_stranger", "safety_state"]
            )
        ],
        scene_structure="two_boxed_regions"
    )


# =============================================================================
# MAIN - Run the model with activation tracking
# =============================================================================

def main():
    print("=" * 60)
    print("PCC INTEGRATION MODEL (v3: with Activation Tracking)")
    print("=" * 60)
    
    # Create activation tracker
    tracker = ActivationTracker()
    
    # Create the system with tracker passed to all components
    visual_system = VisualSystem(tracker)
    vmpfc = vmPFC(tracker)
    dlpfc = dlPFC(tracker)
    dacc = dACC(threshold=0.15, tracker=tracker)
    vpcc = vPCC(tracker)
    dpcc = dPCC(champions=[vmpfc, dlpfc], dacc=dacc, vpcc=vpcc, 
                tracker=tracker, max_iterations=10)
    
    # Test 1: Easy choice
    print("\n\n### TEST 1: Easy Choice (loved one vs stranger) ###")
    tracker.new_trial("easy_choice")
    visual = create_easy_scene()
    visual = visual_system.process_scene(visual)
    print(f"Scene: {visual}")
    result = dpcc.integrate(visual, verbose=True)
    print(f"\nFINAL: Winner = {result.winner}")
    print(f"Self-model now has {len(vpcc.self_model.choice_history)} stored choices")
    
    # Test 2: Moral dilemma
    print("\n\n### TEST 2: Moral Dilemma (1 loved one vs 5 strangers) ###")
    tracker.new_trial("moral_dilemma")
    visual = create_moral_dilemma_scene()
    visual = visual_system.process_scene(visual)
    print(f"Scene: {visual}")
    result = dpcc.integrate(visual, verbose=True)
    print(f"\nFINAL: Winner = {result.winner}")
    print(f"Self-model now has {len(vpcc.self_model.choice_history)} stored choices")
    
    # Print activation summary
    print("\n")
    tracker.print_summary()
    
    # Show iteration comparison
    tracker.print_iteration_comparison()
    
    # Show specific region timeseries
    print("\n--- Key Region Timeseries ---")
    tracker.print_timeseries("dPCC", width=40)
    tracker.print_timeseries("vPCC", width=40)
    tracker.print_timeseries("dACC", width=40)


if __name__ == "__main__":
    main()
