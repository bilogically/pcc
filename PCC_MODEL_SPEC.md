# PCC Integration Model - Comprehensive Specification

**Version**: 0.2 (dPCC/vPCC split implemented)
**Last Updated**: Session of Nov 25, 2025
**Status**: Theoretical framework solid, implementation in progress, validation pending

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Core Architecture](#2-core-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Information Flow](#4-information-flow)
5. [Current Implementation](#5-current-implementation)
6. [Planned Additions](#6-planned-additions)
7. [Empirical Validation Targets](#7-empirical-validation-targets)
8. [Open Questions](#8-open-questions)
9. [Connection to Sparse Self-Template Theory](#9-connection-to-sparse-self-template-theory)

---

## 1. Theoretical Foundation

### 1.1 The Core Insight: Identity as Prediction Problem

**Key theoretical move**: Regional identity is NOT frequency, metadata, or arbitrary labeling. Identity IS the shape of the prediction problem the region is solving.

Each brain region exists because it's specialized to predict something specific:
- **vmPFC**: Predicts value-for-self ("how good/bad is this for ME?")
- **dlPFC**: Predicts rule-based/utilitarian outcomes ("what's logically optimal?")
- **dmPFC**: Predicts other minds ("what are they thinking/wanting?")
- **ACC**: Meta-predictor - predicts whether predictions will succeed (coherence)
- **PCC**: Predicts what configurations effectively integrate across regions

The frequency signatures, receptor densities, layer structures, etc. are all IMPLEMENTATIONS of these prediction commitments - emergent properties of the biophysics required to solve each prediction problem.

### 1.2 Hierarchical Composition of Predictions

Predictions combine hierarchically:
- Sub-regions predict dimensions (vmPFC: value, dmPFC: social, dlPFC: rules)
- These combine relationally into higher-order predictions
- Example: vmPFC + dmPFC + dlPFC → PFC-level prediction of "what should I do?"

The higher-order prediction isn't just a sum - it answers a question that REQUIRES all the sub-predictions.

### 1.3 The Parliamentary/Auction Model

Brain regions aren't neutral calculators - they're **champions** with stakes in outcomes.

**Dual nature of value signals**:
1. **Information**: "My computation says this option is worth X"
2. **Stake**: "I (this region) am investing X of my resources to advocate for this option"

This reframes integration as **arbitration between competing agents**, not passive combination of signals.

**Auction dynamics**:
- Champions bid on their preferred options
- If one clearly wins → resolution
- If deadlock → feedback signal ("someone matched your bid")
- Champions naturally adjust bids based on feedback
- Auction iterates until resolution or timeout

**Phenomenological mapping**:
- Easy decisions: One champion dominates, quick resolution
- Hard decisions: Champions deadlock, multiple iterations
- "Feeling torn": The subjective experience of iterated deadlocks
- Variable decision time: Emerges from auction dynamics

### 1.4 The dPCC/vPCC Split

Two anatomically and functionally distinct circuits:

**Cognitive Pathway (dPCC circuit)**:
```
Visual → dPCC (routes) → Champions bid → dPCC (tallies) → dACC (coherence) → Motor output
```
- dPCC: Routes information, tallies bids, content-agnostic
- dACC: Deadlock detector, threshold check
- Connects to: dlPFC, premotor areas, motor systems

**Self-Integration Pathway (vPCC-sgACC circuit)**:
```
Resolved decision → vPCC (self-relevance) → sgACC (gatekeeper) → Limbic/Autonomic
```
- vPCC: Self-relevance assessment, self-model holder
- sgACC: Gatekeeper between cognition and emotion/autonomic
- Outputs to: Hippocampus, Amygdala, Hypothalamus, back to vPCC

**Key anatomical fact**: vPCC has DIRECT reciprocal connections with sgACC. dPCC does NOT have this connection - it connects to premotor/MCC instead.

### 1.5 Pre-loaded Task Context

The task framing (e.g., "you are choosing between these options") comes from dlPFC and is pre-loaded before stimulus presentation.

This context is represented as a **type schema** specifying required output format:
```python
required_output: {
    visual: required,
    value: required,
    coherence: required
}
```

PCC does **type-checking** against this schema and routes to fill missing fields. This is content-agnostic structural operation, not semantic reasoning.

---

## 2. Core Architecture

### 2.1 System Overview

```
                         PRE-LOADED (dlPFC)
                         Task schema: {visual: req, value: req, coherence: req}
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE PATHWAY                                     │
│                                                                              │
│   Visual ──→ dPCC ──→ Champions ──→ dPCC ──→ dACC ──→ RESOLVED?            │
│   System      │         bid          tally    check      │                  │
│               │                                          │                  │
│               │         ←── DEADLOCK feedback ←──────────┤                  │
│               │                                          │                  │
│               └──────────────────────────────────────────┼─→ Motor Output   │
│                                                          │                  │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                                                           ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SELF-INTEGRATION PATHWAY                                 │
│                                                                              │
│                           vPCC ←──────────────────────────────┐             │
│                      (self-relevance)                         │             │
│                            │                                  │             │
│                            ↓                                  │             │
│                          sgACC                                │             │
│                       (gatekeeper)                            │             │
│                            │                                  │             │
│              ┌─────────────┼─────────────┬───────────────┐   │             │
│              ↓             ↓             ↓               ↓   │             │
│         Hippocampus    Amygdala    Hypothalamus      back to vPCC          │
│         (episodic)    (emotional)  (autonomic:       (update               │
│                        tagging)    HR, cortisol)     self-model)           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

1. **Visual system** sends structured scene representation to dPCC
2. **dPCC** type-checks against schema, sees value missing, routes to champions
3. **Champions** (vmPFC, dlPFC) compute values AND stakes, return bids
4. **dPCC** tallies total stakes per option
5. **dACC** checks if gap exceeds threshold
   - If yes → RESOLVED, proceed to output
   - If no → DEADLOCK, signal back to champions, iterate
6. **Motor output** receives winning action
7. **vPCC** receives resolved decision, assesses self-relevance
8. **sgACC** processes for emotional/autonomic integration
9. **Downstream targets** receive appropriate signals
10. **vPCC** updates self-model with feedback from sgACC

---

## 3. Component Specifications

### 3.1 Visual System

**Role**: Parse scene, extract objects and structure, send to dPCC

**Input**: Raw visual scene (in toy model: predefined scene structure)

**Output**:
```python
VisualInput {
    regions: [
        VisualRegion {
            region_id: str,
            objects: List[str]  # e.g., ["person_loved", "danger_state"]
        },
        ...
    ],
    scene_structure: str  # e.g., "two_boxed_regions"
}
```

**Key point**: Visual system does NOT know this is a "choice" - that framing comes from pre-loaded task context. It just reports what it sees: "two boxed regions containing these objects."

**Empirical basis**: IT cortex outputs population vectors (~100 dimensions) representing object identity with invariance (position, size, etc.). We abstract this as categorical object labels.

### 3.2 dPCC (Dorsal Posterior Cingulate Cortex)

**Role**: Content-agnostic router and bid tallier

**Operations**:
1. Receive visual input
2. Type-check against required schema
3. Route to appropriate regions to fill missing fields
4. Collect bids from champions
5. Tally stakes per option
6. Send to dACC for coherence check
7. If deadlock, propagate feedback and iterate
8. If resolved, hand off to motor + vPCC

**Key properties**:
- Does NOT understand content
- Does NOT generate solutions
- Pure structural/routing operations
- Connects to premotor areas, dACC/MCC (NOT sgACC)

**Activation pattern predictions**:
- High during routing/tallying operations
- Active during both encoding and retrieval (but see encoding/retrieval flip)
- Should show activity proportional to number of champions engaged

### 3.3 Champions (vmPFC, dlPFC, others)

**Role**: Compute value AND stake for options in their domain

#### 3.3.1 vmPFC (Ventromedial PFC)

**Prediction problem**: "What is the value of this for ME?"

**Value map** (current implementation):
```python
{
    "person_loved": 0.95,
    "person_familiar": 0.7,
    "person_stranger": 0.3,
    "danger_state": -0.2,
    "safety_state": 0.1,
}
```

**Characteristics**:
- Champions personal/emotional value
- High activation for self-relevant stimuli
- Connects to sgACC (via vPCC pathway for self-integration)

#### 3.3.2 dlPFC (Dorsolateral PFC)

**Prediction problem**: "What is the logically/utilitarian optimal choice?"

**Value map** (current implementation):
```python
{
    "person_loved": 0.3,      # one person = one person
    "person_familiar": 0.3,
    "person_stranger": 0.3,
    "multiple_persons": 0.8,  # utilitarian weighting
    "danger_state": -0.1,
    "safety_state": 0.1,
}
```

**Characteristics**:
- Champions rule-based/utilitarian value
- Domain of abstract moral principles
- Connects to dACC, premotor (cognitive control pathway)

#### 3.3.3 Bid Generation

**Dual signal**:
- **Information**: Computed value based on domain expertise
- **Stake**: Resource investment in advocating for this option

**Bid adjustment on deadlock**:
- Champions receive feedback that their bid was matched
- Naturally adjust (raise to fight harder, or back off)
- Small random component creates variability

```python
def generate_bid(visual, deadlock_feedback=False):
    base_stake = compute_domain_value(visual)

    if deadlock_feedback:
        adjustment = random.uniform(-0.15, 0.2)  # slight bias toward raising
        base_stake += adjustment

    noise = random.uniform(-0.05, 0.05)
    return clamp(0, 1, base_stake + noise)
```

### 3.4 dACC (Dorsal Anterior Cingulate Cortex)

**Role**: Deadlock detector (meta-predictor of coherence)

**Prediction problem**: "Will predictions succeed? Is this configuration coherent?"

**Operation**:
```python
def check_coherence(region_totals: Dict[str, float]) -> (status, winner):
    sorted_regions = sort_by_value(region_totals)
    gap = sorted_regions[0].value - sorted_regions[1].value

    if gap >= threshold:
        return RESOLVED, sorted_regions[0].id
    else:
        return DEADLOCK, None
```

**Key properties**:
- Simple threshold check
- Does NOT know content
- Does NOT generate solutions
- Just detects whether bid gap is sufficient
- Connects to dlPFC, dPCC (cognitive pathway)

**Threshold**: Currently 0.15 (tunable parameter)

### 3.5 vPCC (Ventral Posterior Cingulate Cortex)

**Role**: Self-relevance assessor and self-model holder

**Prediction problem**: "Is this about ME? Does this relate to my self-model?"

**Operations**:
1. Receive resolved decision from cognitive pathway
2. Assess self-relevance against stored self-template
3. Route to sgACC for limbic/autonomic integration
4. Update self-model with feedback from sgACC

**Self-relevance computation** (current implementation):
```python
def assess_self_relevance(decision):
    base_relevance = 0.3  # all choices have some self-relevance

    # Check for self-relevant objects
    object_relevance = count_overlap(decision.objects, SELF_RELEVANT_OBJECTS) * 0.2

    # Check similarity to past choices
    past_similarity = self_model.similarity_to_past(decision)

    return min(1.0, base_relevance + object_relevance + past_similarity * 0.3)
```

**Self-model**:
```python
SelfModel {
    choice_history: List[Dict],  # past integrated decisions
    autonomic_baseline: float,   # resting arousal level
    value_associations: Dict     # learned object-value mappings
}
```

**Key anatomical fact**: vPCC has DIRECT reciprocal connection with sgACC.

### 3.6 sgACC (Subgenual Anterior Cingulate Cortex)

**Role**: Gatekeeper between cognition and emotion/autonomic systems

**Operations**:
1. Receive self-relevance assessment from vPCC
2. Generate outputs to downstream targets:
   - Hippocampus: Episodic encoding signal
   - Amygdala: Emotional tag
   - Hypothalamus: Autonomic arousal signal
   - Back to vPCC: Self-model update

**Output structure**:
```python
sgACCOutput {
    to_hippocampus: {
        encode: bool,
        strength: float,  # proportional to self-relevance
        context: visual_info,
        choice: winner_id
    },
    to_amygdala: float,       # emotional tag (-1 to 1)
    to_hypothalamus: float,   # autonomic arousal
    to_vPCC: {
        update_self_model: bool,
        choice_features: {...}
    }
}
```

**Arousal computation**:
```python
arousal = conflict_level * arousal_sensitivity
# More contested decisions → higher autonomic response
```

**Emotional tag**:
- vmPFC win → warmer/personal tag (0.6)
- dlPFC win → cooler/principled tag (0.2)

---

## 4. Information Flow

### 4.1 Single Trial Flow (Moral Dilemma Example)

**Pre-trial**: dlPFC loads task schema into dPCC
```
required_output: {visual: req, value: req, coherence: req}
task_type: "choice"
agent: "self"
```

**Stimulus onset**: Visual system processes scene
```
VisualInput {
    regions: [
        {id: "region_1", objects: ["person_loved", "danger_state"]},
        {id: "region_2", objects: ["multiple_persons", "person_stranger", "danger_state"]}
    ],
    scene_structure: "two_boxed_regions"
}
```

**Iteration 1**:
1. dPCC receives visual, type-checks: value missing
2. Routes to champions
3. vmPFC bids: region_1 = 0.95, region_2 = 0.15
4. dlPFC bids: region_1 = 0.20, region_2 = 0.90
5. dPCC tallies: region_1 = 1.15, region_2 = 1.05
6. dACC: gap = 0.10 < threshold → DEADLOCK
7. Feedback propagates to champions

**Iteration 2**:
1. Champions adjust bids based on deadlock feedback
2. vmPFC: region_1 = 0.98 (raised), region_2 = 0.12
3. dlPFC: region_1 = 0.25, region_2 = 0.85 (backed off slightly)
4. dPCC tallies: region_1 = 1.23, region_2 = 0.97
5. dACC: gap = 0.26 > threshold → RESOLVED, winner = region_1

**Post-resolution**:
1. Motor output: prepare action for region_1
2. vPCC receives decision package
3. Self-relevance assessment: 0.75 (involves loved one, similar to past choices)
4. sgACC processes:
   - Hippocampus: encode with strength 0.75
   - Amygdala: emotional tag 0.6 (vmPFC won)
   - Hypothalamus: arousal 0.24 (was contested)
   - vPCC: update self-model with choice features

### 4.2 Data Structure Transformations

```
Visual Scene
    ↓ (visual processing)
VisualInput {regions, structure}
    ↓ (champion bidding)
List[ChampionBid] {champion_id, region_id, stake}
    ↓ (dPCC tallying)
Dict[region_id → total_stake]
    ↓ (dACC coherence check)
(CoherenceStatus, winner_id)
    ↓ (if resolved)
DecisionPackage {winner, winning_champion, conflict_level, choice_objects, ...}
    ↓ (vPCC self-relevance)
(decision, self_relevance_score)
    ↓ (sgACC processing)
sgACCOutput {to_hippocampus, to_amygdala, to_hypothalamus, to_vPCC}
```

---

## 5. Current Implementation

### 5.1 Implemented Components

| Component | Status | Notes |
|-----------|--------|-------|
| VisualInput/VisualRegion | ✅ Complete | Basic scene representation |
| Champion base class | ✅ Complete | Bid generation with deadlock adjustment |
| vmPFC | ✅ Complete | Personal/emotional value champion |
| dlPFC | ✅ Complete | Utilitarian/rule-based champion |
| dPCC | ✅ Complete | Router, tallier, iteration loop |
| dACC | ✅ Complete | Deadlock detector |
| vPCC | ✅ Complete | Self-relevance, self-model |
| sgACC | ✅ Complete | Limbic/autonomic gatekeeper |
| SelfModel | ✅ Basic | Stores choice history, similarity computation |
| IntegrationState | ✅ Complete | Tracks state through integration |

### 5.2 Test Scenarios Implemented

1. **Easy choice**: Loved one vs stranger (no moral weight)
   - Expected: Quick resolution (1 iteration)
   - Result: ✅ Works as expected

2. **Moral dilemma**: 1 loved one vs 5 strangers
   - Expected: Multiple iterations, variable outcomes
   - Result: ✅ Shows deadlock iterations, 70-75% loved one wins

3. **Self-model accumulation**: 20 repeated dilemmas
   - Expected: Self-relevance scores should increase as history builds
   - Result: ✅ Observed increase from 0.5 → 0.8

### 5.3 Current Limitations

1. **No activation tracking**: Can't visualize activity patterns over time
2. **Simplified value maps**: Dictionary lookup, not learned associations
3. **No learning/plasticity**: Champions don't update based on outcomes
4. **No time dynamics**: Iterations are discrete, not continuous time
5. **Limited champions**: Only vmPFC and dlPFC, missing amygdala, hippocampus, etc.
6. **No external validation**: Haven't compared to empirical activation patterns

---

## 6. Planned Additions

### 6.1 Activation Tracking System (NEXT PRIORITY)

**Purpose**: Log activity of each component over time to enable:
- Visualization of activation patterns
- Comparison with empirical fMRI/EEG data
- Validation against known findings

**Proposed structure**:
```python
@dataclass
class ActivationEvent:
    timestamp: int           # simulation step
    region: str              # "vmPFC", "dlPFC", "dACC", "dPCC", "vPCC", "sgACC"
    activity_type: str       # "bid", "route", "tally", "coherence_check", etc.
    magnitude: float         # 0-1 activity level
    details: Dict            # context-specific info

class ActivationTracker:
    events: List[ActivationEvent]

    def log(self, region, activity_type, magnitude, details=None)
    def get_timeseries(self, region) -> List[float]
    def get_region_total(self, region) -> float
    def get_phase_activity(self, phase) -> Dict[str, float]
    def summarize() -> Dict[str, float]
    def plot_activity()
```

**Integration points**:
- dPCC.integrate(): Log routing, tallying
- Champion.generate_bid(): Log bid computation
- dACC.check_coherence(): Log coherence check
- vPCC.assess_self_relevance(): Log self-relevance computation
- sgACC.process(): Log gatekeeper activity

### 6.2 Encoding/Retrieval Flip Test

**Empirical finding to replicate**: vPCC and dPCC show opposite patterns during encoding vs retrieval.

**Test design**:
1. **Encoding phase**: Present novel stimulus, process through system
2. **Retrieval phase**: Present cue, retrieve from self-model

**Predictions**:
- Encoding: dPCC higher (routing new info), vPCC lower
- Retrieval: vPCC higher (accessing self-model), dPCC lower

**Implementation needed**:
- Retrieval task scenario
- Cue-based self-model access in vPCC
- Compare activation patterns between phases

### 6.3 Additional Champions

**Amygdala**:
- Prediction problem: "Is this threatening/salient?"
- High value for threat-related stimuli
- Fast, automatic processing
- Connects to sgACC pathway

**Hippocampus**:
- Prediction problem: "Have I seen this before? What happened?"
- Provides familiarity/novelty signal
- Feeds into vPCC self-model

**dmPFC**:
- Prediction problem: "What are others thinking?"
- Social/mentalizing champion
- Important for social dilemmas

### 6.4 Sparse Self-Template Mode

**Purpose**: Demonstrate ideas of reference emergence

**Implementation**:
```python
class vPCC:
    def __init__(self, sparse_mode=False, sparsity=0.5):
        self.sparse_mode = sparse_mode
        self.sparsity = sparsity  # 0 = normal, 1 = maximally sparse

    def assess_self_relevance(self, decision):
        if self.sparse_mode:
            # Fewer features in template → looser matching
            # More noise → more false positives
            base = 0.3 + random.uniform(0, 0.4 * self.sparsity)
            # Reduced discrimination
            object_relevance *= (1 - self.sparsity * 0.5)
        ...
```

**Expected behavior**:
- Normal mode: Self-relevance high only for truly self-relevant stimuli
- Sparse mode: Self-relevance elevated for many stimuli (ideas of reference)

### 6.5 Learning/Plasticity

**Champion weight updates**:
- Positive outcome → strengthen winning champion's weights
- Negative outcome → weaken
- Implements value learning over time

**Self-model refinement**:
- More specific features accumulated over time
- Template becomes richer, more discriminating

### 6.6 Continuous Time Dynamics

**Current**: Discrete iterations
**Planned**: Continuous time with:
- Activation rise/decay curves
- Temporal integration windows
- Oscillatory dynamics (optional, for frequency-related predictions)

---

## 7. Empirical Validation Targets

### 7.1 Encoding/Retrieval Flip (HIGH PRIORITY)

**Finding**: vPCC shows retrieval > encoding; dPCC shows encoding > retrieval (or more complex pattern)

**Test**: Compare activation patterns in encoding vs retrieval tasks

**Success criterion**: Model shows qualitatively similar pattern

### 7.2 Moral Dilemma Response Times

**Finding**: Harder moral dilemmas (competing values) take longer to resolve

**Test**: Measure iteration count for different dilemma types

**Success criterion**:
- Easy dilemmas: ~1-2 iterations
- Hard dilemmas: 5+ iterations
- Correlation between difficulty and iterations

### 7.3 vmPFC Lesion Effects

**Finding**: vmPFC damage → more utilitarian choices (Phineas Gage etc.)

**Test**: Remove vmPFC champion, run moral dilemmas

**Success criterion**: dlPFC wins more often → utilitarian choices increase

### 7.4 ACC Conflict Detection

**Finding**: dACC activity correlates with response conflict

**Test**: Measure dACC "activity" (number of deadlock detections) across conditions

**Success criterion**: Higher dACC activity when champions are closely matched

### 7.5 Self-Referential Processing in vPCC

**Finding**: vPCC activates more for self-relevant stimuli

**Test**: Compare vPCC self-relevance scores for self-relevant vs neutral stimuli

**Success criterion**: Higher scores for self-relevant items

### 7.6 Autonomic Correlates

**Finding**: Harder decisions → higher physiological arousal (HR, SCR)

**Test**: Compare sgACC autonomic output across conditions

**Success criterion**: Higher arousal signal for contested decisions

---

## 8. Open Questions

### 8.1 Theoretical

1. **What exactly is the integration operation?**
   - Currently: simple stake tallying
   - Reality: probably more complex (constraint satisfaction? attractor dynamics?)

2. **How do champions know when to stop escalating?**
   - Current: random adjustment with slight raise bias
   - Need: principled account of bid adjustment dynamics

3. **What determines the dACC threshold?**
   - Currently: fixed at 0.15
   - Should it be adaptive? Context-dependent?

4. **How does the self-model actually represent "self"?**
   - Current: list of past choices
   - Need: richer representation (traits? values? narrative?)

5. **Where does the "choice frame" binding happen?**
   - We said dPCC combines task context + visual structure
   - But what's the actual mechanism?

### 8.2 Implementation

1. **How to represent predictions more richly?**
   - Current: dictionary lookup
   - Options: probability distributions, neural networks, constraint sets

2. **How to implement continuous time?**
   - Differential equations? Event-driven simulation?

3. **How to handle multiple simultaneous tasks?**
   - Current: single task in isolation

4. **How to scale to realistic complexity?**
   - More champions, more objects, more scenarios

### 8.3 Validation

1. **What level of abstraction should we validate at?**
   - Qualitative patterns? Quantitative fits?

2. **Which empirical findings are most diagnostic?**
   - Need to prioritize based on what would most strongly test the model

3. **How to handle findings that don't fit?**
   - Model revision vs. accepting limitations

---

## 9. Connection to Sparse Self-Template Theory

### 9.1 The Core Claim

In psychosis with ideas of reference:
- The self-template (maintained in vPCC) is **sparse** - has fewer distinctive features
- This causes **looser matching** - more stimuli register as "self-relevant"
- Result: External stimuli (strangers looking, TV, etc.) falsely match self-template
- Phenomenology: "That's about ME" when it isn't

### 9.2 How This Model Implements It

**Normal vPCC**:
```
Self-model: rich, specific features
Similarity computation: discriminating
Result: Only truly self-relevant stimuli get high scores
```

**Sparse vPCC** (psychosis):
```
Self-model: impoverished, few features
Similarity computation: loose, noisy
Result: Many stimuli spuriously match
```

### 9.3 Predictions

1. **Behavioral**: People with IOR should show elevated self-relevance ratings for neutral stimuli

2. **Neural**: vPCC should show:
   - Reduced specificity of self-relevance response
   - Possibly elevated baseline activity (always "matching")
   - Reduced differentiation between self/other stimuli

3. **Therapeutic target**: Interventions that enrich the self-template should reduce IOR
   - Neurofeedback targeting vPCC?
   - Therapeutic work on self-narrative/identity?

### 9.4 Model Demonstration (Planned)

```python
# Normal mode
vpcc_normal = vPCC(sparse_mode=False)
# Process neutral stimulus
relevance_normal = vpcc_normal.assess_self_relevance(neutral_stimulus)
# Expected: low (~0.3)

# Sparse mode (simulating IOR)
vpcc_sparse = vPCC(sparse_mode=True, sparsity=0.7)
# Process same neutral stimulus
relevance_sparse = vpcc_sparse.assess_self_relevance(neutral_stimulus)
# Expected: elevated (~0.6+)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Champion** | Brain region that bids on options based on its domain expertise |
| **Stake** | Resource investment a champion places on an option |
| **Deadlock** | When champion bids are too close to determine a winner |
| **Self-template** | The stored representation of self-relevant features in vPCC |
| **Sparse template** | Impoverished self-template with fewer features |
| **Type schema** | Pre-loaded specification of required output format |
| **Gatekeeper** | sgACC's role mediating between cognition and emotion/autonomic |

## Appendix B: File Structure

```
/home/claude/
├── PCC_MODEL_SPEC.md          # This document
├── pcc_model.py               # Current implementation
└── (planned)
    ├── activation_tracker.py  # Activation logging system
    ├── scenarios.py           # Test scenarios
    ├── validation.py          # Empirical comparison tools
    └── visualization.py       # Plotting utilities
```

## Appendix C: Key Citations Informing This Model

1. **vPCC-sgACC connectivity**: Vogt & Pandya (1987) - direct reciprocal connections
2. **dPCC function**: Leech & Sharp - attention, externally-oriented processing
3. **Moral dilemmas**: Greene et al. - dual-process theory, vmPFC vs dlPFC
4. **ACC conflict monitoring**: Botvinick et al. - conflict detection
5. **PCC in self-reference**: Northoff et al. - cortical midline structures
6. **sgACC as gatekeeper**: Various - interface between cognition and emotion

---

*End of Specification*
