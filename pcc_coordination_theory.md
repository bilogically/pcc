# PCC Coordination Theory: A Computational Framework

## Core Insight

The posterior cingulate cortex has two functionally distinct subregions that handle fundamentally different coordination modes:

### dPCC (Dorsal PCC): "Act as ONE toward the WORLD"
- **Function**: Coordinates whole-brain unified action toward external stimuli
- **Connectivity**: CEN (central executive network), premotor areas, motor areas
- **Active when**: External focus, need to respond as unified agent
- **Mode**: Champions/agents must AGREE on single action → arbitration required
- **Key feature**: Anti-correlated with DMN

### vPCC (Ventral PCC): "Route signals between PARTS"
- **Function**: Routes signals between internal agents who remain separate
- **Connectivity**: DMN, sgACC, hippocampus
- **Active when**: Internal processing, agents work independently  
- **Mode**: No agreement needed → just facilitate communication
- **Key feature**: Core DMN node

## The Encoding/Retrieval Flip Explained

This framework elegantly explains the classic PCC encoding/retrieval asymmetry:

| Process | dPCC Activity | vPCC Activity | Explanation |
|---------|---------------|---------------|-------------|
| **Encoding** | Low | Low | Hippocampus just snapshots the configuration - no active coordination needed |
| **Retrieval** | Low | **HIGH** | vPCC must coordinate agents back INTO the stored shape - this IS remembering |

### Why retrieval needs vPCC but not dPCC:
- Retrieval = reinstantiating a past configuration
- Agents must be routed back into their stored positions
- But this is all INTERNAL - no external action required
- So dPCC (external coordinator) stays quiet
- While vPCC (internal router) is highly active

## The Self as Coordination Pattern

**Critical reframe**: The "self" is not something vPCC checks stimuli against. The self IS the characteristic pattern of internal coordination.

- Your self-model = the typical way your internal agents interact
- Stored as configurations in hippocampus
- Retrieved by vPCC reinstating those patterns
- The shape of [vmPFC ↔ dlPFC ↔ amygdala] relationships IS "you"

## Ideas of Reference from Sparse Self-Template

When vPCC's routing is noisy or the self-model is impoverished:

**Normal vPCC (rich self-template)**:
- Precise pattern matching
- Random stimuli don't match characteristic self-patterns
- "That random thing isn't about me"

**Sparse/Noisy vPCC**:
- Ambiguous pattern matching
- Random stimuli trigger "this might be about me"
- The news anchor seems to speak TO ME
- License plates contain messages FOR ME
- This IS ideas of reference!

## Computational Implementation

The model implements:

1. **Agents** (vmPFC, dlPFC, amygdala, hippocampus) with value preferences
2. **Configurations** as snapshots of agent states  
3. **dPCC** coordinating external choices through champion bidding + dACC arbitration
4. **vPCC** routing internal signals and reinstating memories
5. **Noise parameter** to model sparse self-template

Key demonstration: Unrelated stimuli show ~0.2 self-relevance in normal system but ~0.6+ in sparse system.

## Anatomical Support

From the literature:
- dPCC: "highly active during tasks that require an external focus, especially concerning visuospatial and body orientation"
- dPCC: "functionally highly connected to frontoparietal networks of attention and executive control"
- vPCC: "strong functional connectivity to the rest of the DMN"
- vPCC: "interacts with subgenual cortex to process self-relevant emotional and non-emotional information"
- Critical: vPCC has DIRECT reciprocal connection with sgACC (dPCC does not)

## Implications

1. **Memory disorders**: Impaired vPCC → difficulty reinstating past configurations → amnesia
2. **Psychosis**: Noisy vPCC → ambiguous self-relevance → ideas of reference
3. **Depression**: Overactive vPCC-sgACC loop → rumination (stuck reinstating negative configurations)
4. **Meditation**: Training to reduce dPCC activity (external focus) while maintaining vPCC function (internal awareness)

## Next Steps

- Model specific psychiatric conditions by manipulating noise/connectivity parameters
- Test predictions about dPCC/vPCC activity in encoding vs retrieval tasks
- Explore how sgACC outputs (autonomic, emotional) are shaped by vPCC coordination
- Develop training paradigms that could strengthen sparse self-templates
