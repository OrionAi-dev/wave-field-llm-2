# Physics Diagnostic: Why the 2x PPL Gap?

**Date**: 2026-03-02
**Status**: Approved
**Goal**: Understand WHERE information is lost in the Wave Field pipeline vs Standard attention, using Manim visualizations + quantitative ablation experiments.

## The Three Walls

The Wave Field effective attention is:
```
A_ij^(d) = k_wave(i-j) * phi_q(q_i)_d * phi_k(k_j)_d
```

Three structural limitations vs Standard attention `softmax(QK^T/sqrt(d))`:

### Wall 1: Toeplitz Kernel (position-only routing)
- `k_wave(i-j)` depends ONLY on distance, not content
- Standard gives unique weight to every (i,j) pair
- Prevents position-specific routing ("attend to 3rd token")

### Wall 2: Per-Dimension Factorization (no cross-dim mixing)
- Element-wise `phi_q * phi_k * v` means dim d of output depends only on dim d
- Standard's `q . k` dot product mixes ALL dimensions for scalar weight
- Like d separate 1-bit heads instead of one d-bit head

### Wall 3: LTI Constraint (same kernel everywhere)
- SpectralGate uses only `q[:,:,0,:]` (token 0) — same kernel for all positions
- Standard computes a fresh distribution for each query position
- Prevents content-dependent lookback ("if verb, look further back for subject")

## Part 1: Manim Visualizations

### Scene 1 — Token-to-Field Mapping
- N=16 tokens on number line
- Animate scatter: each token deposits phi_k(k) * v onto field position
- Color by magnitude, show 1:1 mapping (stride >= 1.0)

### Scene 2 — Wave Kernel & Convolution
- Show 8 learned kernels as damped cosines
- Distinguish frozen heads (4 fixed) vs learnable (4 trainable)
- Animate convolution: each deposit "radiates" through its kernel

### Scene 3 — SpectralGate Modulation
- Base kernel FFT spectrum
- Gate tensor from MLP
- (1 + gate) * base_fft — how much does spectrum actually change?
- Key metric: gate magnitude relative to 1.0

### Scene 4 — Effective Attention Matrix Comparison
- Extract A_ij from Wave Field numerically
- Display as heatmap alongside Standard attention heatmap
- Same input, same model scale
- Wave = banded (Toeplitz), Standard = content-routed

### Scene 5 — Rank Bottleneck
- SVD of both attention matrices
- Singular value bar charts side-by-side
- Effective rank comparison

### Scene 6 — Kernel Evolution During Training
- At init: HiPPO frequencies, uniform damping
- During training: which heads sharpen? Which collapse?
- Frozen vs learned head dynamics

### Scene 7 — Feature Map Evolution
- At init: phi(x) = ReLU(x) (identity + ReLU)
- After training: do maps become spiky? SVD of feature map weights.

### Scene 8 — Energy Flow
- Track ||field|| at each stage: deposit -> scatter -> convolve -> gather
- Where does magnitude grow/shrink?

## Part 2: Wall Ablation Experiment

Quantitative experiment to isolate each wall's PPL contribution.

| ID | Experiment | Modification | Tests |
|----|-----------|-------------|-------|
| E1 | Oracle kernel | Replace k_wave(i-j) with softmax(QK^T) weights | Wall 1 cost |
| E2 | Dot-product gather | Replace phi_q * gathered with q . gathered (matmul) | Wall 2 cost |
| E3 | Per-token kernel | Each position gets its own kernel via MLP on pos embed | Wall 3 cost |
| E4 | All walls removed | E1 + E2 + E3 combined | Total structural cost |

All experiments: S1 scale (22M params), 5M tokens, WikiText-2, seed=42.
Baseline comparisons: Wave V4.3.9 (~234 PPL) and Standard (~162.5 PPL).

## Part 3: Data Extraction Script

A `diagnostics/extract_physics.py` script that:
1. Loads a trained Wave model checkpoint (or trains for 100 steps)
2. Runs a forward pass with hooks to capture every intermediate tensor
3. Computes effective attention matrix
4. Computes kernel reach, feature map rank, gate magnitude, energy flow
5. Saves all data as JSON + numpy arrays for Manim to consume

## Implementation Order

1. `diagnostics/extract_physics.py` — data extraction (needed by everything else)
2. `scripts/wall_ablation.py` — E1-E4 experiments (quantitative results)
3. `scripts/manim_physics.py` — Manim scenes 1-8 (visualizations)
