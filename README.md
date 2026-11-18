# Hybrid decoder for the [[18,4,4]] qLDPC code

**Winning solution of the 2026 QEC Decoder Hackathon** (NQCP / Niels Bohr Institute, with judges from Riverlane and NVIDIA), by team *Superposed Devs Unobserved*.

The task: decode the [[18,4,4]] qLDPC code of [Wang et al. (2026)](https://doi.org/10.1038/s41567-025-03157-4) under SI1000 circuit-level noise (p = 0.002, 5 rounds of error correction). From the 70 detector bits of a shot, predict whether each of the 4 logical qubits was flipped. Final ranking: logical error rate (LER), with a penalty for slow decoders.

| | Result |
|---|---|
| Private leaderboard LER | **3.441%** |
| Decode time for the 200,000 test shots | **~1.5 s** on one NVIDIA B200 |
| Final placement | **1st** |

## Video

A short explanation of the solution, presented to the judges:

[![QEC Decoder Hackathon - Superposed Devs Unobserved](https://img.youtube.com/vi/Pm7dpdKC4qc/hqdefault.jpg)](https://youtu.be/Pm7dpdKC4qc)

## The method

`decode()` combines two decoders, both written from scratch:

### 1. An exact maximum-likelihood decoder (`exact.py`)

For each logical qubit, the optimal decision adds up the probability of *every* combination of faults consistent with the observed detectors and compares "flipped" with "not flipped". With ~2,500 possible faults that is 2^2500 combinations, so it cannot be done by enumeration.

The key observation: **every fault in the circuit's detector error model lights detectors in only one time layer, or in two neighbouring time layers.** The 70 detectors form six time layers (7, 14, 14, 14, 14, 7), so the faults split into five time intervals, and the sum can be computed by sweeping through time:

- **Tables.** For each interval, a table `K[u, v]` gives the total probability that the faults in that interval light the pattern `u` on the left layer and `v` on the right layer. Faults that flip the logical qubit enter with a minus sign. The tables come straight from the noise model (via Stim's detector error model), so they are exact. They are built once at import, with Walsh–Hadamard transforms.
- **Sweep.** For each shot, a vector of 2^14 = 16,384 numbers (one per possible pattern of the next layer) is carried forward in time: at each layer it is matched to the observed detectors with an XOR and multiplied by the next table. What remains at the end is `P(not flipped) − P(flipped)`; its sign is the prediction.

This is the forward algorithm / transfer-matrix method, made exact and feasible here because the code has only 14 checks per time layer. It needs ~17 GB of GPU memory for the tables and would not scale to large codes.

### 2. A neural network as a fast filter (`decoder.py`)

A residual MLP (width 1024, 8 blocks, 34M parameters per basis) maps the 70 detector bits to one logit per logical qubit. It was trained with binary cross-entropy on ~6.5 billion shots sampled with Stim from the competition circuits; on its own it reaches 3.60% LER.

### The hybrid

The network decodes every shot first. Predictions it is confident about (|logit| ≥ 2.375, about 88.5% of them) keep its answer; the remaining 11.5% are recomputed by the exact decoder. The exact decoder's table products use the GPU's TensorFloat-32 mode, which did not change any prediction on the test set and made it about 9× faster.

| Decoder | LER (fresh shots) | Time, 200k shots (B200) |
|---|---|---|
| Lookup table from 10M samples | 21% | — |
| Neural network alone | 3.60% | 0.3 s |
| Exact decoder alone | 3.38% | ~60 s (float32) |
| **Hybrid** | **3.38%** | **~1.5 s** |

## Usage

Requires an NVIDIA GPU with ~20 GB of memory (the decoder falls back to the CPU, but building the exact decoder's tables then needs ~20 GB of RAM and decoding is much slower).

```bash
pip install -r requirements.txt
bash download_weights.sh          # network weights from the GitHub release (2 × 135 MB)
```

```python
import stim
from decoder import decode

circuit = stim.Circuit.from_file("circuits/X_circuit.stim")
events, flips = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True)
predictions = decode(circuit, events)          # (shots, 4) array of 0/1
print("LER:", (predictions != flips).mean())
```

`python check_handin.py [test.csv] [md5]` reproduces the organisers' check: LER on fresh shots, decode time, and (given the competition's `test.csv`) the submission file and its md5.

## Files

| File | Purpose |
|---|---|
| `decoder.py` | `decode(circuit, detector_events)`: the network, the hybrid logic, GPU warm-up at import |
| `exact.py` | The exact maximum-likelihood decoder (tables + sweep) |
| `weights/hybrid.json` | Hybrid settings: threshold, chunk size, TF32 switches |
| `weights/X.npz`, `weights/Z.npz` | Network weights, one per basis (downloaded by `download_weights.sh`) |
| `circuits/` | The two competition circuits (X and Z memory experiments) |
| `check_handin.py` | Self-check used before the hand-in |
| `training/` | The code that produced the weights |

## Reproducing the weights

Run in `training/` on a GPU (`python train.py --help` lists the options):

1. `bash run_gpu.sh 1024 8 100000`: 100,000 steps of 16,384 fresh shots per basis from random initialisation (learning rate 1e-3, cosine schedule) → `weights/{X,Z}_w1024_d8.npz`. About 20 minutes on a B200.
2. The depth-8 lines of `run_round2.sh`: 300,000 further steps starting from step 1 (`--init`, learning rate 3e-4) → `weights/{X,Z}_w1024_d8_run2.npz`, the shipped weights. About 2 hours. (The same script also trains a depth-16 model that was not used in the final decoder.)

The first run used an earlier revision of `train.py` in which the Stim worker processes shared one queue, so its batch order depended on process timing; the current revision reads one queue per worker in a fixed order.

## Libraries

numpy, [Stim](https://github.com/quantumlib/Stim) (loading circuits, sampling shots, building the detector error model) and PyTorch. No decoding package or third-party decoder code is used.

## License

MIT, see [LICENSE](LICENSE). The competition circuits come from the hackathon's materials, also released under the MIT license.
