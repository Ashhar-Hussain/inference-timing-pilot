# Timing reliability of early-exit inference under distribution shift (pilot)

Pilot for the MSc proposal *Characterising and Evaluating the Timing Reliability of Embedded Neural-Network Inference under Probabilistic Real-Time Scheduling*.

**Question.** An early-exit CNN is profiled on in-distribution (ID) data; a convolution-based probabilistic response-time analysis picks the shortest period whose predicted deadline-miss probability is <= 1%. What is the real miss rate when the inputs shift?

**Setup.** 3-exit CNN on scikit-learn digits (8x8 upsampled to 32x32), exit threshold 0.9. One CPU thread (Intel(R) Xeon(R) CPU @ 2.20GHz), 2500 timed inferences per condition, conditions interleaved in random order. One core, fixed-priority preemptive: control task (4x per frame, exec U[0.8,1.2]x0.298 ms) above the inference task (period = deadline). Analysis: stationary-backlog, split-convolve-merge stochastic response-time analysis (i.i.d. execution times). Ground truth: trace-driven simulation of the measured execution-time sequence.

## Results
Design period from the ID profile: **4.921 ms** (predicted miss 0.010) vs **7.160 ms** for a WCET (max-observed) design (31% shorter).

| condition | frames at full depth | p99 latency (ms) | predicted miss (design) | real miss (real order) | real miss (shuffled) |
|---|---|---|---|---|---|
| ID | 1.00 | 3.886 | 0.010 | 0.032 | 0.017 |
| Shift_mild | 1.00 | 3.508 | 0.010 | 0.010 | 0.006 |
| Shift_strong | 1.00 | 3.590 | 0.010 | 0.021 | 0.008 |
| OOD_noise | 1.00 | 3.494 | 0.010 | 0.014 | 0.006 |
| Bursty | 1.00 | 3.842 | 0.010 | 0.031 | 0.016 |

Static (no-exit) control under strong shift: median latency +0.2% (KS D = 0.218); early-exit network: -1.3% (KS D = 0.215). Analysis validation: max |analysis - shuffled simulation| = 0.0022.

![pilot figure](results/pilot_figure.png)

## Limitations
Toy dataset and model, cloud CPU instead of an embedded board, simplified uniprocessor analysis (not a reproduction of the TII 2026 CAN analysis), single run.

## Run
Open `timing_reliability_pilot.ipynb` in Google Colab, then Runtime -> Run all (about 3 minutes).
