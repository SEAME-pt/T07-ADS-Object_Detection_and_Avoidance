# Architecture Decision Record: Control Methodology for Autonomous Drive System of JetRacer

## Status
Accepted

## Context
The Waveshare JetRacer, powered by a Jetson Nano, requires an autonomous drive system to navigate a small laboratory testing course using a dash cam and a TensorFlow-based ML model outputting offset (\(e_y\)), facing angle (\($\psi$_{\text{error}}\)), speed (\(v\)), and steering angle (\(\delta\)). The system must track the lane centerline, maintain a desired speed, operate at 50 Hz, and map control outputs to servo/motor commands via GPIO. A local coordinate frame with error dynamics \([e_y, \psi_{\text{error}}, v]\) is used, and the implementation leverages C++ with Python ML communication via TCP sockets.

## Decision
We will implement a dual-loop control architecture:
- **Outer Loop**: Model Predictive Controller (MPC) for trajectory tracking, optimizing steering (\(\delta\)) and acceleration (\(a\)) to minimize \(e_y\) and \(\psi_{\text{error}}\) over a horizon (\(N = 10\)).
- **Inner Loop**: PID controller for speed regulation, converting \(a\) to motor PWM/voltage to track \(v_{\text{ref}}\).

## Considered Options
1. **Pure MPC**:
   - Single MPC optimizes \(\delta\) and \(a\), controlling steering and motor voltage.
2. **Dual-Loop MPC + PID** (Chosen):
   - MPC for trajectory; PID for speed.
3. **Pure PID**:
   - Cascaded PIDs for lateral (steering) and longitudinal (speed) control.
4. **Reinforcement Learning (RL)**:
   - RL agent maps ML outputs to controls, trained via simulation.

## Pros and Cons of Chosen Decision
### Pros
1. **Optimal Trajectory Tracking**: MPC ensures smooth lane following with constraint handling.
2. **Precise Speed Control**: PID compensates for motor nonlinearities.
3. **Decoupled Control**: Simplifies tuning and debugging.
4. **Real-Time Feasibility**: MPC and PID run at 50 Hz on Jetson Nano.
5. **Reuses Code**: Leverages existing MPC (artifact ID `ccabe509-783f-4485-9de7-396d64a70058`).
6. **Robustness**: MPC smooths ML noise; PID corrects velocity errors.

### Cons
1. **Complexity**: Dual-loop is more complex than single controller.
2. **Tuning Effort**: PID gains require empirical tuning.
3. **MPC Convergence**: Failure to converge may affect PID inputs.
4. **Latency**: Slight delay in PID loop (negligible at 50 Hz).
5. **Implementation Overhead**: PID requires GPIO integration.

## Justification
The dual-loop MPC + PID balances optimality (MPC for trajectory) and simplicity (PID for speed), suits the lab’s small course, ensures real-time performance, and reuses existing code. It’s modular, robust to noise, and feasible with 42 C++ expertise.

## Implementation Notes
- **MPC**: Use local error dynamics, output \(\delta, a\).
- **PID**: Implement C++ PID to convert \(a\) to motor commands.
- **Tuning**: MPC weights (\(Q_{e_y} = 100, Q_{\psi} = 10, Q_v = 1, R_\delta = 1, R_a = 1\)); PID gains (\(K_p = 0.5, K_i = 0.1, K_d = 0.01\)).
- **GPIO**: Use Waveshare GPIO for servo/motor control.

## Risks and Mitigations
- **Tuning**: Use Ziegler-Nichols; test conservatively.
- **MPC Convergence**: Fallback to previous inputs.
- **Latency**: Verify 50 Hz timing.

## Date
May 08, 2025