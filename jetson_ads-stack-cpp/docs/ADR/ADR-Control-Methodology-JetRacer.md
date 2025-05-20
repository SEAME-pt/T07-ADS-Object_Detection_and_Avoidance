# Architecture Decision Record: Control Methodology for Autonomous Drive System of JetRacer

## Status
Accepted

## Context
The Waveshare JetRacer, powered by a Jetson Nano, requires an autonomous drive system to navigate a small laboratory testing course using a dash cam and a TensorFlow-based ML model outputting offset **_e_**_y_, facing angle **_ψₑᵣᵣₒᵣ_**, speed **_v_**, and steering angle **_δ_**. The system must track the lane center line, maintain a desired speed, operate at 30 Hz*, and map control outputs to servo/motor commands via GPIO. A local coordinate frame with error dynamics [**_e_**_y_, ψₑᵣᵣₒᵣ, v] is used, and the implementation leverages C++ with Python ML communication via TCP sockets.

**Why 30Hz?**
_The ML lane detection is capable of a 30ms process stream, wich gives 33Hz (or FPS) of state update frequency._

## Decision
We will implement a dual-loop control architecture:
- **Outer Loop**: Model Predictive Controller (MPC) for trajectory tracking, optimizing steering **_δ_** and acceleration **_a_** to minimize **_e_**_y_ and **_ψₑᵣᵣₒᵣ_** over a horizon **_N_** = 10.
- **Inner Loop**: PID controller for speed regulation, converting **_a_** to motor PWM to track **_v_**_ref_.

## Considered Options
1. **Pure MPC**:
   - Single MPC optimizes **_δ_** and **_a_**, controlling steering and motor voltage.
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
4. **Real-Time Feasibility**: MPC and PID run at 30ms on Jetson Nano.
5. **Reuses Code**: Leverages existing MPC.
6. **Robustness**: MPC smooths ML noise; PID corrects velocity errors.

### Cons
1. **Complexity**: Dual-loop is more complex than single controller.
2. **Tuning Effort**: PID gains require empirical tuning.
3. **MPC Convergence**: Failure to converge may affect PID inputs.
4. **Latency**: Slight delay in PID loop (negligible at 30ms ).
5. **Implementation Overhead**: PID requires GPIO integration.

## Justification
The dual-loop MPC + PID balances optimality (MPC for trajectory) and simplicity (PID for speed), suits the lab’s small course, ensures real-time performance, and reuses existing code. It’s modular, robust to noise, and feasible with C++ expertise.

## Implementation Notes
- **MPC**: Use local error dynamics, output (**_δ_**, **_a_**).
- **PID**: Implement C++ PID to convert **_a_** to motor commands.
- **Tuning MPC weights**: **_Qe_**_y_ = 100, **_Qψ_** = 10, **_Q_**_v_ = 1, **_Rδ_** = 1, **_Ra_** = 1, determined by ML MPC training RNN;
- **Tuning PID gains**: **_K_**_p_ , **_K_**_i_, **_K_**_d_ , will be determined using an autotune method, e.g. Ziegler-Nichols
- **GPIO**: Use Waveshare GPIO for servo/motor control (**_I2C_** interface).

## Risks and Mitigations
- **Tuning**: Use Ziegler-Nichols; test conservatively.
- **MPC Convergence**: Fallback to previous inputs.
- **Latency**: Verify 30ms timing.

## Date
May 08, 2025
