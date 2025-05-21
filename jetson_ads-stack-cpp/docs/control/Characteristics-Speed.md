# JetRacer Speed Characteristics Report

## Objective
Determine the maximum longitudinal speed (**$v_{\text{max}}$**) of the Waveshare JetRacer to set velocity constraints for the Model Predictive Controller (MPC) and PID speed loop in the autonomous drive system. Accurate speed limits ensure safe and stable operation in the lab’s testing course.

## Methodology
We measured the JetRacer’s speed using an encoder wheel and validated it with on-floor tests:
1. **Encoder Setup**:
   - Used an 18-slot encoder wheel attached to the motor shaft.
   - Configured an infrared sensor to detect slot transitions, generating pulses.
   - Measured pulse frequency with a 30 ms sampling interval (33 Hz).
2. **Motor Specifications**:
   - Maximum motor speed: 720 RPM (revolutions per minute).
3. **Procedure**:
   - Ran the motor at 100% PWM to achieve maximum speed, logging encoder pulses over 10 ms intervals.
   - Calculated theoretical speed from motor RPM and wheel geometry.
   - Conducted on-floor tests by driving the JetRacer on a straight track, measuring speed via encoder and comparing with theoretical values.
4. **Parameters**:
   - Wheel diameter **$W_d$** : **$21.5 \, cm \, (0.215 m)$**.
   - Encoder Slots, Pulses per turn **$E_s$** : **$18 \,{slots} \, \text{x} \, 2 \, edges = 36 \, edges$** (rising + falling).
   - Sampling interval **$T_s$**: 30 ms (**$\frac{1}{33 \text{FPS}}$**).
   - Time logging: `finish_time_stamp - start_time_stamp` in milliseconds.

## Calculations
### Theoretical Speed
- **Motor Speed** : 720 RPM = **$720 / 60 = 12 \, \text{rev/s}$**.
- **Wheel Circumference** : **$0.215 \, \text{m}$**.
- **Linear Speed** :

  **$v_{\text{theoretical}} = 12 \, \text{rev/s} \cdot 0.215 \, \text{m} = 2.58 \, \text{m/s}$**

  Convert to km/h:

  **$v_{\text{theoretical}} = 2.58 \cdot 3.6 \approx 9.29 \, \text{km/h}$**


### Encoder-Based Speed
- **Pulses per Revolution**: 36 slots.
- **Pulses per Second at 100% PWM, max speed**:

  **$\text{Pulses/s} = 36 \cdot 12 = 432 \, \text{pulses/s}$**

- **Pulses in 30 ms at 100% PWM, max speed**:

  **$\text{Pulses @ 0.03 s} = 432 \cdot 0.03 = 12.96 \, \text{pulses}$**

- **Distance per Pulse** **$(D_p)$**:

  **$\frac{W_d}{E_s} = \frac{0.215}{36} \approx 0.006 \, \text{m}$** **$(6 \,mm)$**, **$error E_{D_p}=0,5 \cdot D_p = 0,003 m$**

- **Speed from Encoder** (assuming 13 pulses measured in 30 ms):

  **$v = \text{pulses} \cdot \frac{D_p}{T_s} = 13 \cdot \frac{0.006}{0.03} = 2.6 \, \text{m/s}$**

  Convert to km/h:

  **$v = 2.6 \cdot 3.6 \approx 9.36 \, \text{km/h}$**

- **Adjusted for 9 km/h**:

  **$9 \, \text{km/h} = \frac{9}{3.6} = 2.5 \, \text{m/s}$**

  Pulses in 30 ms:

  **$\text{Pulses} = \frac{v \cdot \text{Time}}{\text{Distance/pulse}} = \frac{2.5 \cdot 0.03}{0.006} \approx 13\ pulses$**

  This suggests the motor operates below 720 RPM under load, likely 700 RPM:

  **$\text{RPM} = \frac{2.1 \cdot 100 \cdot 60}{18} \approx 700 \, \text{RPM}$**


### On-Floor Validation
- On-floor tests yielded speeds close to 9 km/h (2.5 m/s), confirming the encoder measurements under realistic conditions (friction, load).
- Discrepancy from 9.29 km/h (0.29 km/h) is attributed to:
  - Motor efficiency losses.
  - Floor Surface friction.

## Results
- **Maximum Speed (**$v_{\text{max}}$**)**: **$2.5 \, \text{m/s} \approx 9 \, \text{km/h}$**.
- **Validation**: On-floor tests matched encoder calculations.
- **Practical Limit**: Set **$v_{\text{max}} = 2.0 \, \text{m/s}$** for MPC to ensure safety margin.

## Application to MPC
The MPC and PID loops use:
\[
**$0 \leq v_k \leq v_{\text{max}}, \quad v_{\text{max}} = 2.0 \, \text{m/s}$**
\]
- **MPC**: Constrains velocity in the state vector **$[e_y, \psi_{\text{error}}, v]$**
- **PID**: Tracks **$v_{\text{ref}} = 1.0 \, \text{m/s}$**, well below **$v_{\text{max}}$**, ensuring stable speed control.

## Notes
- The 720 RPM motor spec likely assumes no load; actual RPM under load is ~700 RPM, explaining the 9 km/h result.
- Encoder resolution (18 slots) is sufficient for 100 Hz sampling but could be improved with higher slot count for precision.
- Future tests could use a tachometer to directly measure wheel RPM.

## Date
May 09, 2025


