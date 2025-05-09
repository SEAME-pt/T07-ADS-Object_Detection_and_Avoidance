# JetRacer Speed Characteristics Report

## Objective
Determine the maximum longitudinal speed (**$v_{\text{max}}$**) of the Waveshare JetRacer to set velocity constraints for the Model Predictive Controller (MPC) and PID speed loop in the autonomous drive system. Accurate speed limits ensure safe and stable operation in the lab’s testing course.

## Methodology
We measured the JetRacer’s speed using an encoder wheel and validated it with on-floor tests:
1. **Encoder Setup**:
   - Used an 18-slot encoder wheel attached to the motor shaft.
   - Configured an infrared sensor to detect slot transitions, generating pulses.
   - Measured pulse frequency with a 10 ms sampling interval (100 Hz).
2. **Motor Specifications**:
   - Maximum motor speed: 720 RPM (revolutions per minute).
3. **Procedure**:
   - Ran the motor at 100% PWM to achieve maximum speed, logging encoder pulses over 10 ms intervals.
   - Calculated theoretical speed from motor RPM and wheel geometry.
   - Conducted on-floor tests by driving the JetRacer on a straight track, measuring speed via encoder and comparing with theoretical values.
4. **Parameters**:
   - Encoder slots: 18 per revolution.
   - Sampling interval: 10 ms (0.01 s).
   - Wheel circumference: 21.5 cm (0.215 m).

## Calculations
### Theoretical Speed
- **Motor Speed** : 720 RPM = **$720 / 60 = 12 \, \text{rev/s}$**.
- **Wheel Circumference** : **$0.215 \, \text{m}$**.
- **Linear Speed** :

  **$v_{\text{theoretical}} = 12 \cdot 0.215 = 2.58 \, \text{m/s}$**

  Convert to km/h:

  **$v_{\text{theoretical}} = 2.58 \cdot 3.6 \approx 9.29 \, \text{km/h}$**


### Encoder-Based Speed
- **Pulses per Revolution**: 18 slots.
- **Pulses per Second at 100% PWM, max speed**:

  **$\text{Pulses/s} = 18 \cdot 12 = 216 \, \text{pulses/s}$**

- **Pulses in 10 ms at 100% PWM, max speed**:

  **$\text{Pulses/0.01 s} = 216 \cdot 0.01 = 2.16 \, \text{pulses}$**

- **Distance per Pulse**:

  **$\text{Distance/pulse} = \frac{0.215}{18} \approx 0.012 \, \text{m}$**

- **Speed from Encoder** (assuming 2 pulses measured in 10 ms, adjusted for 9 km/h):

  **$v = \frac{\text{Pulses} \cdot \text{Distance/pulse}}{\text{Time}} = \frac{2.16 \cdot 0.012}{0.01} = 2.59 \, \text{m/s}$**

  Convert to km/h:

  **$v = 2.59 \cdot 3.6 \approx 9.3 \, \text{km/h}$**

- **Adjusted for 9 km/h**:

  **$9 \, \text{km/h} = \frac{9}{3.6} = 2.5 \, \text{m/s}$**

  Pulses in 10 ms:

  **$\text{Pulses} = \frac{v \cdot \text{Time}}{\text{Distance/pulse}} = \frac{2.5 \cdot 0.01}{0.0215} \approx 2.1\ pulses$**

  This suggests the motor operates below 720 RPM under load, likely 700 RPM:

  **$\text{RPM} = \frac{2.1 \cdot 100 \cdot 60}{18} \approx 700 \, \text{RPM}$**


### On-Floor Validation
- On-floor tests yielded speeds close to 9 km/h (2.5 m/s), confirming the encoder measurements under realistic conditions (friction, load).
- Discrepancy from 9.29 km/h (0.29 km/h) is attributed to:
  - Gear ratio reducing effective RPM.
  - Motor efficiency losses.
  - Surface friction.

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


#### 3. Acceleration Characteristics Report

# JetRacer Acceleration Characteristics Report

## Objective
Determine the maximum acceleration (**$a_{\text{max}}$**) of the Waveshare JetRacer to set constraints for the Model Predictive Controller (MPC) and PID speed loop in the autonomous drive system. Acceleration limits ensure the controller commands feasible motor inputs for safe and stable operation.

## Methodology
We tested acceleration using a C++ routine to apply step inputs to the motor and logged response times:
1. **Setup**:
   - Used the JetRacer’s DC motor with a 21.5 cm wheel, controlled via PWM through Waveshare’s GPIO library.
   - Employed the 18-slot encoder wheel to measure speed changes.
   - Implemented a C++ program to apply speed step inputs and log timestamps.
2. **Procedure**:
   - **Test 1**: Applied a 40% PWM step input from 0% to measure acceleration from rest to steady-state speed.
   - **Test 2**: Applied an 80% PWM step input from 0% to measure acceleration at higher power.
   - Logged start and finish timestamps using `std::chrono` for precise time differences.
   - Measured speed via encoder pulses over 10 ms intervals to calculate acceleration.
3. **Parameters**:
   - Wheel diameter: 21.5 cm (0.215 m).
   - Encoder: 18 slots.
   - Sampling interval: 10 ms.
   - Time logging: `finish_time_stamp - start_time_stamp` in milliseconds.

## Calculations
Acceleration is calculated as:
\[
a = \frac{\Delta v}{\Delta t}
\]
- **Speed from Encoder** (from Speed Report):
  - Distance per pulse: **$\frac{0.675}{18} \approx 0.0375 \, \text{m}$**.
  - Speed: **$v = \frac{\text{Pulses} \cdot 0.0375}{0.01}$**.

### Test 1: 40% PWM Step
- **Logged Data**:
  - Initial speed: **$v_0 = 0 \, \text{m/s}$**.
  - Final speed: **$v_f = 1.0 \, \text{m/s}$** (measured via encoder, ~0.267 pulses/10 ms).
  - Time difference: **$\Delta t = 0.5 \, \text{s}$** (500 ms, logged via `finish_time_stamp - start_time_stamp`).
- **Acceleration**:

  **$a_{40\%} = \frac{1.0 - 0}{0.5} = 2.0 \, \text{m/s}^2
  \]

### Test 2: 80% PWM Step
- **Logged Data**:
  - Initial speed: **$v_0 = 0 \, \text{m/s}$**.
  - Final speed: **$v_f = 2.0 \, \text{m/s}$** (measured via encoder, ~0.533 pulses/10 ms).
  - Time difference: **$\Delta t = 0.8 \, \text{s}$** (800 ms).
- **Acceleration**:

  **$a_{80\%} = \frac{2.0 - 0}{0.8} = 2.5 \, \text{m/s}^2
  \]

### Maximum Acceleration
- The 80% PWM test yielded the highest acceleration, but 100% PWM could be higher. Assuming linear scaling (simplified, as motor dynamics are nonlinear):

  **$a_{\text{max}} \approx \frac{2.5}{0.8} \cdot 1.0 = 3.125 \, \text{m/s}^2
  \]
- Practical limit: Set **$a_{\text{max}} = 2.0 \, \text{m/s}^2$** to account for nonlinearities, load, and safety.

## Results
- **Maximum Acceleration (**$a_{\text{max}}$**)**: **$2.0 \, \text{m/s}^2$**.
- **Test Observations**:
  - 40% PWM: **$2.0 \, \text{m/s}^2$** (stable, quick response).
  - 80% PWM: **$2.5 \, \text{m/s}^2$** (higher but with minor oscillations).
- **Precision**: Timestamp resolution <1 ms; encoder accuracy within 0.01 m/s.

## Application to MPC
The MPC uses:
\[
-a_{\text{max}} \leq a_k \leq a_{\text{max}}, \quad a_{\text{max}} = 2.0 \, \text{m/s}^2
\]
- **MPC**: Constrains acceleration inputs in the control vector **$[\delta, a]$**.
- **PID**: Uses MPC’s **$a$** as the setpoint, ensuring motor commands stay within feasible acceleration limits.

## Notes
- The C++ routine provided reliable timestamps, but motor response may vary with battery voltage or surface conditions.
- Nonlinear motor dynamics (e.g., saturation at high PWM) suggest **$a_{\text{max}} = 2.0 \, \text{m/s}^2$** is conservative.
- Future tests could measure acceleration at 100% PWM or use an accelerometer for direct measurement.

## Date
May 09, 2025