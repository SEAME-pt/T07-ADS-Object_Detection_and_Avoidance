# JetRacer Acceleration Characteristics Report

## Objective
Determine the maximum acceleration (**$a_{\text{max}}$**) of the Waveshare JetRacer to set constraints for the Model Predictive Controller (MPC) and PID speed loop in the autonomous drive system. Acceleration limits ensure the controller commands feasible motor inputs for safe and stable operation.

## Methodology
We tested acceleration using a C++ routine to apply step inputs to the motor and logged response times:
1. **Setup**:
   - Used the JetRacer’s DC motor with a 21.5 cm wheel, controlled via PWM through Waveshare’s GPIO library, $I^2C$.
   - Employed the 18-slot encoder wheel to measure speed changes.
   - Implemented a C++ program to apply speed step inputs and log timestamps.
2. **Procedure**:
   - JetSon Car on the floor with 5m free trajectory ahead
   - **Test 1**: Applied a 40% PWM step input from 0% to measure acceleration from rest to steady-state speed.
   - **Test 2**: Applied an 80% PWM step input from 0% to measure acceleration at higher power.
   - Logged start and finish timestamps using `std::chrono` for precise time differences.
   - Measured speed via encoder pulses over 10 ms intervals to calculate acceleration.
3. **Parameters**:
   - Wheel diameter **$W_d$** : 21.5 cm (0.215 m).
   - Encoder Slots, Pulses per turn **$E_s$** : 18 slots, 36 pulses.
   - Sampling interval **$T_s$**: 30 ms.
   - Time logging: `finish_time_stamp - start_time_stamp` in milliseconds.

## Calculations
Acceleration is calculated as:
\[
**$a = \frac{\Delta v}{\Delta t}$**
\]
- **Speed from Encoder** (from Speed Report):
  - Distance per slot, per pulse **$D_p$** = **$\frac{W_d}{E_s} = \frac{0.215}{36} \approx 0.006 \, (m)$**
  - Speed : **$v = {pulses}\ \cdot \frac{D_p}{T_c} \approx {pulse} \cdot \frac{0.006}{0.03} \, (m/s)$**

### Tests previous notes:
    PWM values are ranging from 0 to 4095 equivalent to 0km/h (0 m/s) to 9,3km/h (2,58 m/s) respectively
  - Ideal Maximal Speed (wheels on the air) **$(v_{max,i})$** : **$9,3km/h \equiv \text{PMW}(4095)$**
  - Real Maximal Speed (wheels on the floor) **$(v_{max,r})$**: 9,0km/h

To define the speed, the encoder measurements will be accounted to check the actual speed

### Test 1: 40% Step,  **$(v=1\ m/s)$**, PWM = 1588
- **Logged Data**:
  - Initial speed : **$v_0 = 0 \, {m/s} \equiv PWM = 0$**.
  - Final speed : **$v_f = 1.0 \, \text{m/s}$** (measured via encoder, ~4 pulses/48 ms).
  - Time difference: **$\Delta t = 0.5 \, \text{s}$** (500 ms, logged via `finish_time_stamp - start_time_stamp`).
- **Acceleration**:

  **$a_{40\%} = \frac{1.0 - 0}{0.5} = 2.0 \, \text{m/s}^2$**

### Test 2: 80% Step,   **$(v = 2\ m/s)$**, PWM = 3176
- **Logged Data**:
  - Initial speed: **$v_0 = 0 \, \text{m/s} \equiv 0\ \text{PWM}$**.
  - Final speed: **$v_f = 2.0 \, \text{m/s} \equiv 3176\ \text{PWM}$** (measured via encoder, ~8 pulses/48 ms).
  - Time difference: **$\Delta t = 0.8 \, \text{s}$** (800 ms, logged via `finish_time_stamp - start_time_stamp`).
- **Acceleration**:

  **$a_{80\%} = \frac{2.0 - 0}{0.8} = 2.5 \, \text{m/s}^2$**


### Maximum Acceleration
- The 80% PWM test yielded the highest acceleration, but 100% PWM could be higher. Assuming linear scaling (simplified, as motor dynamics are nonlinear):

  **$a_{\text{max}} \approx \frac{2.5}{0.8} \cdot 1.0 = 3.125 \, \text{m/s}^2$**

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
**$a_{\text{max}} \leq a_k \leq a_{\text{max}}, \quad a_{\text{max}} = 2.0 \, \text{m/s}^2$**
\]
- **MPC**: Constrains acceleration inputs in the control vector **$[\delta, a]$**.
- **PID**: Uses MPC’s **$a$** as the setpoint, ensuring motor commands stay within feasible acceleration limits.

## Notes
- The C++ routine provided reliable timestamps, but motor response may vary with battery voltage or surface conditions.
- Nonlinear motor dynamics (e.g., saturation at high PWM) suggest **$a_{\text{max}} = 2.0 \, \text{m/s}^2$** is conservative.
- Future tests could measure acceleration at 100% PWM or use an accelerometer for direct measurement.

## Date
May 09, 2025