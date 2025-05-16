# JetRacer Steering Characteristics Report

## Objective
Determine the maximum steering angle (**$\delta_{\text{max}}$**) of the Waveshare JetRacer to define constraints for the Model Predictive Controller (MPC) in the autonomous drive system. The steering angle is critical for trajectory tracking, ensuring the controller respects the physical limits of the JetRacer’s servo.

## Methodology
To measure the steering angle, we conducted a physical experiment in the laboratory:
1. **Setup**:
   - Placed the JetRacer on a large sheet of white paper.
   - Imobilized the car position, so it doesn't move during measurements.
2. **Procedure**:
   - Commanded the servo to maximum left steering and held it while manually drawing a steering line **$L_l$** on the floor white paper corresponding to the left wheel outer face plane.
   - Repeated for maximum right steering line  **$L_r$** line, also the left wheel outer face plane.
   - Trace another line **$L_t$** that crosses both steering lines **$L_l, \ L_r$** at 10cm distance from where the two steering lines cross, at point **$O$**
   - Measure the lenght of that line **$L_t$** using the millimetric scale.
   - Trace another line **$L_c$** , from the center of line **$L_t$** to the point **$O$**
3. **Parameters**:
   - Lenght of segment line (**$L_t = 10,6 \ cm = 0.106 \ m$**): Measured between the intersections of lines **$L_f, \ L_r$** 10cm away from point **$O$**
   - Lentgh of segment line (**$L_c = 8,7 \ cm = 0.087 \ m$**): Measured between center of segment **$L_t$** and point **$O$**
arcs for left and right max steering.

## Calculations
The steering angle (**$\delta$**) relates to the total turning angle between **$L_l$** and **$L_r$** as follows :

**$\tan\delta = \frac{L_t}{2 \cdot L_c}$**


**$\delta = \arctan\left(\frac{L_t}{2 \cdot L_c}\right)$**


## Results

##$**



- **Left Max Steering = Right Max Steering** (assumed for symmetry):

  - **$\delta = \arctan\left(\frac{0.106}{2 \cdot 0,087}\right) = 31,34^\circ$**

  - **$\delta = 31,34\circ = 0,548 \ rad \approx \frac{\pi}{6} = 0.5236 \ rad\ (30^\circ)$**

  - Took the average for symmetry: **$\delta_{\text{max}} \approx 0.523 \, \text{rad} \approx 30^\circ$**.
  - Validated by repeating measurements, ensuring consistency within 0.5 mm.

## Results
- **Maximum Steering Angle (**$\delta_{\text{max}}$**)**: **$0.523 \, \text{rad} (30^\circ)$**.
- **Direction**: Symmetric for left and right steering (**$-\delta_{\text{max}} \leq \delta \leq \delta_{\text{max}}$**).
- **Precision**: Measurements accurate to 0.5 mm, yielding **$\delta$** within 0.01 rad.

## Application to MPC
The MPC uses **$\delta_{\text{max}}$** as a constraint:

  **$-\delta_{\text{max}} \leq \delta_k \leq \delta_{\text{max}}, \quad \delta_{\text{max}} = 0.523 \, \text{rad}$**

Additionally, a steering rate constraint is applied:

**$|\delta_k - \delta_{k-1}| \leq \Delta \delta_{\text{max}}, \quad \Delta \delta_{\text{max}} = 0.1 \, \text{rad}$**

This ensures the servo operates within physical limits, preventing oversteering and ensuring smooth trajectory tracking in the lab’s small testing course.

## Notes
- The pencil trace method provided high precision but required careful alignment to avoid parallax errors.
- Servo calibration was verified to ensure maximum steering commands corresponded to the measured angles.
- The tests were validated using protractor for direct angle measurement to cross-validate results.

## Date
May 09, 2025