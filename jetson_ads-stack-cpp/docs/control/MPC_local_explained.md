MPC working with **global coordinates** can get complex, especially for a **laboratory context** with a **small testing course** for the **Waveshare JetRacer**. Since we're testing in a controlled environment (a compact track), we can simplify the **Model Predictive Control (MPC)** framework by using a **local coordinate frame** relative to the car or the lane, which aligns perfectly with our **`ML-Lane Detection` - outputs :** steering angle \($\delta$\), longitudinal speed \($v$\), facing angle \($\psi_{\text{error}}$\), and offset \($e_{\text{y}}$\). This approach reduces complexity, avoids the need for absolute global positioning, and leverages **C++ skills** for a clean, efficient implementation on the **Jetson Nano**.

Let’s clarify the setup, and implement the C++ MPC code to work in a local frame, keeping it real-time at 30 Hz.(check data source)

### Why Local Coordinates?
In a **small testing course** (e.g., a lab with a track a few meters long), we don’t need a global reference frame (like GPS or a world map) because:
- The JetRacer operates based on **local sensor data** (dash cam for lane detection).
- Our `ML-Lane Detection` model provides **relative measurements**:
  - **Offset** ($e_{\text{y}}$): Distance from the car’s center to the lane centerline.
  - **Facing angle** (**$\psi_{\text{error}}$**): Angle between the car’s heading and the lane’s tangent.
  - **Speed** \(**$v$**\): Longitudinal velocity.
  - **Steering angle** (\(**$\delta$**\)): Current steering input.
- A **local frame** (centered on the car or aligned with the lane) simplifies the dynamics and reference trajectory, focusing on **tracking the lane** rather than maintaining global \(**$x$**, **$y$**\) positions.
- It’s easier to compute and more robust in a lab where global positioning (e.g., via odometry or SLAM) might be noisy or unaccurate.

### Local Coordinate Frame Approach
We’ll redefine the **bicycle model** and **MPC** in a **car-centric local frame**, where:
- The origin is at the car’s **center of mass** at each time step.
- The x-axis aligns with the car’s **longitudinal axis** (heading direction).
- The y-axis is perpendicular (lateral direction).
- The **lane centerline** is represented by the `ML-Lane Detection` - Lane Detection model’s **offset $e_{\text{y}}$** and **facing angle $\psi_{\text{error}}$**, which define the desired path relative to the car.

In this frame:
- **States**: Instead of global \([**$x$**, **$y$**, $\psi_{\text{error}}$, v]\), we use local errors:
  - **$e_{\text{y}}$**: Cross-track error (offset from lane centerline, directly from `ML-Lane Detection` - Lane Detection).
  - **$\psi_{\text{error}}$**: Heading error (facing angle, directly from `ML-Lane Detection` - Lane Detection).
  - \(**$v$**\): Longitudinal velocity (from `ML-Lane Detection` - Lane Detection).
- **Inputs**: [**$\delta$**\, **$a$**] (steering angle, acceleration).
- **Reference trajectory**: A sequence of desired [**$e{\text{y}}$**, **$\psi_{\text{error}}$**, **$v$**] (e.g., [0, 0, **$v_{\text{ref}}$**] to stay on the centerline).
- **Cost function**: Penalizes deviations from \(**$e_{\text{y}}$** = 0\), \(**$\psi_{\text{error}}$** = 0\), and control effort.

### Simplified Bicycle Model in Local Frame
In the local frame, we model the **error dynamics** rather than global position. The states are:
- **$e_{\text{y}}$**: Lateral offset from the lane centerline (m).
- **$\psi_{\text{error}}$**: Heading error relative to the lane’s tangent (rad).
- **$v$**: Longitudinal velocity (**$m/s$**).

The **dynamics** describe how these errors evolve:
- **Cross-track error rate** **$\dot{e}_{\text{y}}$** :

  $\dot{e}_y = v \sin(\psi_{\text{error}})$

  - If the car’s heading is misaligned (**$\psi_{\text{error}} \neq 0$**), the lateral error changes based on velocity.
- **Heading error rate** **$\dot{\psi}_{\text{error}}$** :

  **$\dot{\psi}_{\text{error}} = \frac{v}{L} \tan(\delta) - \dot{\psi}_{\text{lane}}$**

  - **$\frac{v}{L} \tan(\delta)$** : Yaw rate from steering (same as global model).
  - **$\dot{\psi}_{\text{lane}}$**: Rate of change of the lane’s tangent angle (curvature-related).
  - In a lab with, gentle curves, we can approximate **$\dot{\psi}_{\text{lane}} \approx 0$** for simplicity, assuming the lane’s curvature is small over the prediction horizon (**$N \cdot \Delta t = 0.3 \ {s}$**).
- **Velocity rate** **$\dot{v}$** :

  **$\dot{v} = a$**


- **Discrete dynamics** (_Euler_, **$\Delta t = 0.03 \ {s}$**):

  **$e_{y,k+1} = e_{y,k} + v_k \sin(\psi_{\text{error},k}) \cdot \Delta t$**


  **$\psi_{\text{error},k+1} = \psi_{\text{error},k} + \frac{v_k}{L} \tan(\delta_k)  \cdot \Delta t$**


  **$v_{k+1} = v_k + a_k \cdot \Delta t$**


### MPC Formulation
- **States**: **$[e_y, \psi_{\text{error}}, v]$**.
- **Inputs**: **$[\delta, a]$**.
- **Cost function**:

  **$J = \sum_{k=0}^{N-1} \left( Q_{e_y} e_{y,k}^2 + Q_{\psi} \psi_{error,k}^2 + Q_v (v_k - v_{ref})^2 + R_\delta \delta_k^2 + R_a a_k^2 \right)$**

  - **$e_{y,k}$** : Penalize lateral offset ( desired **$e_y = 0$**).
  - **$\psi_{error,k}$** : Penalize heading misalignment (desired **$\psi_{error} = 0$**).
  - **$v_k - v_{\text{ref}}$** : Track desired speed (e.g., **$v_{{ref}} = 1 {m/s}$**).
  - Weights : **$Q_{e_y} = 100$**, **$Q_{\psi} = 10$**, **$Q_v = 1$**, **$R_\delta = 1$**, **$R_a = 1$**.
- **Constraints**:
  - Steering angle : **$-\delta_{\text{max}} \leq \delta_k \leq \delta_{\text{max}}$** (0.523 rad = 30°) (check source data)
  - Steering rate: **$|\delta_k - \delta_{k-1}| \leq \Delta \delta_{\text{max}}$**, where **$\Delta \delta_{\text{max}} = 0.1 rad$**
  - Acceleration limits : **$-a_{{max}} \leq a_k \leq a_{\text{max}}$**  **$(2 m/s²)$**. (check source data)
  - Velocity limits : **$0 \leq v_k \leq v_{{max}}$** **$(2,52 m/s)$** (check source data)
- **Reference trajectory** : **$[e_{y,\text{ref}}, \psi_{\text{error},\text{ref}}, v_{\text{ref}}] = [0, 0, 1.0]$** for all **$k$**, aiming to stay on the centerline with aligned heading.

### `ML-Lane Detection` - Lane Detection Model Integration
our `ML-Lane Detection` - Lane Detection model provides:
- **Offset** ($e_{\text{y}}$) : Initial state for **$e_{y,0}$**.
- **Facing angle** ($\psi_{{error}}$) : Initial state for **$\psi_{{error},0}$**.
- **Speed** ($v$) : Initial state for **$v_0$**.
- **Steering angle** ($\delta$) : Current control input for rate constraints.

The `ML-Lane Detection` - Lane Detection model likely outputs a single **$[e_y, \psi_{\text{error}}, v, \delta]$** per frame (from the dash cam). Since MPC needs a trajectory for **$N = 10$** steps, we’ll:
- Use the current **$[e_y, \psi_{\text{error}}, v]$** as the initial state.
- Assume the reference is **$[0, 0, v_{{ref}}]$** (stay on the centerline, aligned, at desired speed).

If the `ML-Lane Detection` model would provide a sequence of offsets and angles (e.g., predicted lane path), we could incorporate that! Let's reason about it in a separated topic!

### C++ Code
Let's implement the C++ MPC code to use the **local error dynamics**, keeping the socket-based integration for `ML-Lane Detection` outputs. The changes are:
- States: **$[e_y, \psi_{\text{error}}, v]$** instead of **$[x,\ y,\ \psi,\ v]$**.
- Dynamics: Local error model.
- Cost function: Penalize **$\ e_y, \ \psi_{\text{error}}, \ v - v_{\text{ref}}$**.
- `ML-Lane Detection` data : Parse **$[e_y,\ \psi_{{error}},\ v,\ \delta]$** and use **$\delta$** for rate constraints.


###link to cpp code

##Test Server

### Updated Python `ML-Lane Detection` Server
The Python server sends a single **$[e_y,\ \psi_{{error}},\ v,\ \delta]$** per cycle, reflecting our `ML-Lane Detection` model’s output. Replace `generate_ML_data` with our TensorFlow logic.


###link to python code

###link to setting instructions
