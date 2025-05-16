**PID tuning** 

We’re controlling **speed**, so the goal is:
**smooth, stable tracking of the target velocity** (which came from MPC via acceleration integration), **without overshoot or oscillation.**

---

### 🛠️ 1. **Prep before tuning**

- Set a constant \( a^\star \) (like 0.5 m/s²) to hold speed steady during test.
- Use `v_target = v_current + a_star * dt` every loop.
- Log:
  - `v_current`
  - `v_target`
  - `PWM output`
  - `PID error`

---

### 🧪 2. **Start with only P**

```cpp
Kp = small (try 0.1), Ki = 0, Kd = 0
```

- Run it. If:
  - Speed **tracks slowly** → increase Kp
  - Speed **overshoots and oscillates** → lower Kp
- Goal: Fast rise, no oscillation.

---

### 🌊 3. **Add I (integral)**

```cpp
Kp = good, Ki = small (0.01 or less)
```

- Helps eliminate **steady-state error**.
- Too much Ki = overshoot, "wavy" response (integral windup).

Add an **integral clamp** if needed:
```cpp
integral_ = std::clamp(integral_, -i_limit, i_limit);
```

---

### ⚡ 4. **Add D (derivative)**

```cpp
Kd = 0.01 – 0.1 (start small)
```

- Helps **dampen** the system if we see **ringing/overshoot**.
- Too much = noise amplification (especially with encoders).

---

### 🔧 5. **Typical working set (for small RC car):**

```cpp
Kp = 0.4
Ki = 0.02
Kd = 0.05
```

But you **must tune for our car** — weight, motor power, encoder resolution, etc. all matter.

---

### 🎯 Key tuning strategy:

| Behavior                    | Adjust this | Direction       |
|----------------------------|-------------|-----------------|
| Too slow to reach target   | Kp          | Increase        |
| Overshooting target        | Kp / Kd     | Lower Kp / Raise Kd |
| Never quite reaches target | Ki          | Increase        |
| Noisy PWM / jittery output | Kd          | Lower it        |

---

### 🧪 Bonus: use step test

1. Set a step in `v_target` (e.g. jump from 0.5 → 1.5 m/s).
2. Watch response (log it).
3. Adjust gains until we get fast rise, little overshoot, stable settle.

---

