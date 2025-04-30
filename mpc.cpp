#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>
#include <vector>

using CppAD::AD;
using Eigen::Vector4d;
using Eigen::Vector2d;
using Eigen::MatrixXd;

// JetRacer parameters
const double L = 0.15;         // Wheelbase (m)
const double DT = 0.100;       // Time step (s)
const int N = 10;              // Prediction horizon N x DT = 1.0 s
const double V_MAX = 2.0;      // Max velocity (m/s) -> approx 7.2 km/h
const double V_MIN = 0.0;      // Min velocity (m/s)
const double DELTA_MIN = -0.523; // Min steering angle (rad, -30 deg)
const double DELTA_MAX = 0.523; // Max steering angle (rad, 30 deg)
const double A_MAX = 2.0;      // Max acceleration (m/s^2)

// MPC weights
const double Q_X = 100.0;      // Weight for x-position error
const double Q_Y = 100.0;      // Weight for y-position error
const double Q_PSI = 10.0;     // Weight for heading error
const double R_DELTA = 1.0;    // Weight for steering effort
const double R_A = 1.0;        // Weight for acceleration effort

// Bicycle model dynamics
Vector4d dynamics(const Vector4d& state, const Vector2d& input) {
    double x = state(0), y = state(1), psi = state(2), v = state(3);
    double delta = input(0), a = input(1);
    Vector4d next_state;
    next_state << x + v * cos(psi) * DT,
                  y + v * sin(psi) * DT,
                  psi + (v / L) * tan(delta) * DT,
                  v + a * DT;
    return next_state;
}

// MPC optimization class
class MPC {
public:
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    // Store reference trajectory
    std::vector<Vector4d> ref_traj;

    MPC(const std::vector<Vector4d>& ref) : ref_traj(ref) {}

    // Operator for Ipopt to evaluate objective and constraints
    void operator()(ADvector& fg, const ADvector& vars) {
        // fg[0] is the objective (cost function)
        fg[0] = 0.0;

        // Extract states and inputs from vars
        // vars layout: [x_0, y_0, psi_0, v_0, ..., x_N, y_N, psi_N, v_N, delta_0, a_0, ..., delta_{N-1}, a_{N-1}]
        int n_states = 4 * (N + 1); // 4 states per time step
        int n_inputs = 2 * N;       // 2 inputs per time step

        // Compute cost function
        for (int k = 0; k < N; ++k) {
            AD<double> x = vars[k * 4];
            AD<double> y = vars[k * 4 + 1];
            AD<double> psi = vars[k * 4 + 2];
            AD<double> v = vars[k * 4 + 3];
            AD<double> delta = vars[n_states + k * 2];
            AD<double> a = vars[n_states + k * 2 + 1];

            // Tracking error
            fg[0] += Q_X * CppAD::pow(x - ref_traj[k](0), 2);
            fg[0] += Q_Y * CppAD::pow(y - ref_traj[k](1), 2);
            fg[0] += Q_PSI * CppAD::pow(psi - ref_traj[k](2), 2);
            // Control effort
            fg[0] += R_DELTA * CppAD::pow(delta, 2);
            fg[0] += R_A * CppAD::pow(a, 2);
        }

        // Dynamics constraints
        for (int k = 0; k < N; ++k) {
            // Current state
            AD<double> x_k = vars[k * 4];
            AD<double> y_k = vars[k * 4 + 1];
            AD<double> psi_k = vars[k * 4 + 2];
            AD<double> v_k = vars[k * 4 + 3];
            // Next state
            AD<double> x_kp1 = vars[(k + 1) * 4];
            AD<double> y_kp1 = vars[(k + 1) * 4 + 1];
            AD<double> psi_kp1 = vars[(k + 1) * 4 + 2];
            AD<double> v_kp1 = vars[(k + 1) * 4 + 3];
            // Inputs
            AD<double> delta_k = vars[n_states + k * 2];
            AD<double> a_k = vars[n_states + k * 2 + 1];

            // Dynamics equations
            fg[1 + k * 4] = x_kp1 - (x_k + v_k * CppAD::cos(psi_k) * DT);
            fg[1 + k * 4 + 1] = y_kp1 - (y_k + v_k * CppAD::sin(psi_k) * DT);
            fg[1 + k * 4 + 2] = psi_kp1 - (psi_k + (v_k / L) * CppAD::tan(delta_k) * DT);
            fg[1 + k * 4 + 3] = v_kp1 - (v_k + a_k * DT);
        }
    }
};

// Solver function
Vector2d solve_mpc(const Vector4d& current_state, const std::vector<Vector4d>& ref_traj) {
    // Initialize variables
    size_t n_vars = 4 * (N + 1) + 2 * N; // States + inputs
    size_t n_constraints = 4 * N;         // Dynamics constraints
    CppAD::vector<double> vars(n_vars);
    CppAD::vector<double> vars_lower(n_vars), vars_upper(n_vars);
    CppAD::vector<double> constraints_lower(n_constraints), constraints_upper(n_constraints);

    // Initialize state variables
    for (int k = 0; k <= N; ++k) {
        if (k == 0) {
            // Initial state
            vars[k * 4] = current_state(0);     // x
            vars[k * 4 + 1] = current_state(1); // y
            vars[k * 4 + 2] = current_state(2); // psi
            vars[k * 4 + 3] = current_state(3); // v
        } else {
            // Future states (initial guess)
            vars[k * 4] = current_state(0);
            vars[k * 4 + 1] = current_state(1);
            vars[k * 4 + 2] = current_state(2);
            vars[k * 4 + 3] = current_state(3);
        }
        // State bounds (relaxed for simplicity)
        vars_lower[k * 4] = -1e19; vars_upper[k * 4] = 1e19;       // x
        vars_lower[k * 4 + 1] = -1e19; vars_upper[k * 4 + 1] = 1e19; // y
        vars_lower[k * 4 + 2] = -1e19; vars_upper[k * 4 + 2] = 1e19; // psi
        vars_lower[k * 4 + 3] = 0.0; vars_upper[k * 4 + 3] = V_MAX; // v
    }

    // Initialize input variables
    for (int k = 0; k < N; ++k) {
        vars[4 * (N + 1) + k * 2] = 0.0;     // delta
        vars[4 * (N + 1) + k * 2 + 1] = 0.0; // a
        // Input bounds
        vars_lower[4 * (N + 1) + k * 2] = -DELTA_MAX;
        vars_upper[4 * (N + 1) + k * 2] = DELTA_MAX;
        vars_lower[4 * (N + 1) + k * 2 + 1] = -A_MAX;
        vars_upper[4 * (N + 1) + k * 2 + 1] = A_MAX;
    }

    // Constraints (dynamics equality constraints)
    for (int k = 0; k < n_constraints; ++k) {
        constraints_lower[k] = 0.0;
        constraints_upper[k] = 0.0;
    }

    // Set up Ipopt options
    std::string options;
    options += "Integer print_level 0\n";
    options += "String sb yes\n";
    options += "Numeric max_cpu_time 0.01\n"; // 10 ms limit for real-time

    // Object for function evaluation
    MPC mpc(ref_traj);

    // Solve the NLP
    CppAD::ipopt::solve_result<CppAD::vector<double>> solution;
    CppAD::ipopt::solve(options, vars, vars_lower, vars_upper, constraints_lower, constraints_upper, mpc, solution);

    // Check solver status
    if (solution.status != CppAD::ipopt::solve_result<CppAD::vector<double>>::success) {
        std::cerr << "Ipopt failed to converge!" << std::endl;
        return Vector2d::Zero();
    }

    // Extract first control input
    Vector2d control;
    control << solution.x[4 * (N + 1)], solution.x[4 * (N + 1) + 1]; // delta, a
    return control;
}

// Simulate ML model output (replace with actual ML integration)
std::vector<Vector4d> get_reference_trajectory(const Vector4d& current_state) {
    std::vector<Vector4d> ref_traj(N);
    for (int k = 0; k < N; ++k) {
        // Simple linear path for demo (replace with ML output)
        ref_traj[k] << current_state(0) + k * DT * current_state(3), // x
                       current_state(1) + 0.1 * k,                   // y
                       0.0,                                          // psi
                       1.0;                                          // v
    }
    return ref_traj;
}

int main() {
    // Initial state [x, y, psi, v]
    Vector4d state(0.0, 0.0, 0.0, 1.0);

    // Simulate control loop (50 Hz)
    for (int i = 0; i < 100; ++i) {
        // Get reference trajectory from ML model
        std::vector<Vector4d> ref_traj = get_reference_trajectory(state);

        // Solve MPC
        Vector2d control = solve_mpc(state, ref_traj);

        // Apply control (in real setup, send to servo/motors)
        std::cout << "Step " << i << ": delta = " << control(0) << ", a = " << control(1) << std::endl;

        // Update state
        state = dynamics(state, control);
    }

    return 0;
}