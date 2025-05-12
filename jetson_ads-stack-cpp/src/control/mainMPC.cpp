#include "SocketClient.hpp"
#include "MPC.hpp"
#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>

// Error dynamics
Eigen::Vector3d dynamics(const Eigen::Vector3d& state, const Eigen::Vector2d& input) {
    const double L = 0.2;  // Wheelbase (m)
    const double DT = 0.02; // Time step (s)

    double ey = state(0), psi_err = state(1), v = state(2);
    double delta = input(0), a = input(1);
    Eigen::Vector3d next_state;
    next_state << ey + v * sin(psi_err) * DT,
                  psi_err + (v / L) * tan(delta) * DT,
                  v + a * DT;
    return next_state;
}

// Solver function
Eigen::Vector2d solve_mpc(const Eigen::Vector3d& current_state, double prev_delta) {
    const int N = 10;              // Prediction horizon
    const double V_MAX = 2.0;      // Max velocity (m/s)
    const double DELTA_MAX = 0.523; // Max steering angle (rad)
    const double A_MAX = 2.0;      // Max acceleration (m/s^2)
    const double DELTA_RATE_MAX = 0.1; // Max steering rate (rad/step)

    size_t n_vars = 3 * (N + 1) + 2 * N;
    size_t n_constraints = 3 * N + 2 * N;
    CppAD::vector<double> vars(n_vars);
    CppAD::vector<double> vars_lower(n_vars), vars_upper(n_vars);
    CppAD::vector<double> constraints_lower(n_constraints), constraints_upper(n_constraints);

    for (int k = 0; k <= N; ++k) {
        if (k == 0) {
            vars[k * 3] = current_state(0);
            vars[k * 3 + 1] = current_state(1);
            vars[k * 3 + 2] = current_state(2);
        } else {
            vars[k * 3] = 0.0;
            vars[k * 3 + 1] = 0.0;
            vars[k * 3 + 2] = 1.0; // V_REF
        }
        vars_lower[k * 3] = -1e19; vars_upper[k * 3] = 1e19;
        vars_lower[k * 3 + 1] = -1e19; vars_upper[k * 3 + 1] = 1e19;
        vars_lower[k * 3 + 2] = 0.0; vars_upper[k * 3 + 2] = V_MAX;
    }

    for (int k = 0; k < N; ++k) {
        vars[3 * (N + 1) + k * 2] = prev_delta;
        vars[3 * (N + 1) + k * 2 + 1] = 0.0;
        vars_lower[3 * (N + 1) + k * 2] = -DELTA_MAX;
        vars_upper[3 * (N + 1) + k * 2] = DELTA_MAX;
        vars_lower[3 * (N + 1) + k * 2 + 1] = -A_MAX;
        vars_upper[3 * (N + 1) + k * 2 + 1] = A_MAX;
    }

    for (int k = 0; k < 3 * N; ++k) {
        constraints_lower[k] = 0.0;
        constraints_upper[k] = 0.0;
    }
    for (int k = 3 * N; k < 3 * N + 2 * N; ++k) {
        constraints_lower[k] = -1e19;
        constraints_upper[k] = 0.0;
    }

    std::string options;
    options += "Integer print_level 0\n";
    options += "String sb yes\n";
    options += "Numeric max_cpu_time 0.01\n";

    MPC mpc(prev_delta);
    CppAD::ipopt::solve_result<CppAD::vector<double>> solution;
    CppAD::ipopt::solve(options, vars, vars_lower, vars_upper, constraints_lower, constraints_upper, mpc, solution);

    if (solution.status != CppAD::ipopt::solve_result<CppAD::vector<double>>::success) {
        std::cerr << "Ipopt failed to converge!" << std::endl;
        return Eigen::Vector2d::Zero();
    }

    Eigen::Vector2d control;
    control << solution.x[3 * (N + 1)], solution.x[3 * (N + 1) + 1]; // delta, a
    return control;
}

int main() {
    Eigen::Vector3d state(0.1, 0.05, 1.0); // Initial [ey, psi_err, v]
    double prev_delta = 0.0;

    SocketClient client("127.0.0.1", 12345);

    for (int i = 0; i < 100; ++i) {
        client.send_state(state);

        double ey, psi_err, v, delta;
        try {
            std::tie(ey, psi_err, v, delta) = client.receive_ml_data(); // Assuming this function returns ey, psi_err, v, delta
			std::cout << "Received: ey = " << ey << ", psi_err = " << psi_err << ", v = " << v << ", delta = " << delta << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Socket error: " << e.what() << std::endl;
            ey = 0.1; psi_err = 0.0; v = 1.0; delta = 0.0;
        }

        state << ey, psi_err, v;
        prev_delta = delta;

        Eigen::Vector2d control = solve_mpc(state, prev_delta);
        std::cout << "Step " << i << ": delta = " << control(0) << ", a = " << control(1) << std::endl;

        state = dynamics(state, control);
    }

    return 0;
}