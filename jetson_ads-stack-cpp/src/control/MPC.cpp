#include "MPC.hpp"



MPC::MPC(double delta) : prev_delta(delta) {}

MPC::MPC(const MPC& other) : prev_delta(other.prev_delta) {}

MPC& MPC::operator=(const MPC& other) {
    if (this != &other) {
        prev_delta = other.prev_delta;
    }
    return *this;
}

MPC::~MPC() {}

void MPC::operator()(CppAD::vector<CppAD::AD<double>>& fg, const CppAD::vector<CppAD::AD<double>>& vars) {
    fg[0] = 0.0;
    int n_states = 3 * (N + 1);
    int n_inputs = 2 * N;

    for (int k = 0; k < N; ++k) {
        CppAD::AD<double> ey = vars[k * 3];
        CppAD::AD<double> psi_err = vars[k * 3 + 1];
        CppAD::AD<double> v = vars[k * 3 + 2];
        CppAD::AD<double> delta = vars[n_states + k * 2];
        CppAD::AD<double> a = vars[n_states + k * 2 + 1];

        fg[0] += Q_EY * CppAD::pow(ey, 2);
        fg[0] += Q_PSI_ERR * CppAD::pow(psi_err, 2);
        fg[0] += Q_V * CppAD::pow(v - V_REF, 2);
        fg[0] += R_DELTA * CppAD::pow(delta, 2);
        fg[0] += R_A * CppAD::pow(a, 2);
    }

    for (int k = 0; k < N; ++k) {
        CppAD::AD<double> ey_k = vars[k * 3];
        CppAD::AD<double> psi_err_k = vars[k * 3 + 1];
        CppAD::AD<double> v_k = vars[k * 3 + 2];
        CppAD::AD<double> ey_kp1 = vars[(k + 1) * 3];
        CppAD::AD<double> psi_err_kp1 = vars[(k + 1) * 3 + 1];
        CppAD::AD<double> v_kp1 = vars[(k + 1) * 3 + 2];
        CppAD::AD<double> delta_k = vars[n_states + k * 2];
        CppAD::AD<double> a_k = vars[n_states + k * 2 + 1];

        fg[1 + k * 3] = ey_kp1 - (ey_k + v_k * CppAD::sin(psi_err_k) * DT);
        fg[1 + k * 3 + 1] = psi_err_kp1 - (psi_err_k + (v_k / L) * CppAD::tan(delta_k) * DT);
        fg[1 + k * 3 + 2] = v_kp1 - (v_k + a_k * DT);
    }

    for (int k = 0; k < N; ++k) {
        CppAD::AD<double> delta_k = vars[n_states + k * 2];
        CppAD::AD<double> prev = (k == 0) ? prev_delta : vars[n_states + (k - 1) * 2];
        fg[1 + 3 * N + k] = delta_k - prev - DELTA_RATE_MAX;
        fg[1 + 3 * N + k + N] = prev - delta_k - DELTA_RATE_MAX;
    }
}

// Error dynamics
Vector3d MPC::dynamics(const Vector3d& state, const Vector2d& input) {
    double ey = state(0), psi_err = state(1), v = state(2);
    double delta = input(0), a = input(1);
    Vector3d next_state;
    next_state << ey + v * sin(psi_err) * DT,
                  psi_err + (v / L) * tan(delta) * DT,
                  v + a * DT;
    return next_state;
}

// Solver function
Vector2d MPC::solve_mpc(const Vector3d& current_state, double prev_delta) {
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
            vars[k * 3 + 2] = V_REF;
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
        return Vector2d::Zero();
    }

    Vector2d control;
    control << solution.x[3 * (N + 1)], solution.x[3 * (N + 1) + 1];
    return control;
}