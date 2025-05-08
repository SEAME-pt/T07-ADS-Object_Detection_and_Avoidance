#include "MPC.hpp"

// JetRacer parameters
const double L = 0.2;          // Wheelbase (m)
const double DT = 0.02;        // Time step (s)
const int N = 10;              // Prediction horizon
const double V_MAX = 2.0;      // Max velocity (m/s)
const double DELTA_MAX = 0.523; // Max steering angle (rad, 30 deg)
const double A_MAX = 2.0;      // Max acceleration (m/s^2)
const double DELTA_RATE_MAX = 0.1; // Max steering rate (rad/step)
const double V_REF = 1.0;      // Reference velocity (m/s)

// MPC weights
const double Q_EY = 100.0;     // Weight for cross-track error
const double Q_PSI_ERR = 10.0; // Weight for heading error
const double Q_V = 1.0;        // Weight for velocity error
const double R_DELTA = 1.0;    // Weight for steering effort
const double R_A = 1.0;        // Weight for acceleration effort

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