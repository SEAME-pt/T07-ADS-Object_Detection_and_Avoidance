
#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

using CppAD::AD;
using Eigen::Vector3d;
using Eigen::Vector2d;

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

// Socket client class
class SocketClient {
private:
    int sockfd;
public:
    SocketClient(const std::string& host, int port) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr);

        if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Connection failed");
        }
    }

    std::tuple<double, double, double, double> receive_ml_data() {
        double buffer[4]; // ey, psi_err, v, delta
        ssize_t bytes_received = recv(sockfd, buffer, sizeof(buffer), 0);
        if (bytes_received != sizeof(buffer)) {
            throw std::runtime_error("Incomplete ML data received");
        }
        return {buffer[0], buffer[1], buffer[2], buffer[3]};
    }

    void send_state(const Vector3d& state) {
        double buffer[3] = {state(0), state(1), state(2)};
        send(sockfd, buffer, sizeof(buffer), 0);
    }

    ~SocketClient() { close(sockfd); }
};

// Error dynamics
Vector3d dynamics(const Vector3d& state, const Vector2d& input) {
    double ey = state(0), psi_err = state(1), v = state(2);
    double delta = input(0), a = input(1);
    Vector3d next_state;
    next_state << ey + v * sin(psi_err) * DT,
                  psi_err + (v / L) * tan(delta) * DT,
                  v + a * DT;
    return next_state;
}

// MPC optimization class
class MPC {
public:
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    double prev_delta;

    MPC(double delta) : prev_delta(delta) {}

    void operator()(ADvector& fg, const ADvector& vars) {
        fg[0] = 0.0;
        int n_states = 3 * (N + 1);
        int n_inputs = 2 * N;

        for (int k = 0; k < N; ++k) {
            AD<double> ey = vars[k * 3];
            AD<double> psi_err = vars[k * 3 + 1];
            AD<double> v = vars[k * 3 + 2];
            AD<double> delta = vars[n_states + k * 2];
            AD<double> a = vars[n_states + k * 2 + 1];

            fg[0] += Q_EY * CppAD::pow(ey, 2);
            fg[0] += Q_PSI_ERR * CppAD::pow(psi_err, 2);
            fg[0] += Q_V * CppAD::pow(v - V_REF, 2);
            fg[0] += R_DELTA * CppAD::pow(delta, 2);
            fg[0] += R_A * CppAD::pow(a, 2);
        }

        for (int k = 0; k < N; ++k) {
            AD<double> ey_k = vars[k * 3];
            AD<double> psi_err_k = vars[k * 3 + 1];
            AD<double> v_k = vars[k * 3 + 2];
            AD<double> ey_kp1 = vars[(k + 1) * 3];
            AD<double> psi_err_kp1 = vars[(k + 1) * 3 + 1];
            AD<double> v_kp1 = vars[(k + 1) * 3 + 2];
            AD<double> delta_k = vars[n_states + k * 2];
            AD<double> a_k = vars[n_states + k * 2 + 1];

            fg[1 + k * 3] = ey_kp1 - (ey_k + v_k * CppAD::sin(psi_err_k) * DT);
            fg[1 + k * 3 + 1] = psi_err_kp1 - (psi_err_k + (v_k / L) * CppAD::tan(delta_k) * DT);
            fg[1 + k * 3 + 2] = v_kp1 - (v_k + a_k * DT);
        }

        for (int k = 0; k < N; ++k) {
            AD<double> delta_k = vars[n_states + k * 2];
            AD<double> prev = (k == 0) ? prev_delta : vars[n_states + (k - 1) * 2];
            fg[1 + 3 * N + k] = delta_k - prev - DELTA_RATE_MAX;
            fg[1 + 3 * N + k + N] = prev - delta_k - DELTA_RATE_MAX;
        }
    }
};

// Solver function
Vector2d solve_mpc(const Vector3d& current_state, double prev_delta) {
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

int main() {
    Vector3d state(0.1, 0.05, 1.0); // Initial [ey, psi_err, v]
    double prev_delta = 0.0;

    SocketClient client("127.0.0.1", 12345);

    for (int i = 0; i < 100; ++i) {
        client.send_state(state);

        double ey, psi_err, v, delta;
        try {
            std::tie(ey, psi_err, v, delta) = client.receive_ml_data();
        } catch (const std::exception& e) {
            std::cerr << "Socket error: " << e.what() << std::endl;
            ey = 0.1; psi_err = 0.0; v = 1.0; delta = 0.0;
        }

        state << ey, psi_err, v;
        prev_delta = delta;

        Vector2d control = solve_mpc(state, prev_delta);
        std::cout << "Step " << i << ": delta = " << control(0) << ", a = " << control(1) << std::endl;

        state = dynamics(state, control);
    }

    return 0;
}
