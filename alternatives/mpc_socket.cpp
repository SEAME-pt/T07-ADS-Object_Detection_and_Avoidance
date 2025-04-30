/**
 * @file mpc_socket.cpp
 * @brief Implementation of Model Predictive Control (MPC) with socket communication for trajectory input.
 *
 * This file contains the implementation of an MPC algorithm that receives reference trajectories
 * via a socket connection and computes optimal control inputs for a JetRacer-like vehicle.
 */

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
using Eigen::Vector4d;
using Eigen::Vector2d;
using Eigen::MatrixXd;

// JetRacer parameters
const double L = 0.2;          ///< Wheelbase (m)
const double DT = 0.02;        ///< Time step (s)
const int N = 10;              ///< Prediction horizon
const double V_MAX = 2.0;      ///< Max velocity (m/s)
const double DELTA_MAX = 0.523; ///< Max steering angle (rad, 30 deg)
const double A_MAX = 2.0;      ///< Max acceleration (m/s^2)

// MPC weights
const double Q_X = 100.0;      ///< Weight for x-position error
const double Q_Y = 100.0;      ///< Weight for y-position error
const double Q_PSI = 10.0;     ///< Weight for heading error
const double R_DELTA = 1.0;    ///< Weight for steering effort
const double R_A = 1.0;        ///< Weight for acceleration effort

/**
 * @class SocketClient
 * @brief A simple socket client for receiving reference trajectories.
 */
class SocketClient {
private:
    int sockfd; ///< Socket file descriptor

public:
    /**
     * @brief Constructor for SocketClient.
     * @param host The IP address of the server.
     * @param port The port number of the server.
     * @throws std::runtime_error if the socket creation or connection fails.
     */
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

    /**
     * @brief Receives a reference trajectory from the server.
     * @return A vector of reference states (Vector4d).
     * @throws std::runtime_error if the trajectory is incomplete or the socket fails.
     */
    std::vector<Vector4d> receive_trajectory() {
        std::vector<Vector4d> traj(N);
        double buffer[N * 4];
        ssize_t bytes_received = recv(sockfd, buffer, sizeof(buffer), 0);
        if (bytes_received != sizeof(buffer)) {
            throw std::runtime_error("Incomplete trajectory received");
        }
        for (int i = 0; i < N; ++i) {
            traj[i] << buffer[i * 4], buffer[i * 4 + 1], buffer[i * 4 + 2], buffer[i * 4 + 3];
        }
        return traj;
    }

    /**
     * @brief Destructor for SocketClient. Closes the socket.
     */
    ~SocketClient() { close(sockfd); }
};

/**
 * @brief Computes the next state of the vehicle using the bicycle model dynamics.
 * @param state The current state of the vehicle [x, y, psi, v].
 * @param input The control input [delta, a].
 * @return The next state of the vehicle.
 */
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

/**
 * @class MPC
 * @brief Model Predictive Control optimization class.
 */
class MPC {
public:
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector; ///< Alias for AD vector type

    std::vector<Vector4d> ref_traj; ///< Reference trajectory

    /**
     * @brief Constructor for MPC.
     * @param ref The reference trajectory.
     */
    MPC(const std::vector<Vector4d>& ref) : ref_traj(ref) {}

    /**
     * @brief Operator for Ipopt to evaluate the objective and constraints.
     * @param fg The cost function and constraints.
     * @param vars The optimization variables.
     */
    void operator()(ADvector& fg, const ADvector& vars) {
        fg[0] = 0.0;
        int n_states = 4 * (N + 1);
        int n_inputs = 2 * N;

        for (int k = 0; k < N; ++k) {
            AD<double> x = vars[k * 4];
            AD<double> y = vars[k * 4 + 1];
            AD<double> psi = vars[k * 4 + 2];
            AD<double> v = vars[k * 4 + 3];
            AD<double> delta = vars[n_states + k * 2];
            AD<double> a = vars[n_states + k * 2 + 1];

            fg[0] += Q_X * CppAD::pow(x - ref_traj[k](0), 2);
            fg[0] += Q_Y * CppAD::pow(y - ref_traj[k](1), 2);
            fg[0] += Q_PSI * CppAD::pow(psi - ref_traj[k](2), 2);
            fg[0] += R_DELTA * CppAD::pow(delta, 2);
            fg[0] += R_A * CppAD::pow(a, 2);
        }

        for (int k = 0; k < N; ++k) {
            AD<double> x_k = vars[k * 4];
            AD<double> y_k = vars[k * 4 + 1];
            AD<double> psi_k = vars[k * 4 + 2];
            AD<double> v_k = vars[k * 4 + 3];
            AD<double> x_kp1 = vars[(k + 1) * 4];
            AD<double> y_kp1 = vars[(k + 1) * 4 + 1];
            AD<double> psi_kp1 = vars[(k + 1) * 4 + 2];
            AD<double> v_kp1 = vars[(k + 1) * 4 + 3];
            AD<double> delta_k = vars[n_states + k * 2];
            AD<double> a_k = vars[n_states + k * 2 + 1];

            fg[1 + k * 4] = x_kp1 - (x_k + v_k * CppAD::cos(psi_k) * DT);
            fg[1 + k * 4 + 1] = y_kp1 - (y_k + v_k * CppAD::sin(psi_k) * DT);
            fg[1 + k * 4 + 2] = psi_kp1 - (psi_k + (v_k / L) * CppAD::tan(delta_k) * DT);
            fg[1 + k * 4 + 3] = v_kp1 - (v_k + a_k * DT);
        }
    }
};

/**
 * @brief Solves the MPC optimization problem.
 * @param current_state The current state of the vehicle.
 * @param ref_traj The reference trajectory.
 * @return The optimal control input [delta, a].
 */
Vector2d solve_mpc(const Vector4d& current_state, const std::vector<Vector4d>& ref_traj) {
    size_t n_vars = 4 * (N + 1) + 2 * N;
    size_t n_constraints = 4 * N;
    CppAD::vector<double> vars(n_vars);
    CppAD::vector<double> vars_lower(n_vars), vars_upper(n_vars);
    CppAD::vector<double> constraints_lower(n_constraints), constraints_upper(n_constraints);

    for (int k = 0; k <= N; ++k) {
        if (k == 0) {
            vars[k * 4] = current_state(0);
            vars[k * 4 + 1] = current_state(1);
            vars[k * 4 + 2] = current_state(2);
            vars[k * 4 + 3] = current_state(3);
        } else {
            vars[k * 4] = current_state(0);
            vars[k * 4 + 1] = current_state(1);
            vars[k * 4 + 2] = current_state(2);
            vars[k * 4 + 3] = current_state(3);
        }
        vars_lower[k * 4] = -1e19; vars_upper[k * 4] = 1e19;
        vars_lower[k * 4 + 1] = -1e19; vars_upper[k * 4 + 1] = 1e19;
        vars_lower[k * 4 + 2] = -1e19; vars_upper[k * 4 + 2] = 1e19;
        vars_lower[k * 4 + 3] = 0.0; vars_upper[k * 4 + 3] = V_MAX;
    }

    for (int k = 0; k < N; ++k) {
        vars[4 * (N + 1) + k * 2] = 0.0;
        vars[4 * (N + 1) + k * 2 + 1] = 0.0;
        vars_lower[4 * (N + 1) + k * 2] = -DELTA_MAX;
        vars_upper[4 * (N + 1) + k * 2] = DELTA_MAX;
        vars_lower[4 * (N + 1) + k * 2 + 1] = -A_MAX;
        vars_upper[4 * (N + 1) + k * 2 + 1] = A_MAX;
    }

    for (int k = 0; k < n_constraints; ++k) {
        constraints_lower[k] = 0.0;
        constraints_upper[k] = 0.0;
    }

    std::string options;
    options += "Integer print_level 0\n";
    options += "String sb yes\n";
    options += "Numeric max_cpu_time 0.01\n";

    MPC mpc(ref_traj);
    CppAD::ipopt::solve_result<CppAD::vector<double>> solution;
    CppAD::ipopt::solve(options, vars, vars_lower, vars_upper, constraints_lower, constraints_upper, mpc, solution);

    if (solution.status != CppAD::ipopt::solve_result<CppAD::vector<double>>::success) {
        std::cerr << "Ipopt failed to converge!" << std::endl;
        return Vector2d::Zero();
    }

    Vector2d control;
    control << solution.x[4 * (N + 1)], solution.x[4 * (N + 1) + 1];
    return control;
}

/**
 * @brief Gets the reference trajectory from an ML model via a socket connection.
 * @param current_state The current state of the vehicle.
 * @return The reference trajectory.
 */
std::vector<Vector4d> get_reference_trajectory(const Vector4d& current_state) {
    static SocketClient client("127.0.0.1", 12345);
    try {
        return client.receive_trajectory();
    } catch (const std::exception& e) {
        std::cerr << "Socket error: " << e.what() << std::endl;
        // Fallback: Return a simple trajectory
        std::vector<Vector4d> ref_traj(N);
        for (int k = 0; k < N; ++k) {
            ref_traj[k] << current_state(0) + k * DT * current_state(3),
                           current_state(1) + 0.1 * k,
                           0.0,
                           1.0;
        }
        return ref_traj;
    }
}

/**
 * @brief Main function to run the MPC control loop.
 * @return Exit status.
 */
int main() {
    Vector4d state(0.0, 0.0, 0.0, 1.0);

    for (int i = 0; i < 100; ++i) {
        std::vector<Vector4d> ref_traj = get_reference_trajectory(state);
        Vector2d control = solve_mpc(state, ref_traj);
        std::cout << "Step " << i << ": delta = " << control(0) << ", a = " << control(1) << std::endl;
        state = dynamics(state, control);
    }

    return 0;
}