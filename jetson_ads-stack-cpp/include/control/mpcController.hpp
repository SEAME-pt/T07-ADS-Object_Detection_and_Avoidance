#ifndef MPC_HPP
#define MPC_HPP

#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

using CppAD::AD;
using Eigen::Vector3d;
using Eigen::Vector2d;

// JetRacer parameters
const double L = 0.15;				// Wheelbase (m)
const double DT = 0.03;				// Time step (s)
const int N = 10;					// Prediction horizon
const double V_MAX = 2.0;			// Max velocity (m/s)
const double DELTA_MAX = 0.523;		// Max steering angle (rad, 30 deg)
const double A_MAX = 2.0;			// Max acceleration (m/s^2)
const double DELTA_RATE_MAX = 0.1;	// Max steering rate (rad/step)
const double V_REF = 1.0;			// Reference velocity (m/s)

// MPC weights
const double Q_EY = 100.0;			// Weight for cross-track error
const double Q_PSI_ERR = 10.0;		// Weight for heading error
const double Q_V = 1.0;				// Weight for velocity error
const double R_DELTA = 1.0;			// Weight for steering effort
const double R_A = 1.0;				// Weight for acceleration effort

class MPC {
public:
    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

private:
    double prev_delta;

public:
    // Canonical form
    MPC(double delta = 0.0);
    MPC(const MPC& other);
    MPC& operator=(const MPC& other);
    virtual ~MPC();

    // Optimization operator
    void operator()(ADvector& fg, const ADvector& vars);
	Vector3d dynamics(const Vector3d& state, const Vector2d& input);
	Vector2d solve_mpc(const Vector3d& current_state, double prev_delta);

};

#endif // MPC_HPP