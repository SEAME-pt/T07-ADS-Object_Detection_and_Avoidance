#ifndef MPC_HPP
#define MPC_HPP

#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>

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
};

#endif // MPC_HPP