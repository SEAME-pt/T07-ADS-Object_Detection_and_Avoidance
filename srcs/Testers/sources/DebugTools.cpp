#include "DebugTools.hpp"
#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include <iostream>
#include <fstream>   // For file handling
#include <chrono>
#include <atomic>

// Measures the time to reach a target speed in km/h
void DebugTools::measureAccelerationTime(JetCar& car, int targetSpeed) {
    using namespace std::chrono;
    std::atomic<bool> targetReached = false;
    std::atomic<float> currentSpeed = 0.0f;

    SpeedSubscriber subscriber;
    subscriber.start([&](float speed) {
        currentSpeed = speed;
        if (speed >= targetSpeed) {
            targetReached = true;
        }
    });

    std::cout << "Starting measurement until reaching " << targetSpeed << " km/h\n";

    // auto startTime = high_resolution_clock::now();
    car.set_motor_speed(100); // Max PWM

    while (!targetReached) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        // save the current speed to a file
        std::ofstream logFile("speed_log.csv", std::ios::app); // Append mode
        if (logFile.is_open()) {
            logFile << currentSpeed << "\n";
            logFile.close();
        } else {
            std::cerr << "Failed to open speed log file!\n";
        }
    }

    // auto endTime = high_resolution_clock::now();
    car.set_motor_speed(0);
    subscriber.stop();
}

void DebugTools::autoTunePID(JetCar& car) {
    float dt = 0.03f;
    float sim_time = 10.0f;
    float v_target = 2.0f;
    std::atomic<float> currentSpeed = 0.0f;

    std::cout << "Starting auto-tune PID...\n";
    auto [kp, ki, kd] = auto_tune_pid(dt, sim_time, v_target);

    std::cout << "PID Tuning completed:\n";
    std::cout << "Kp = " << kp << "\n";
    std::cout << "Ki = " << ki << "\n";
    std::cout << "Kd = " << kd << "\n";

    // Salva os resultados em arquivo
    // std::ofstream outFile("pid_gains.txt");
    // if (outFile.is_open()) {
    //     outFile << "Kp: " << kp << "\n";
    //     outFile << "Ki: " << ki << "\n";
    //     outFile << "Kd: " << kd << "\n";
    //     outFile.close();
    // } else {
    //     std::cerr << "Error saving PID gains.\n";
    // }
}
