// SpeedSubscriber.hpp
#pragma once
#include <zmq.hpp>
#include <thread>
#include <atomic>
#include <functional>
#include <string>

class SpeedSubscriber {
public:
    SpeedSubscriber(const std::string& address = "tcp://localhost:5555")
        : context(1), subscriber(context, zmq::socket_type::sub), running(false) {
        subscriber.connect(address);
        subscriber.setsockopt(ZMQ_SUBSCRIBE, "speed", 5);
    }

    void start(std::function<void(float)> callback) {
        running = true;
        listenThread = std::thread([this, callback]() {
            while (running) {
                zmq::message_t message;
                if (subscriber.recv(message, zmq::recv_flags::none)) {
                    std::string msgStr(static_cast<char*>(message.data()), message.size());
                    auto delimiterPos = msgStr.find(' ');
                    if (delimiterPos != std::string::npos) {
                        std::string key = msgStr.substr(0, delimiterPos);
                        std::string valueStr = msgStr.substr(delimiterPos + 1);
                        try {
                            float value = std::stof(valueStr);
                            callback(value);
                        } catch (...) {}
                    }
                }
            }
        });
    }

    void stop() {
        running = false;
        if (listenThread.joinable()) {
            listenThread.join();
        }
    }

    ~SpeedSubscriber() {
        stop();
    }

private:
    zmq::context_t context;
    zmq::socket_t subscriber;
    std::thread listenThread;
    std::atomic<bool> running;
};
