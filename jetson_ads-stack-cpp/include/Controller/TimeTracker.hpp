#ifndef TIME_TRACKER_HPP
#define TIME_TRACKER_HPP

#include <chrono>

class TimeTracker {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    TimeTracker() : ts0(), ts1(), dt(0) {}

    void mark() {
        ts0 = std::chrono::high_resolution_clock::now();
        if (ts1.time_since_epoch().count() != 0) {
            dt = std::chrono::duration_cast<std::chrono::microseconds>(ts0 - ts1).count();
        }
        ts1 = ts0;
    }

    long long delta() const {
        return dt;
    }

private:
    TimePoint ts0;
    TimePoint ts1;
    long long dt;
};

#endif // TIME_TRACKER_HPP

