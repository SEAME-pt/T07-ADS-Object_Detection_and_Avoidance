

#include "vehicle/implementation/vehicle.h"
#include "vehicle/generated/core/vehicle.publisher.h"
#include "vehicle/generated/core/vehicle.data.h"

using namespace Cpp::Vehicle;

Vehicle::Vehicle()
    : m_publisher(std::make_unique<VehiclePublisher>())
{
}
Vehicle::~Vehicle()
{
}

void Vehicle::setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState)
{
    if (m_data.m_LowVoltageSystemState != LowVoltageSystemState) {
        m_data.m_LowVoltageSystemState = LowVoltageSystemState;
        m_publisher->publishLowVoltageSystemStateChanged(LowVoltageSystemState);
    }
}

Vehicle_LowVoltageSystemState_ValueEnum Vehicle::getLowVoltageSystemState() const
{
    return m_data.m_LowVoltageSystemState;
}

void Vehicle::setSpeed(float Speed)
{
    if (m_data.m_Speed != Speed) {
        m_data.m_Speed = Speed;
        m_publisher->publishSpeedChanged(Speed);
    }
}

float Vehicle::getSpeed() const
{
    return m_data.m_Speed;
}

void Vehicle::setTraveledDistance(float TraveledDistance)
{
    if (m_data.m_TraveledDistance != TraveledDistance) {
        m_data.m_TraveledDistance = TraveledDistance;
        m_publisher->publishTraveledDistanceChanged(TraveledDistance);
    }
}

float Vehicle::getTraveledDistance() const
{
    return m_data.m_TraveledDistance;
}

void Vehicle::setTraveledDistanceSinceStart(float TraveledDistanceSinceStart)
{
    if (m_data.m_TraveledDistanceSinceStart != TraveledDistanceSinceStart) {
        m_data.m_TraveledDistanceSinceStart = TraveledDistanceSinceStart;
        m_publisher->publishTraveledDistanceSinceStartChanged(TraveledDistanceSinceStart);
    }
}

float Vehicle::getTraveledDistanceSinceStart() const
{
    return m_data.m_TraveledDistanceSinceStart;
}

void Vehicle::setStartTime(const std::string& StartTime)
{
    if (m_data.m_StartTime != StartTime) {
        m_data.m_StartTime = StartTime;
        m_publisher->publishStartTimeChanged(StartTime);
    }
}

const std::string& Vehicle::getStartTime() const
{
    return m_data.m_StartTime;
}

void Vehicle::setTripDuration(float TripDuration)
{
    if (m_data.m_TripDuration != TripDuration) {
        m_data.m_TripDuration = TripDuration;
        m_publisher->publishTripDurationChanged(TripDuration);
    }
}

float Vehicle::getTripDuration() const
{
    return m_data.m_TripDuration;
}

void Vehicle::setTripMeterReading(float TripMeterReading)
{
    if (m_data.m_TripMeterReading != TripMeterReading) {
        m_data.m_TripMeterReading = TripMeterReading;
        m_publisher->publishTripMeterReadingChanged(TripMeterReading);
    }
}

float Vehicle::getTripMeterReading() const
{
    return m_data.m_TripMeterReading;
}

void Vehicle::setIsBrokenDown(bool IsBrokenDown)
{
    if (m_data.m_IsBrokenDown != IsBrokenDown) {
        m_data.m_IsBrokenDown = IsBrokenDown;
        m_publisher->publishIsBrokenDownChanged(IsBrokenDown);
    }
}

bool Vehicle::getIsBrokenDown() const
{
    return m_data.m_IsBrokenDown;
}

void Vehicle::setIsMoving(bool IsMoving)
{
    if (m_data.m_IsMoving != IsMoving) {
        m_data.m_IsMoving = IsMoving;
        m_publisher->publishIsMovingChanged(IsMoving);
    }
}

bool Vehicle::getIsMoving() const
{
    return m_data.m_IsMoving;
}

void Vehicle::setAverageSpeed(float AverageSpeed)
{
    if (m_data.m_AverageSpeed != AverageSpeed) {
        m_data.m_AverageSpeed = AverageSpeed;
        m_publisher->publishAverageSpeedChanged(AverageSpeed);
    }
}

float Vehicle::getAverageSpeed() const
{
    return m_data.m_AverageSpeed;
}

void Vehicle::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    if (m_data.m_PowerOptimizeLevel != PowerOptimizeLevel) {
        m_data.m_PowerOptimizeLevel = PowerOptimizeLevel;
        m_publisher->publishPowerOptimizeLevelChanged(PowerOptimizeLevel);
    }
}

int32_t Vehicle::getPowerOptimizeLevel() const
{
    return m_data.m_PowerOptimizeLevel;
}

void Vehicle::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    if (m_data.m_IsAutoPowerOptimize != IsAutoPowerOptimize) {
        m_data.m_IsAutoPowerOptimize = IsAutoPowerOptimize;
        m_publisher->publishIsAutoPowerOptimizeChanged(IsAutoPowerOptimize);
    }
}

bool Vehicle::getIsAutoPowerOptimize() const
{
    return m_data.m_IsAutoPowerOptimize;
}

IVehiclePublisher& Vehicle::_getPublisher() const
{
    return *m_publisher;
}
