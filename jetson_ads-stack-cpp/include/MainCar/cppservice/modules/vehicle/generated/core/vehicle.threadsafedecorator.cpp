

#include "vehicle/generated/core/vehicle.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleThreadSafeDecorator::VehicleThreadSafeDecorator(std::shared_ptr<IVehicle> impl)
    : m_impl(impl)
{
}
void VehicleThreadSafeDecorator::setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lowVoltageSystemStateMutex);
    m_impl->setLowVoltageSystemState(LowVoltageSystemState);
}

Vehicle_LowVoltageSystemState_ValueEnum VehicleThreadSafeDecorator::getLowVoltageSystemState() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_lowVoltageSystemStateMutex);
    return m_impl->getLowVoltageSystemState();
}
void VehicleThreadSafeDecorator::setSpeed(float Speed)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_speedMutex);
    m_impl->setSpeed(Speed);
}

float VehicleThreadSafeDecorator::getSpeed() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_speedMutex);
    return m_impl->getSpeed();
}
void VehicleThreadSafeDecorator::setTraveledDistance(float TraveledDistance)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceMutex);
    m_impl->setTraveledDistance(TraveledDistance);
}

float VehicleThreadSafeDecorator::getTraveledDistance() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_traveledDistanceMutex);
    return m_impl->getTraveledDistance();
}
void VehicleThreadSafeDecorator::setTraveledDistanceSinceStart(float TraveledDistanceSinceStart)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceSinceStartMutex);
    m_impl->setTraveledDistanceSinceStart(TraveledDistanceSinceStart);
}

float VehicleThreadSafeDecorator::getTraveledDistanceSinceStart() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_traveledDistanceSinceStartMutex);
    return m_impl->getTraveledDistanceSinceStart();
}
void VehicleThreadSafeDecorator::setStartTime(const std::string& StartTime)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeMutex);
    m_impl->setStartTime(StartTime);
}

const std::string& VehicleThreadSafeDecorator::getStartTime() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_startTimeMutex);
    return m_impl->getStartTime();
}
void VehicleThreadSafeDecorator::setTripDuration(float TripDuration)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_tripDurationMutex);
    m_impl->setTripDuration(TripDuration);
}

float VehicleThreadSafeDecorator::getTripDuration() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_tripDurationMutex);
    return m_impl->getTripDuration();
}
void VehicleThreadSafeDecorator::setTripMeterReading(float TripMeterReading)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_tripMeterReadingMutex);
    m_impl->setTripMeterReading(TripMeterReading);
}

float VehicleThreadSafeDecorator::getTripMeterReading() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_tripMeterReadingMutex);
    return m_impl->getTripMeterReading();
}
void VehicleThreadSafeDecorator::setIsBrokenDown(bool IsBrokenDown)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isBrokenDownMutex);
    m_impl->setIsBrokenDown(IsBrokenDown);
}

bool VehicleThreadSafeDecorator::getIsBrokenDown() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isBrokenDownMutex);
    return m_impl->getIsBrokenDown();
}
void VehicleThreadSafeDecorator::setIsMoving(bool IsMoving)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isMovingMutex);
    m_impl->setIsMoving(IsMoving);
}

bool VehicleThreadSafeDecorator::getIsMoving() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isMovingMutex);
    return m_impl->getIsMoving();
}
void VehicleThreadSafeDecorator::setAverageSpeed(float AverageSpeed)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averageSpeedMutex);
    m_impl->setAverageSpeed(AverageSpeed);
}

float VehicleThreadSafeDecorator::getAverageSpeed() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_averageSpeedMutex);
    return m_impl->getAverageSpeed();
}
void VehicleThreadSafeDecorator::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    m_impl->setPowerOptimizeLevel(PowerOptimizeLevel);
}

int32_t VehicleThreadSafeDecorator::getPowerOptimizeLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    return m_impl->getPowerOptimizeLevel();
}
void VehicleThreadSafeDecorator::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    m_impl->setIsAutoPowerOptimize(IsAutoPowerOptimize);
}

bool VehicleThreadSafeDecorator::getIsAutoPowerOptimize() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    return m_impl->getIsAutoPowerOptimize();
}

IVehiclePublisher& VehicleThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}