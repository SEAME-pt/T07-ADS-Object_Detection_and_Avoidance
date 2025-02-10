

#include "vehicle/generated/core/vehicledriver.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleDriverThreadSafeDecorator::VehicleDriverThreadSafeDecorator(std::shared_ptr<IVehicleDriver> impl)
    : m_impl(impl)
{
}
void VehicleDriverThreadSafeDecorator::setDistractionLevel(float DistractionLevel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_distractionLevelMutex);
    m_impl->setDistractionLevel(DistractionLevel);
}

float VehicleDriverThreadSafeDecorator::getDistractionLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_distractionLevelMutex);
    return m_impl->getDistractionLevel();
}
void VehicleDriverThreadSafeDecorator::setIsEyesOnRoad(bool IsEyesOnRoad)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isEyesOnRoadMutex);
    m_impl->setIsEyesOnRoad(IsEyesOnRoad);
}

bool VehicleDriverThreadSafeDecorator::getIsEyesOnRoad() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isEyesOnRoadMutex);
    return m_impl->getIsEyesOnRoad();
}
void VehicleDriverThreadSafeDecorator::setIsHandsOnWheel(bool IsHandsOnWheel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isHandsOnWheelMutex);
    m_impl->setIsHandsOnWheel(IsHandsOnWheel);
}

bool VehicleDriverThreadSafeDecorator::getIsHandsOnWheel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isHandsOnWheelMutex);
    return m_impl->getIsHandsOnWheel();
}
void VehicleDriverThreadSafeDecorator::setAttentiveProbability(float AttentiveProbability)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_attentiveProbabilityMutex);
    m_impl->setAttentiveProbability(AttentiveProbability);
}

float VehicleDriverThreadSafeDecorator::getAttentiveProbability() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_attentiveProbabilityMutex);
    return m_impl->getAttentiveProbability();
}
void VehicleDriverThreadSafeDecorator::setFatigueLevel(float FatigueLevel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_fatigueLevelMutex);
    m_impl->setFatigueLevel(FatigueLevel);
}

float VehicleDriverThreadSafeDecorator::getFatigueLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_fatigueLevelMutex);
    return m_impl->getFatigueLevel();
}
void VehicleDriverThreadSafeDecorator::setHeartRate(int32_t HeartRate)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_heartRateMutex);
    m_impl->setHeartRate(HeartRate);
}

int32_t VehicleDriverThreadSafeDecorator::getHeartRate() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_heartRateMutex);
    return m_impl->getHeartRate();
}

IVehicleDriverPublisher& VehicleDriverThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}