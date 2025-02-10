

#include "vehicle/implementation/vehicledriver.h"
#include "vehicle/generated/core/vehicledriver.publisher.h"
#include "vehicle/generated/core/vehicledriver.data.h"

using namespace Cpp::Vehicle;

VehicleDriver::VehicleDriver()
    : m_publisher(std::make_unique<VehicleDriverPublisher>())
{
}
VehicleDriver::~VehicleDriver()
{
}

void VehicleDriver::setDistractionLevel(float DistractionLevel)
{
    if (m_data.m_DistractionLevel != DistractionLevel) {
        m_data.m_DistractionLevel = DistractionLevel;
        m_publisher->publishDistractionLevelChanged(DistractionLevel);
    }
}

float VehicleDriver::getDistractionLevel() const
{
    return m_data.m_DistractionLevel;
}

void VehicleDriver::setIsEyesOnRoad(bool IsEyesOnRoad)
{
    if (m_data.m_IsEyesOnRoad != IsEyesOnRoad) {
        m_data.m_IsEyesOnRoad = IsEyesOnRoad;
        m_publisher->publishIsEyesOnRoadChanged(IsEyesOnRoad);
    }
}

bool VehicleDriver::getIsEyesOnRoad() const
{
    return m_data.m_IsEyesOnRoad;
}

void VehicleDriver::setIsHandsOnWheel(bool IsHandsOnWheel)
{
    if (m_data.m_IsHandsOnWheel != IsHandsOnWheel) {
        m_data.m_IsHandsOnWheel = IsHandsOnWheel;
        m_publisher->publishIsHandsOnWheelChanged(IsHandsOnWheel);
    }
}

bool VehicleDriver::getIsHandsOnWheel() const
{
    return m_data.m_IsHandsOnWheel;
}

void VehicleDriver::setAttentiveProbability(float AttentiveProbability)
{
    if (m_data.m_AttentiveProbability != AttentiveProbability) {
        m_data.m_AttentiveProbability = AttentiveProbability;
        m_publisher->publishAttentiveProbabilityChanged(AttentiveProbability);
    }
}

float VehicleDriver::getAttentiveProbability() const
{
    return m_data.m_AttentiveProbability;
}

void VehicleDriver::setFatigueLevel(float FatigueLevel)
{
    if (m_data.m_FatigueLevel != FatigueLevel) {
        m_data.m_FatigueLevel = FatigueLevel;
        m_publisher->publishFatigueLevelChanged(FatigueLevel);
    }
}

float VehicleDriver::getFatigueLevel() const
{
    return m_data.m_FatigueLevel;
}

void VehicleDriver::setHeartRate(int32_t HeartRate)
{
    if (m_data.m_HeartRate != HeartRate) {
        m_data.m_HeartRate = HeartRate;
        m_publisher->publishHeartRateChanged(HeartRate);
    }
}

int32_t VehicleDriver::getHeartRate() const
{
    return m_data.m_HeartRate;
}

IVehicleDriverPublisher& VehicleDriver::_getPublisher() const
{
    return *m_publisher;
}
