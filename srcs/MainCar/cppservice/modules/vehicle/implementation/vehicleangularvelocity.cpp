

#include "vehicle/implementation/vehicleangularvelocity.h"
#include "vehicle/generated/core/vehicleangularvelocity.publisher.h"
#include "vehicle/generated/core/vehicleangularvelocity.data.h"

using namespace Cpp::Vehicle;

VehicleAngularVelocity::VehicleAngularVelocity()
    : m_publisher(std::make_unique<VehicleAngularVelocityPublisher>())
{
}
VehicleAngularVelocity::~VehicleAngularVelocity()
{
}

void VehicleAngularVelocity::setRoll(float Roll)
{
    if (m_data.m_Roll != Roll) {
        m_data.m_Roll = Roll;
        m_publisher->publishRollChanged(Roll);
    }
}

float VehicleAngularVelocity::getRoll() const
{
    return m_data.m_Roll;
}

void VehicleAngularVelocity::setPitch(float Pitch)
{
    if (m_data.m_Pitch != Pitch) {
        m_data.m_Pitch = Pitch;
        m_publisher->publishPitchChanged(Pitch);
    }
}

float VehicleAngularVelocity::getPitch() const
{
    return m_data.m_Pitch;
}

void VehicleAngularVelocity::setYaw(float Yaw)
{
    if (m_data.m_Yaw != Yaw) {
        m_data.m_Yaw = Yaw;
        m_publisher->publishYawChanged(Yaw);
    }
}

float VehicleAngularVelocity::getYaw() const
{
    return m_data.m_Yaw;
}

IVehicleAngularVelocityPublisher& VehicleAngularVelocity::_getPublisher() const
{
    return *m_publisher;
}
