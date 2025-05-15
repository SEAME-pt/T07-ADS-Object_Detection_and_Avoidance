

#include "vehicle/generated/core/vehicleangularvelocity.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleAngularVelocityThreadSafeDecorator::VehicleAngularVelocityThreadSafeDecorator(std::shared_ptr<IVehicleAngularVelocity> impl)
    : m_impl(impl)
{
}
void VehicleAngularVelocityThreadSafeDecorator::setRoll(float Roll)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rollMutex);
    m_impl->setRoll(Roll);
}

float VehicleAngularVelocityThreadSafeDecorator::getRoll() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rollMutex);
    return m_impl->getRoll();
}
void VehicleAngularVelocityThreadSafeDecorator::setPitch(float Pitch)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_pitchMutex);
    m_impl->setPitch(Pitch);
}

float VehicleAngularVelocityThreadSafeDecorator::getPitch() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_pitchMutex);
    return m_impl->getPitch();
}
void VehicleAngularVelocityThreadSafeDecorator::setYaw(float Yaw)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yawMutex);
    m_impl->setYaw(Yaw);
}

float VehicleAngularVelocityThreadSafeDecorator::getYaw() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_yawMutex);
    return m_impl->getYaw();
}

IVehicleAngularVelocityPublisher& VehicleAngularVelocityThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}