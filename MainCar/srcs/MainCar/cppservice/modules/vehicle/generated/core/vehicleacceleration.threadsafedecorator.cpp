

#include "vehicle/generated/core/vehicleacceleration.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleAccelerationThreadSafeDecorator::VehicleAccelerationThreadSafeDecorator(std::shared_ptr<IVehicleAcceleration> impl)
    : m_impl(impl)
{
}
void VehicleAccelerationThreadSafeDecorator::setLongitudinal(float Longitudinal)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudinalMutex);
    m_impl->setLongitudinal(Longitudinal);
}

float VehicleAccelerationThreadSafeDecorator::getLongitudinal() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_longitudinalMutex);
    return m_impl->getLongitudinal();
}
void VehicleAccelerationThreadSafeDecorator::setLateral(float Lateral)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lateralMutex);
    m_impl->setLateral(Lateral);
}

float VehicleAccelerationThreadSafeDecorator::getLateral() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_lateralMutex);
    return m_impl->getLateral();
}
void VehicleAccelerationThreadSafeDecorator::setVertical(float Vertical)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalMutex);
    m_impl->setVertical(Vertical);
}

float VehicleAccelerationThreadSafeDecorator::getVertical() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_verticalMutex);
    return m_impl->getVertical();
}

IVehicleAccelerationPublisher& VehicleAccelerationThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}