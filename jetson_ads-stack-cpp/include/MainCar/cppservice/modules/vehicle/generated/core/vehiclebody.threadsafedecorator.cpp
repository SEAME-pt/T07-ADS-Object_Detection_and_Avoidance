

#include "vehicle/generated/core/vehiclebody.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyThreadSafeDecorator::VehicleBodyThreadSafeDecorator(std::shared_ptr<IVehicleBody> impl)
    : m_impl(impl)
{
}
void VehicleBodyThreadSafeDecorator::setPowerOptimizeLevel(int32_t PowerOptimizeLevel)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    m_impl->setPowerOptimizeLevel(PowerOptimizeLevel);
}

int32_t VehicleBodyThreadSafeDecorator::getPowerOptimizeLevel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelMutex);
    return m_impl->getPowerOptimizeLevel();
}
void VehicleBodyThreadSafeDecorator::setIsAutoPowerOptimize(bool IsAutoPowerOptimize)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    m_impl->setIsAutoPowerOptimize(IsAutoPowerOptimize);
}

bool VehicleBodyThreadSafeDecorator::getIsAutoPowerOptimize() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeMutex);
    return m_impl->getIsAutoPowerOptimize();
}

IVehicleBodyPublisher& VehicleBodyThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}