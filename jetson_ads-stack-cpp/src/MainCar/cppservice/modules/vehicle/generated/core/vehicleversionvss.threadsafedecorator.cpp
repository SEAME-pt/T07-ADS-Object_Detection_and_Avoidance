

#include "vehicle/generated/core/vehicleversionvss.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleVersionVssThreadSafeDecorator::VehicleVersionVssThreadSafeDecorator(std::shared_ptr<IVehicleVersionVss> impl)
    : m_impl(impl)
{
}
void VehicleVersionVssThreadSafeDecorator::setMajor(int32_t Major)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_majorMutex);
    m_impl->setMajor(Major);
}

int32_t VehicleVersionVssThreadSafeDecorator::getMajor() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_majorMutex);
    return m_impl->getMajor();
}
void VehicleVersionVssThreadSafeDecorator::setMinor(int32_t Minor)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minorMutex);
    m_impl->setMinor(Minor);
}

int32_t VehicleVersionVssThreadSafeDecorator::getMinor() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_minorMutex);
    return m_impl->getMinor();
}
void VehicleVersionVssThreadSafeDecorator::setPatch(int32_t Patch)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_patchMutex);
    m_impl->setPatch(Patch);
}

int32_t VehicleVersionVssThreadSafeDecorator::getPatch() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_patchMutex);
    return m_impl->getPatch();
}
void VehicleVersionVssThreadSafeDecorator::setLabel(const std::string& Label)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_labelMutex);
    m_impl->setLabel(Label);
}

const std::string& VehicleVersionVssThreadSafeDecorator::getLabel() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_labelMutex);
    return m_impl->getLabel();
}

IVehicleVersionVssPublisher& VehicleVersionVssThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}