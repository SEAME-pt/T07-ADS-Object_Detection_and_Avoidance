

#include "vehicle/generated/core/vehicledriveridentifierdeprecated.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleDriverIdentifierDeprecatedThreadSafeDecorator::VehicleDriverIdentifierDeprecatedThreadSafeDecorator(std::shared_ptr<IVehicleDriverIdentifierDeprecated> impl)
    : m_impl(impl)
{
}
void VehicleDriverIdentifierDeprecatedThreadSafeDecorator::setSubjectDeprecated(const std::string& Subject_Deprecated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_subjectDeprecatedMutex);
    m_impl->setSubjectDeprecated(Subject_Deprecated);
}

const std::string& VehicleDriverIdentifierDeprecatedThreadSafeDecorator::getSubjectDeprecated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_subjectDeprecatedMutex);
    return m_impl->getSubjectDeprecated();
}
void VehicleDriverIdentifierDeprecatedThreadSafeDecorator::setIssuerDeprecated(const std::string& Issuer_Deprecated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_issuerDeprecatedMutex);
    m_impl->setIssuerDeprecated(Issuer_Deprecated);
}

const std::string& VehicleDriverIdentifierDeprecatedThreadSafeDecorator::getIssuerDeprecated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_issuerDeprecatedMutex);
    return m_impl->getIssuerDeprecated();
}

IVehicleDriverIdentifierDeprecatedPublisher& VehicleDriverIdentifierDeprecatedThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}