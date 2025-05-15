

#include "vehicle/implementation/vehicledriveridentifierdeprecated.h"
#include "vehicle/generated/core/vehicledriveridentifierdeprecated.publisher.h"
#include "vehicle/generated/core/vehicledriveridentifierdeprecated.data.h"

using namespace Cpp::Vehicle;

VehicleDriverIdentifierDeprecated::VehicleDriverIdentifierDeprecated()
    : m_publisher(std::make_unique<VehicleDriverIdentifierDeprecatedPublisher>())
{
}
VehicleDriverIdentifierDeprecated::~VehicleDriverIdentifierDeprecated()
{
}

void VehicleDriverIdentifierDeprecated::setSubjectDeprecated(const std::string& Subject_Deprecated)
{
    if (m_data.m_Subject_Deprecated != Subject_Deprecated) {
        m_data.m_Subject_Deprecated = Subject_Deprecated;
        m_publisher->publishSubjectDeprecatedChanged(Subject_Deprecated);
    }
}

const std::string& VehicleDriverIdentifierDeprecated::getSubjectDeprecated() const
{
    return m_data.m_Subject_Deprecated;
}

void VehicleDriverIdentifierDeprecated::setIssuerDeprecated(const std::string& Issuer_Deprecated)
{
    if (m_data.m_Issuer_Deprecated != Issuer_Deprecated) {
        m_data.m_Issuer_Deprecated = Issuer_Deprecated;
        m_publisher->publishIssuerDeprecatedChanged(Issuer_Deprecated);
    }
}

const std::string& VehicleDriverIdentifierDeprecated::getIssuerDeprecated() const
{
    return m_data.m_Issuer_Deprecated;
}

IVehicleDriverIdentifierDeprecatedPublisher& VehicleDriverIdentifierDeprecated::_getPublisher() const
{
    return *m_publisher;
}
