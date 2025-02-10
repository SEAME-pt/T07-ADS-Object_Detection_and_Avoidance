

#include "vehicle/generated/core/vehiclediagnostics.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleDiagnosticsThreadSafeDecorator::VehicleDiagnosticsThreadSafeDecorator(std::shared_ptr<IVehicleDiagnostics> impl)
    : m_impl(impl)
{
}
void VehicleDiagnosticsThreadSafeDecorator::setDtcCount(int32_t DTCCount)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcCountMutex);
    m_impl->setDtcCount(DTCCount);
}

int32_t VehicleDiagnosticsThreadSafeDecorator::getDtcCount() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dtcCountMutex);
    return m_impl->getDtcCount();
}
void VehicleDiagnosticsThreadSafeDecorator::setDtcList(const std::list<std::string>& DTCList)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcListMutex);
    m_impl->setDtcList(DTCList);
}

const std::list<std::string>& VehicleDiagnosticsThreadSafeDecorator::getDtcList() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_dtcListMutex);
    return m_impl->getDtcList();
}

IVehicleDiagnosticsPublisher& VehicleDiagnosticsThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}