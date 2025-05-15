
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleDiagnosticsThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Diagnostics interface.
* @see Vehicle_Diagnostics
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleDiagnostics> testVehicleDiagnostics = std::make_unique<VehicleDiagnosticsThreadSafeDecorator>(std::make_shared<VehicleDiagnostics>());

// Thread safe access
auto dtcCount = testVehicleDiagnostics->getDtcCount();
testVehicleDiagnostics->setDtcCount(0);
auto dtcList = testVehicleDiagnostics->getDtcList();
testVehicleDiagnostics->setDtcList(std::list<std::string>());
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleDiagnosticsThreadSafeDecorator : public IVehicleDiagnostics
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Diagnostics object to make thread safe.
    */
    explicit VehicleDiagnosticsThreadSafeDecorator(std::shared_ptr<IVehicleDiagnostics> impl);

    /** Guards and forwards call to Vehicle_Diagnostics implementation. */
    void setDtcCount(int32_t DTCCount) override;
    /** Guards and forwards call to Vehicle_Diagnostics implementation. */
    int32_t getDtcCount() const override;

    /** Guards and forwards call to Vehicle_Diagnostics implementation. */
    void setDtcList(const std::list<std::string>& DTCList) override;
    /** Guards and forwards call to Vehicle_Diagnostics implementation. */
    const std::list<std::string>& getDtcList() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Diagnostics changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Diagnostics.
    */
    IVehicleDiagnosticsPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Diagnostics object which is guarded */
    std::shared_ptr<IVehicleDiagnostics> m_impl;
    // Mutex for dtcCount property
    mutable std::shared_timed_mutex m_dtcCountMutex;
    // Mutex for dtcList property
    mutable std::shared_timed_mutex m_dtcListMutex;
};
} // namespace Vehicle
} // namespace Cpp
