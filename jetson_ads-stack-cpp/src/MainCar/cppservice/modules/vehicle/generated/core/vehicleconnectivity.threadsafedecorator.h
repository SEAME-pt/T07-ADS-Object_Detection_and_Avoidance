
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleConnectivityThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Connectivity interface.
* @see Vehicle_Connectivity
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleConnectivity> testVehicleConnectivity = std::make_unique<VehicleConnectivityThreadSafeDecorator>(std::make_shared<VehicleConnectivity>());

// Thread safe access
auto isConnectivityAvailable = testVehicleConnectivity->getIsConnectivityAvailable();
testVehicleConnectivity->setIsConnectivityAvailable(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleConnectivityThreadSafeDecorator : public IVehicleConnectivity
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Connectivity object to make thread safe.
    */
    explicit VehicleConnectivityThreadSafeDecorator(std::shared_ptr<IVehicleConnectivity> impl);

    /** Guards and forwards call to Vehicle_Connectivity implementation. */
    void setIsConnectivityAvailable(bool IsConnectivityAvailable) override;
    /** Guards and forwards call to Vehicle_Connectivity implementation. */
    bool getIsConnectivityAvailable() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Connectivity changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Connectivity.
    */
    IVehicleConnectivityPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Connectivity object which is guarded */
    std::shared_ptr<IVehicleConnectivity> m_impl;
    // Mutex for isConnectivityAvailable property
    mutable std::shared_timed_mutex m_isConnectivityAvailableMutex;
};
} // namespace Vehicle
} // namespace Cpp
