
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsHazardThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Hazard interface.
* @see Vehicle_Body_Lights_Hazard
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsHazard> testVehicleBodyLightsHazard = std::make_unique<VehicleBodyLightsHazardThreadSafeDecorator>(std::make_shared<VehicleBodyLightsHazard>());

// Thread safe access
auto isSignaling = testVehicleBodyLightsHazard->getIsSignaling();
testVehicleBodyLightsHazard->setIsSignaling(false);
auto isDefect = testVehicleBodyLightsHazard->getIsDefect();
testVehicleBodyLightsHazard->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsHazardThreadSafeDecorator : public IVehicleBodyLightsHazard
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Hazard object to make thread safe.
    */
    explicit VehicleBodyLightsHazardThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsHazard> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Hazard implementation. */
    void setIsSignaling(bool IsSignaling) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Hazard implementation. */
    bool getIsSignaling() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Hazard implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Hazard implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Hazard changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Hazard.
    */
    IVehicleBodyLightsHazardPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Hazard object which is guarded */
    std::shared_ptr<IVehicleBodyLightsHazard> m_impl;
    // Mutex for isSignaling property
    mutable std::shared_timed_mutex m_isSignalingMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
