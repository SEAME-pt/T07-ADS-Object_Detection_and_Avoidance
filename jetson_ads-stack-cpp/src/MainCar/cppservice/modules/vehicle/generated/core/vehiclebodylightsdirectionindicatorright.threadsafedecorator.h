
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_DirectionIndicator_Right interface.
* @see Vehicle_Body_Lights_DirectionIndicator_Right
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsDirectionIndicatorRight> testVehicleBodyLightsDirectionIndicatorRight = std::make_unique<VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator>(std::make_shared<VehicleBodyLightsDirectionIndicatorRight>());

// Thread safe access
auto isSignaling = testVehicleBodyLightsDirectionIndicatorRight->getIsSignaling();
testVehicleBodyLightsDirectionIndicatorRight->setIsSignaling(false);
auto isDefect = testVehicleBodyLightsDirectionIndicatorRight->getIsDefect();
testVehicleBodyLightsDirectionIndicatorRight->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator : public IVehicleBodyLightsDirectionIndicatorRight
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_DirectionIndicator_Right object to make thread safe.
    */
    explicit VehicleBodyLightsDirectionIndicatorRightThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsDirectionIndicatorRight> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Right implementation. */
    void setIsSignaling(bool IsSignaling) override;
    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Right implementation. */
    bool getIsSignaling() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Right implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Right implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_DirectionIndicator_Right changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_DirectionIndicator_Right.
    */
    IVehicleBodyLightsDirectionIndicatorRightPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_DirectionIndicator_Right object which is guarded */
    std::shared_ptr<IVehicleBodyLightsDirectionIndicatorRight> m_impl;
    // Mutex for isSignaling property
    mutable std::shared_timed_mutex m_isSignalingMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
