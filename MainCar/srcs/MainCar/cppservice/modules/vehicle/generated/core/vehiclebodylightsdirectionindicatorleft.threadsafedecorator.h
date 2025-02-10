
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_DirectionIndicator_Left interface.
* @see Vehicle_Body_Lights_DirectionIndicator_Left
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsDirectionIndicatorLeft> testVehicleBodyLightsDirectionIndicatorLeft = std::make_unique<VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator>(std::make_shared<VehicleBodyLightsDirectionIndicatorLeft>());

// Thread safe access
auto isSignaling = testVehicleBodyLightsDirectionIndicatorLeft->getIsSignaling();
testVehicleBodyLightsDirectionIndicatorLeft->setIsSignaling(false);
auto isDefect = testVehicleBodyLightsDirectionIndicatorLeft->getIsDefect();
testVehicleBodyLightsDirectionIndicatorLeft->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator : public IVehicleBodyLightsDirectionIndicatorLeft
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_DirectionIndicator_Left object to make thread safe.
    */
    explicit VehicleBodyLightsDirectionIndicatorLeftThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsDirectionIndicatorLeft> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Left implementation. */
    void setIsSignaling(bool IsSignaling) override;
    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Left implementation. */
    bool getIsSignaling() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Left implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_DirectionIndicator_Left implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_DirectionIndicator_Left changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_DirectionIndicator_Left.
    */
    IVehicleBodyLightsDirectionIndicatorLeftPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_DirectionIndicator_Left object which is guarded */
    std::shared_ptr<IVehicleBodyLightsDirectionIndicatorLeft> m_impl;
    // Mutex for isSignaling property
    mutable std::shared_timed_mutex m_isSignalingMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
