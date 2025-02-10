
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsFogFrontThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Fog_Front interface.
* @see Vehicle_Body_Lights_Fog_Front
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsFogFront> testVehicleBodyLightsFogFront = std::make_unique<VehicleBodyLightsFogFrontThreadSafeDecorator>(std::make_shared<VehicleBodyLightsFogFront>());

// Thread safe access
auto isOn = testVehicleBodyLightsFogFront->getIsOn();
testVehicleBodyLightsFogFront->setIsOn(false);
auto isDefect = testVehicleBodyLightsFogFront->getIsDefect();
testVehicleBodyLightsFogFront->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsFogFrontThreadSafeDecorator : public IVehicleBodyLightsFogFront
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Fog_Front object to make thread safe.
    */
    explicit VehicleBodyLightsFogFrontThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsFogFront> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Front implementation. */
    void setIsOn(bool IsOn) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Front implementation. */
    bool getIsOn() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Front implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Front implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Fog_Front changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Fog_Front.
    */
    IVehicleBodyLightsFogFrontPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Fog_Front object which is guarded */
    std::shared_ptr<IVehicleBodyLightsFogFront> m_impl;
    // Mutex for isOn property
    mutable std::shared_timed_mutex m_isOnMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
