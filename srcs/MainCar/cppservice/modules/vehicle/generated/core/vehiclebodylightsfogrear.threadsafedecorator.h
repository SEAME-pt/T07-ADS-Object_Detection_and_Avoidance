
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleBodyLightsFogRearThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Body_Lights_Fog_Rear interface.
* @see Vehicle_Body_Lights_Fog_Rear
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleBodyLightsFogRear> testVehicleBodyLightsFogRear = std::make_unique<VehicleBodyLightsFogRearThreadSafeDecorator>(std::make_shared<VehicleBodyLightsFogRear>());

// Thread safe access
auto isOn = testVehicleBodyLightsFogRear->getIsOn();
testVehicleBodyLightsFogRear->setIsOn(false);
auto isDefect = testVehicleBodyLightsFogRear->getIsDefect();
testVehicleBodyLightsFogRear->setIsDefect(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsFogRearThreadSafeDecorator : public IVehicleBodyLightsFogRear
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Body_Lights_Fog_Rear object to make thread safe.
    */
    explicit VehicleBodyLightsFogRearThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsFogRear> impl);

    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Rear implementation. */
    void setIsOn(bool IsOn) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Rear implementation. */
    bool getIsOn() const override;

    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Rear implementation. */
    void setIsDefect(bool IsDefect) override;
    /** Guards and forwards call to Vehicle_Body_Lights_Fog_Rear implementation. */
    bool getIsDefect() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Fog_Rear changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Body_Lights_Fog_Rear.
    */
    IVehicleBodyLightsFogRearPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Body_Lights_Fog_Rear object which is guarded */
    std::shared_ptr<IVehicleBodyLightsFogRear> m_impl;
    // Mutex for isOn property
    mutable std::shared_timed_mutex m_isOnMutex;
    // Mutex for isDefect property
    mutable std::shared_timed_mutex m_isDefectMutex;
};
} // namespace Vehicle
} // namespace Cpp
