
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorleft.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights_DirectionIndicator_Left implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsDirectionIndicatorLeft : public IVehicleBodyLightsDirectionIndicatorLeft
{
public:
    explicit VehicleBodyLightsDirectionIndicatorLeft();
    ~VehicleBodyLightsDirectionIndicatorLeft();
public:
    /**
    * IsSignaling Indicates if light is signaling or off. True = signaling. False = Off.
    */
    void setIsSignaling(bool IsSignaling) override;
    bool getIsSignaling() const override;
    
    /**
    * IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    void setIsDefect(bool IsDefect) override;
    bool getIsDefect() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_DirectionIndicator_Left changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights_DirectionIndicator_Left.
    */
    IVehicleBodyLightsDirectionIndicatorLeftPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights_DirectionIndicator_Left. */
    std::unique_ptr<IVehicleBodyLightsDirectionIndicatorLeftPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights_DirectionIndicator_Left. */
    VehicleBodyLightsDirectionIndicatorLeftData m_data;
};
} // namespace Vehicle
} // namespace Cpp
