
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylightsparking.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights_Parking implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsParking : public IVehicleBodyLightsParking
{
public:
    explicit VehicleBodyLightsParking();
    ~VehicleBodyLightsParking();
public:
    /**
    * IsOn Indicates if light is on or off. True = On. False = Off.
    */
    void setIsOn(bool IsOn) override;
    bool getIsOn() const override;
    
    /**
    * IsDefect Indicates if light is defect. True = Light is defect. False = Light has no defect.
    */
    void setIsDefect(bool IsDefect) override;
    bool getIsDefect() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Parking changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights_Parking.
    */
    IVehicleBodyLightsParkingPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights_Parking. */
    std::unique_ptr<IVehicleBodyLightsParkingPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights_Parking. */
    VehicleBodyLightsParkingData m_data;
};
} // namespace Vehicle
} // namespace Cpp
