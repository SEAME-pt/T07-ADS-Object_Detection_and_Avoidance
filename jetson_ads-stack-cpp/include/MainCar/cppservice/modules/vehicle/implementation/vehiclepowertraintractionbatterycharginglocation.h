
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharginglocation.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_Location implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingLocation : public IVehiclePowertrainTractionBatteryChargingLocation
{
public:
    explicit VehiclePowertrainTractionBatteryChargingLocation();
    ~VehiclePowertrainTractionBatteryChargingLocation();
public:
    /**
    * Latitude Latitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    void setLatitude(double Latitude) override;
    double getLatitude() const override;
    
    /**
    * Longitude Longitude of last or current charging event in WGS 84 geodetic coordinates.
    */
    void setLongitude(double Longitude) override;
    double getLongitude() const override;
    
    /**
    * Altitude Altitude relative to WGS 84 reference ellipsoid of last or current charging event.
    */
    void setAltitude(double Altitude) override;
    double getAltitude() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Location changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Location.
    */
    IVehiclePowertrainTractionBatteryChargingLocationPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_Location. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingLocationPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_Location. */
    VehiclePowertrainTractionBatteryChargingLocationData m_data;
};
} // namespace Vehicle
} // namespace Cpp
