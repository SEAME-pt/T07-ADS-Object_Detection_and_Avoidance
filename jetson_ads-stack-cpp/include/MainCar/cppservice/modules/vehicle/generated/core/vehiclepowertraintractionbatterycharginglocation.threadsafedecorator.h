
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_Location interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_Location
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingLocation> testVehiclePowertrainTractionBatteryChargingLocation = std::make_unique<VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingLocation>());

// Thread safe access
auto latitude = testVehiclePowertrainTractionBatteryChargingLocation->getLatitude();
testVehiclePowertrainTractionBatteryChargingLocation->setLatitude(0.0);
auto longitude = testVehiclePowertrainTractionBatteryChargingLocation->getLongitude();
testVehiclePowertrainTractionBatteryChargingLocation->setLongitude(0.0);
auto altitude = testVehiclePowertrainTractionBatteryChargingLocation->getAltitude();
testVehiclePowertrainTractionBatteryChargingLocation->setAltitude(0.0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingLocation
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_Location object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingLocation> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    void setLatitude(double Latitude) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    double getLatitude() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    void setLongitude(double Longitude) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    double getLongitude() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    void setAltitude(double Altitude) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_Location implementation. */
    double getAltitude() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Location changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Location.
    */
    IVehiclePowertrainTractionBatteryChargingLocationPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_Location object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingLocation> m_impl;
    // Mutex for latitude property
    mutable std::shared_timed_mutex m_latitudeMutex;
    // Mutex for longitude property
    mutable std::shared_timed_mutex m_longitudeMutex;
    // Mutex for altitude property
    mutable std::shared_timed_mutex m_altitudeMutex;
};
} // namespace Vehicle
} // namespace Cpp
