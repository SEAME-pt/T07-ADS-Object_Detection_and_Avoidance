
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclelowvoltagebattery.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_LowVoltageBattery implementation.
*/
class CPP_VEHICLE_EXPORT VehicleLowVoltageBattery : public IVehicleLowVoltageBattery
{
public:
    explicit VehicleLowVoltageBattery();
    ~VehicleLowVoltageBattery();
public:
    /**
    * NominalVoltage Nominal Voltage of the battery.
    */
    void setNominalVoltage(int32_t NominalVoltage) override;
    int32_t getNominalVoltage() const override;
    
    /**
    * NominalCapacity Nominal capacity of the low voltage battery.
    */
    void setNominalCapacity(int32_t NominalCapacity) override;
    int32_t getNominalCapacity() const override;
    
    /**
    * CurrentVoltage Current Voltage of the low voltage battery.
    */
    void setCurrentVoltage(float CurrentVoltage) override;
    float getCurrentVoltage() const override;
    
    /**
    * CurrentCurrent Current current flowing in/out of the low voltage battery. Positive = Current flowing in to battery, e.g. during charging or driving. Negative = Current flowing out of battery, e.g. when using the battery to start a combustion engine.
    */
    void setCurrentCurrent(float CurrentCurrent) override;
    float getCurrentCurrent() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_LowVoltageBattery changes and signal emission.
    * @return The publisher for Vehicle_LowVoltageBattery.
    */
    IVehicleLowVoltageBatteryPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_LowVoltageBattery. */
    std::unique_ptr<IVehicleLowVoltageBatteryPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_LowVoltageBattery. */
    VehicleLowVoltageBatteryData m_data;
};
} // namespace Vehicle
} // namespace Cpp
