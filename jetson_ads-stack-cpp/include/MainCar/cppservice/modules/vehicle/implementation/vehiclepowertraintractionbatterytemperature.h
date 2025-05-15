
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterytemperature.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Temperature implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryTemperature : public IVehiclePowertrainTractionBatteryTemperature
{
public:
    explicit VehiclePowertrainTractionBatteryTemperature();
    ~VehiclePowertrainTractionBatteryTemperature();
public:
    /**
    * Average Current average temperature of the battery cells.
    */
    void setAverage(float Average) override;
    float getAverage() const override;
    
    /**
    * Min Current minimum temperature of the battery cells, i.e. temperature of the coldest cell.
    */
    void setMin(float Min) override;
    float getMin() const override;
    
    /**
    * Max Current maximum temperature of the battery cells, i.e. temperature of the hottest cell.
    */
    void setMax(float Max) override;
    float getMax() const override;
    
    /**
    * CellTemperature Array of cell temperatures. Length or array shall correspond to number of cells in vehicle.
    */
    void setCellTemperature(const std::list<float>& CellTemperature) override;
    const std::list<float>& getCellTemperature() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Temperature changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Temperature.
    */
    IVehiclePowertrainTractionBatteryTemperaturePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Temperature. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryTemperaturePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Temperature. */
    VehiclePowertrainTractionBatteryTemperatureData m_data;
};
} // namespace Vehicle
} // namespace Cpp
