
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebody.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBody : public IVehicleBody
{
public:
    explicit VehicleBody();
    ~VehicleBody();
public:
    /**
    * PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    int32_t getPowerOptimizeLevel() const override;
    
    /**
    * IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    bool getIsAutoPowerOptimize() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Body changes and signal emission.
    * @return The publisher for Vehicle_Body.
    */
    IVehicleBodyPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body. */
    std::unique_ptr<IVehicleBodyPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body. */
    VehicleBodyData m_data;
};
} // namespace Vehicle
} // namespace Cpp
