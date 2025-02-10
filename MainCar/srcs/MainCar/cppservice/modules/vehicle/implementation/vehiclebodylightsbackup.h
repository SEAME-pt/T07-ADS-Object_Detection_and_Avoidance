
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclebodylightsbackup.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Body_Lights_Backup implementation.
*/
class CPP_VEHICLE_EXPORT VehicleBodyLightsBackup : public IVehicleBodyLightsBackup
{
public:
    explicit VehicleBodyLightsBackup();
    ~VehicleBodyLightsBackup();
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
    * Access to a publisher, use it to subscribe for Vehicle_Body_Lights_Backup changes and signal emission.
    * @return The publisher for Vehicle_Body_Lights_Backup.
    */
    IVehicleBodyLightsBackupPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Body_Lights_Backup. */
    std::unique_ptr<IVehicleBodyLightsBackupPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Body_Lights_Backup. */
    VehicleBodyLightsBackupData m_data;
};
} // namespace Vehicle
} // namespace Cpp
