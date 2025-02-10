#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

// ********************************************************************
// Enumeration Vehicle_Powertrain_Transmission_Type_Value
// ********************************************************************
Vehicle_Powertrain_Transmission_Type_ValueEnum toVehiclePowertrainTransmissionTypeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN;
        case 1: return Vehicle_Powertrain_Transmission_Type_ValueEnum::SEQUENTIAL;
        case 2: return Vehicle_Powertrain_Transmission_Type_ValueEnum::H;
        case 3: return Vehicle_Powertrain_Transmission_Type_ValueEnum::AUTOMATIC;
        case 4: return Vehicle_Powertrain_Transmission_Type_ValueEnum::DSG;
        case 5: return Vehicle_Powertrain_Transmission_Type_ValueEnum::CVT;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_Transmission_DriveType_Value
// ********************************************************************
Vehicle_Powertrain_Transmission_DriveType_ValueEnum toVehiclePowertrainTransmissionDriveTypeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN;
        case 1: return Vehicle_Powertrain_Transmission_DriveType_ValueEnum::FORWARD_WHEEL_DRIVE;
        case 2: return Vehicle_Powertrain_Transmission_DriveType_ValueEnum::REAR_WHEEL_DRIVE;
        case 3: return Vehicle_Powertrain_Transmission_DriveType_ValueEnum::ALL_WHEEL_DRIVE;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_Transmission_PerformanceMode_Value
// ********************************************************************
Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum toVehiclePowertrainTransmissionPerformanceModeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL;
        case 1: return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::SPORT;
        case 2: return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::ECONOMY;
        case 3: return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::SNOW;
        case 4: return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::RAIN;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_Transmission_GearChangeMode_Value
// ********************************************************************
Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum toVehiclePowertrainTransmissionGearChangeModeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL;
        case 1: return Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::AUTOMATIC;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_Value
// ********************************************************************
Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum toVehiclePowertrainElectricMotorEngineCoolantLevelValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW;
        case 1: return Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::LOW;
        case 2: return Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::NORMAL;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortFrontLeftSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortFrontRightSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortRearLeftSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortRearMiddleSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortRearRightSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum toVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionSupportedInletTypesValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_2_AC;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_3_AC;
        case 3: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_4_DC;
        case 4: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_CCS_DC;
        case 5: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_2_CCS_DC;
        case 6: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::TESLA_ROADSTER;
        case 7: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::TESLA_HPWC;
        case 8: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::TESLA_SUPERCHARGER;
        case 9: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::GBT_AC;
        case 10: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::GBT_DC;
        case 11: return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::OTHER;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum toVehiclePowertrainTractionBatteryChargingStartStopChargingValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::START;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::STOP;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::START;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum toVehiclePowertrainTractionBatteryChargingTimerModeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE;
        case 1: return Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::START_TIME;
        case 2: return Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::END_TIME;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_Value
// ********************************************************************
Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum toVehiclePowertrainTractionBatteryBatteryConditioningRequestedModeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE;
        case 1: return Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::FAST_CHARGING_PREPARATION;
        case 2: return Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::DRIVING_PREPARATION;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE;
    }
}

// ********************************************************************
// Enumeration Vehicle_Powertrain_Type_Value
// ********************************************************************
Vehicle_Powertrain_Type_ValueEnum toVehiclePowertrainTypeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Powertrain_Type_ValueEnum::COMBUSTION;
        case 1: return Vehicle_Powertrain_Type_ValueEnum::HYBRID;
        case 2: return Vehicle_Powertrain_Type_ValueEnum::ELECTRIC;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Powertrain_Type_ValueEnum::COMBUSTION;
    }
}

// ********************************************************************
// Enumeration Vehicle_Body_Lights_LightSwitch_Value
// ********************************************************************
Vehicle_Body_Lights_LightSwitch_ValueEnum toVehicleBodyLightsLightSwitchValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF;
        case 1: return Vehicle_Body_Lights_LightSwitch_ValueEnum::POSITION;
        case 2: return Vehicle_Body_Lights_LightSwitch_ValueEnum::DAYTIME_RUNNING_LIGHTS;
        case 3: return Vehicle_Body_Lights_LightSwitch_ValueEnum::AUTO;
        case 4: return Vehicle_Body_Lights_LightSwitch_ValueEnum::BEAM;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF;
    }
}

// ********************************************************************
// Enumeration Vehicle_Body_Lights_Brake_IsActive_Value
// ********************************************************************
Vehicle_Body_Lights_Brake_IsActive_ValueEnum toVehicleBodyLightsBrakeIsActiveValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE;
        case 1: return Vehicle_Body_Lights_Brake_IsActive_ValueEnum::ACTIVE;
        case 2: return Vehicle_Body_Lights_Brake_IsActive_ValueEnum::ADAPTIVE;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE;
    }
}

// ********************************************************************
// Enumeration Vehicle_LowVoltageSystemState_Value
// ********************************************************************
Vehicle_LowVoltageSystemState_ValueEnum toVehicleLowVoltageSystemStateValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED;
        case 1: return Vehicle_LowVoltageSystemState_ValueEnum::LOCK;
        case 2: return Vehicle_LowVoltageSystemState_ValueEnum::OFF;
        case 3: return Vehicle_LowVoltageSystemState_ValueEnum::ACC;
        case 4: return Vehicle_LowVoltageSystemState_ValueEnum::ON;
        case 5: return Vehicle_LowVoltageSystemState_ValueEnum::START;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED;
    }
}

// ********************************************************************
// Enumeration Vehicle_CurrentLocation_GNSSReceiver_FixType_Value
// ********************************************************************
Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum toVehicleCurrentLocationGnssReceiverFixTypeValueEnum(std::uint8_t v, bool *ok)
{
    if (ok != nullptr) {
        *ok = true;
    }
    switch (v) {
        case 0: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE;
        case 1: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::TWO_D;
        case 2: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::TWO_D_SATELLITE_BASED_AUGMENTATION;
        case 3: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::TWO_D_GROUND_BASED_AUGMENTATION;
        case 4: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::TWO_D_SATELLITE_AND_GROUND_BASED_AUGMENTATION;
        case 5: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::THREE_D;
        case 6: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::THREE_D_SATELLITE_BASED_AUGMENTATION;
        case 7: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::THREE_D_GROUND_BASED_AUGMENTATION;
        case 8: return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::THREE_D_SATELLITE_AND_GROUND_BASED_AUGMENTATION;
        default:
            if (ok != nullptr) {
                *ok = false;
            }
            return Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE;
    }
}
} // namespace Vehicle
} // namespace Cpp
