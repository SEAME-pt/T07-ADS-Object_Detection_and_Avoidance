#pragma once

#include <nlohmann/json.hpp>
#include <zmq.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "vehicle.h"


using json = nlohmann::json;
using namespace Cpp::Vehicle;

class vehicleInformation : public IVehicleVehicleIdentificationSubscriber {
private:
    zmq::socket_t& publisher; // Referência para o socket do ZeroMQ
    std::unordered_map<std::string, std::string> currentValues; // Armazena os valores atuais dos atributos
    std::string jsonFilePath = "vehicle_info.json"; // Caminho para o arquivo JSON

    // Helper to send message via ZeroMQ
    void sendMessage(const std::string& key, const std::string& value) {
        std::ostringstream message;
        message << key << " " << value;

        zmq::message_t zmqMessage(message.str().size());
        memcpy(zmqMessage.data(), message.str().c_str(), message.str().size());
        publisher.send(zmqMessage, zmq::send_flags::none);
    }

    // update json
    void updateJsonFile() {
        json j;
        for (const auto& [key, value] : currentValues) {
            j[key] = value;
        }
        std::ofstream file(jsonFilePath);
        if (file.is_open()) {
            file << j.dump(4);
            file.close();
        } else {
            std::cerr << "Erro to open json file" << std::endl;
        }
    }

public:
    explicit vehicleInformation(zmq::socket_t& pub) : publisher(pub) {}

    void publishVehicleInfo(const IVehicleVehicleIdentification* vehicleInfo) {
        currentValues["vin"] = vehicleInfo->getVin();
        currentValues["wmi"] = vehicleInfo->getWmi();
        currentValues["brand"] = vehicleInfo->getBrand();
        currentValues["model"] = vehicleInfo->getModel();
        currentValues["year"] = std::to_string(vehicleInfo->getYear());
        currentValues["body_type"] = vehicleInfo->getBodyType();
        currentValues["license_plate"] = vehicleInfo->getLicensePlate();
        currentValues["seating_capacity"] = std::to_string(vehicleInfo->getVehicleSeatingCapacity());
        currentValues["special_usage"] = vehicleInfo->getVehicleSpecialUsage();
        currentValues["exterior_color"] = vehicleInfo->getVehicleExteriorColor();

        const auto& extras = vehicleInfo->getOptionalExtras();
        std::ostringstream extrasStream;
        for (const auto& extra : extras) {
            extrasStream << extra << ",";
        }
        currentValues["optional_extras"] = extrasStream.str();

        updateJsonFile();
    }

    // on changed methods
    void onVinChanged(const std::string& VIN) override {
        currentValues["vin"] = VIN;
        updateJsonFile();
        sendMessage("vin", VIN);
    }

    void onWmiChanged(const std::string& WMI) override {
        currentValues["wmi"] = WMI;
        updateJsonFile();
        sendMessage("wmi", WMI);
    }

    void onBrandChanged(const std::string& Brand) override {
        currentValues["brand"] = Brand;
        updateJsonFile();
        sendMessage("brand", Brand);
    }

    void onModelChanged(const std::string& Model) override {
        currentValues["model"] = Model;
        updateJsonFile();
        sendMessage("model", Model);
    }

    void onYearChanged(int32_t Year) override {
        currentValues["year"] = std::to_string(Year);
        updateJsonFile();
        sendMessage("year", std::to_string(Year));
    }

    void onBodyTypeChanged(const std::string& BodyType) override {
        currentValues["body_type"] = BodyType;
        updateJsonFile();
        sendMessage("body_type", BodyType);
    }

    void onLicensePlateChanged(const std::string& LicensePlate) override {
        currentValues["license_plate"] = LicensePlate;
        updateJsonFile();
        sendMessage("license_plate", LicensePlate);
    }

    void onVehicleSeatingCapacityChanged(int32_t VehicleSeatingCapacity) override {
        currentValues["seating_capacity"] = std::to_string(VehicleSeatingCapacity);
        updateJsonFile();
        sendMessage("seating_capacity", std::to_string(VehicleSeatingCapacity));
    }

    void onVehicleSpecialUsageChanged(const std::string& VehicleSpecialUsage) override {
        currentValues["special_usage"] = VehicleSpecialUsage;
        updateJsonFile();
        sendMessage("special_usage", VehicleSpecialUsage);
    }

    void onVehicleExteriorColorChanged(const std::string& VehicleExteriorColor) override {
        currentValues["exterior_color"] = VehicleExteriorColor;
        updateJsonFile();
        sendMessage("exterior_color", VehicleExteriorColor);
    }

    void onOptionalExtrasChanged(const std::list<std::string>& OptionalExtras) override {
        std::ostringstream extrasStream;
        for (const auto& extra : OptionalExtras) {
            extrasStream << extra << ",";
        }
        currentValues["optional_extras"] = extrasStream.str();
        updateJsonFile();
        sendMessage("optional_extras", extrasStream.str());
    }
    void onAcrissCodeChanged(const std::string& AcrissCode) {}
    void onDateVehicleFirstRegisteredChanged(const std::string& DateVehicleFirstRegistered) {}
    void onMeetsEmissionStandardChanged(const std::string& MeetsEmissionStandard) {}
    void onProductionDateChanged(const std::string& ProductionDate) {}
    void onPurchaseDateChanged(const std::string& PurchaseDate) {}
    void onVehicleModelDateChanged(const std::string& VehicleModelDate) {}
    void onVehicleConfigurationChanged(const std::string& VehicleConfiguration) {}
    void onVehicleInteriorColorChanged(const std::string& VehicleInteriorColor) {}
    void onVehicleInteriorTypeChanged(const std::string& VehicleInteriorType) {}
    void onKnownVehicleDamagesChanged(const std::string& KnownVehicleDamages) {}
};
