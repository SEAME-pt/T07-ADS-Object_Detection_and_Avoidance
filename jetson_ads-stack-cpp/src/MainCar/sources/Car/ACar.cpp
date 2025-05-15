#include "ACar.hpp"
#include <iostream>

ACar::ACar() : _make("Unknown"), _model("Unknown"), _year(0) {}

ACar::ACar(const std::string& make, const std::string& model, const unsigned int year)
    : _make(make), _model(model), _year(year) {}

ACar::ACar(const ACar& other) : _make(other._make), _model(other._model), _year(other._year) {}

ACar& ACar::operator=(const ACar& other) {
    if (this != &other) {
        _make = other._make;
        _model = other._model;
        _year = other._year;
    }
    return *this;
}

ACar::~ACar() {}

std::string ACar::getInfo() const {
    return _make + " " + _model + " " + std::to_string(_year);
}
