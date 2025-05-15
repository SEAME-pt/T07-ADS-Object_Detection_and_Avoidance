#pragma once
#include <iostream>
#include <string>

class ACar {
protected:
    std::string _make;
    std::string _model;
    unsigned int _year;

public:
    ACar();
    ACar(const std::string& make, const std::string& model, const unsigned int year);

    ACar(const ACar& original);
    ACar& operator=(const ACar& original);

    virtual void drive() const = 0;
    virtual std::string getInfo() const;

    virtual ~ACar();
};

