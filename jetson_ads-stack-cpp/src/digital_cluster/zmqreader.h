#ifndef ZMQREADER_H
#define ZMQREADER_H

#include <QObject>
#include <QThread>
#include <zmq.hpp>
#include <iostream>
#include <atomic>

class ZMQReader : public QThread
{
    Q_OBJECT

public:
    explicit ZMQReader(const QString &address, QObject *parent = nullptr);
    ~ZMQReader() override;

    void stop();

signals:
    void speedReceived(QString speed);
    void batteryReceived(QString battery);
    void headLightsReceived(QString headLights);
    void brakeLightReceived(QString brakeLight);
    void turnLightLeftReceived(QString turnLightLeft);
    void turnLightRightReceived(QString turnLightRight);
    void emergencyLightsReceived(QString emergencyLights);
    void totalDistanceReceived(QString totalDistance);
    void lkasReceived(QString lkas);
    void autoPilotReceived(QString autoPilot);
    void lineLeftReceived(QString lineLeft);
    void lineRightReceived(QString lineRight);

protected:
    void run() override;

private:
    QString m_address;
    std::unique_ptr<zmq::context_t> m_context;
    std::unique_ptr<zmq::socket_t> m_socket;
    // bool m_running;
    std::atomic<bool> m_running{false}; // Flag atômica
};

#endif // ZMQREADER_H

