#ifndef SYSTEM_H
#define SYSTEM_H

#include <QObject>
#include <QDebug>
#include <QString>
#include <zmq.hpp> // Adicionado para usar ZMQ

class System : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString speed READ speed WRITE setSpeed NOTIFY speedChanged FINAL)
    Q_PROPERTY(QString batteryPer READ batteryPer WRITE setBatteryPer NOTIFY batteryPerChanged FINAL)
    Q_PROPERTY(QString headLights READ headLights WRITE setHeadLights NOTIFY headLightsChanged FINAL)
    Q_PROPERTY(QString brakeLight READ brakeLight WRITE setBrakeLight NOTIFY brakeLightChanged FINAL)
    Q_PROPERTY(QString turnLightLeft READ turnLightLeft WRITE setTurnLightLeft NOTIFY turnLightLeftChanged FINAL)
    Q_PROPERTY(QString turnLightRight READ turnLightRight WRITE setTurnLightRight NOTIFY turnLightRightChanged FINAL)
    Q_PROPERTY(QString emergencyLights READ emergencyLights WRITE setEmergencyLights NOTIFY emergencyLightsChanged FINAL)
    Q_PROPERTY(QString totalDistance READ totalDistance WRITE setTotalDistance NOTIFY totalDistanceChanged FINAL)
    Q_PROPERTY(QString lkas READ lkas WRITE setLkas NOTIFY lkasChanged FINAL)
    Q_PROPERTY(QString autoPilot READ autoPilot WRITE setAutoPilot NOTIFY autoPilotChanged FINAL)
    Q_PROPERTY(QString lineRight READ lineRight WRITE setLineRight NOTIFY lineRightChanged FINAL)
    Q_PROPERTY(QString lineLeft READ lineLeft WRITE setLineLeft NOTIFY lineLeftChanged FINAL)
    Q_PROPERTY(QString gamePad READ gamePad WRITE setGamePad NOTIFY gamePadChanged FINAL)

	Q_PROPERTY(QString horn READ horn WRITE setHorn NOTIFY hornChanged FINAL)
	Q_PROPERTY(QString lightsLow READ lightsLow WRITE setLightsLow NOTIFY lightsLowChanged FINAL)
	Q_PROPERTY(QString lightSpark READ lightSpark WRITE setLightSpark NOTIFY lightSparkChanged FINAL)
	Q_PROPERTY(QString isMoving READ isMoving WRITE setIsMoving NOTIFY isMovingChanged FINAL)
	// batteryPer já existe, mas pode ser mapeado para batteryPercentage

public:
    // explicit System(QObject *parent = nullptr);
    explicit System(zmq::context_t &context, QObject *parent = nullptr);

    QString speed() const;
    void setSpeed(const QString &newSpeed);

    QString batteryPer() const;
    void setBatteryPer(const QString &newBatteryPer);

    QString headLights() const;
    void setHeadLights(const QString &newHeadLights);

    QString brakeLight() const;
    void setBrakeLight(const QString &newBrakeLight);

    QString turnLightLeft() const;
    void setTurnLightLeft(const QString &newTurnLightLeft);

    QString turnLightRight() const;
    void setTurnLightRight(const QString &newTurnLightRight);

    QString emergencyLights() const;
    void setEmergencyLights(const QString &newEmergencyLights);

    QString totalDistance() const;
    void setTotalDistance(const QString &newTotalDistance);

    QString lkas() const;
    void setLkas(const QString &newLkas);
    void setLkasFromZMQ(const QString &newLkas); // Novo: método para ZMQ

    QString autoPilot() const;
    void setAutoPilot(const QString &newAutoPilot);
    void setAutoPilotFromZMQ(const QString &newAutoPilot); // Novo: método para ZMQ

    QString lineRight() const;
    void setLineRight(const QString &newLineRight);

    QString lineLeft() const;
    void setLineLeft(const QString &newLineLeft);

    QString gamePad() const;
    void setGamePad(const QString &newGamePad);

	// Novo: métodos para horn, lightsLow, lightSpark e isMoving
	QString horn() const;
    void setHorn(const QString &newHorn);
    QString lightsLow() const;
    void setLightsLow(const QString &newLightsLow);
    QString lightSpark() const;
    void setLightSpark(const QString &newLightSpark);
    QString isMoving() const;
    void setIsMoving(const QString &newIsMoving);

signals:

    void speedChanged();

    void batteryPerChanged();

    void headLightsChanged();

    void brakeLightChanged();

    void turnLightLeftChanged();

    void turnLightRightChanged();

    void emergencyLightsChanged();

    void totalDistanceChanged();

    void lkasChanged();

    void autoPilotChanged();

    void lineRightChanged();

    void lineLeftChanged();

    void gamePadChanged();

	// Novo: sinais para horn, lightsLow, lightSpark e isMoving
	void hornChanged();
    void lightsLowChanged();
    void lightSparkChanged();
    void isMovingChanged();

private:

    QString m_speed;
    QString m_batteryPer;
    QString m_headLights;
    QString m_brakeLight;
    QString m_turnLightLeft;
    QString m_turnLightRight;
    QString m_emergencyLights;
    QString m_totalDistance;
    QString m_lkas;
    QString m_autoPilot;
    QString m_lineRight;
    QString m_lineLeft;
    QString m_gamePad;

	// Novo: atributos para horn, lightsLow, lightSpark e isMoving
	QString m_horn;
    QString m_lightsLow;
    QString m_lightSpark;
    QString m_isMoving;

    bool m_updatingFromZMQ; // Novo: flag para evitar loop
    zmq::socket_t m_zmqSocket; // Novo: socket para enviar ao ZMQ

    void sendToZMQ(const QString &key, const QString &value); // Novo: função para enviar ao ZMQ

};

#endif // SYSTEM_H
