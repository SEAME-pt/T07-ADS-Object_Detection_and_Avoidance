#include "System.h"

System::System(zmq::context_t &context, QObject *parent)
    : QObject{parent},
    m_speed("0"),
    m_batteryPer("0"),
    m_headLights("false"),
    m_brakeLight("false"),
    m_turnLightLeft("false"),
    m_turnLightRight("false"),
    m_emergencyLights("false"),
    m_totalDistance("0"),
    m_lkas("false"),
    m_autoPilot("false"),
    m_lineRight("false"),
    m_lineLeft("false"),
    m_gamePad(""),
    m_updatingFromZMQ(false), // Inicializa a flag
    m_zmqSocket(context, ZMQ_PUB), // Inicializa o socket PUB
    m_horn("false"),
    m_lightsLow("false"),
    m_lightSpark("false"),
    m_isMoving("false")
{
    try {
        m_zmqSocket.bind("tcp://*:5557"); // Tenta uma porta diferente
        // qDebug() << "ZMQ socket vinculado com sucesso à porta 5557";
    } catch (const zmq::error_t &e) {
        qDebug() << "Erro ao vincular ZMQ socket:" << e.what();
        throw; // Re-throw para depuração, ou trate de outra forma
    }
}


QString System::speed() const
{
    return m_speed;
}

void System::setSpeed(const QString &newSpeed)
{
    if (m_speed == newSpeed)
        return;
    m_speed = newSpeed;
    emit speedChanged();
}

QString System::batteryPer() const
{
    return m_batteryPer;
}

void System::setBatteryPer(const QString &newBatteryPer)
{
    if (m_batteryPer == newBatteryPer)
        return;
    m_batteryPer = newBatteryPer;
    emit batteryPerChanged();
}

QString System::headLights() const
{
    return m_headLights;
}

void System::setHeadLights(const QString &newHeadLights)
{
    if (m_headLights == newHeadLights)
        return;
    m_headLights = newHeadLights;
    emit headLightsChanged();
}

QString System::brakeLight() const
{
    return m_brakeLight;
}

void System::setBrakeLight(const QString &newBrakeLight)
{
    if (m_brakeLight == newBrakeLight)
        return;
    m_brakeLight = newBrakeLight;
    emit brakeLightChanged();
}

QString System::turnLightLeft() const
{
    return m_turnLightLeft;
}

void System::setTurnLightLeft(const QString &newTurnLightLeft)
{
    if (m_turnLightLeft == newTurnLightLeft)
        return;
    m_turnLightLeft = newTurnLightLeft;
    emit turnLightLeftChanged();
}

QString System::turnLightRight() const
{
    return m_turnLightRight;
}

void System::setTurnLightRight(const QString &newTurnLightRight)
{
    if (m_turnLightRight == newTurnLightRight)
        return;
    m_turnLightRight = newTurnLightRight;
    emit turnLightRightChanged();
}

QString System::emergencyLights() const
{
    return m_emergencyLights;
}

void System::setEmergencyLights(const QString &newEmergencyLights)
{
    if (m_emergencyLights == newEmergencyLights)
        return;
    m_emergencyLights = newEmergencyLights;
    emit emergencyLightsChanged();
}

QString System::totalDistance() const
{
    return m_totalDistance;
}

void System::setTotalDistance(const QString &newTotalDistance)
{
    if (m_totalDistance == newTotalDistance)
        return;
    m_totalDistance = newTotalDistance;
    emit totalDistanceChanged();
}

QString System::lkas() const
{
    return m_lkas;
}

// void System::setLkas(const QString &newLkas)
// {
//     if (m_lkas == newLkas)
//         return;
//     m_lkas = newLkas;
//     qDebug() << "System: lkas alterado para: " << m_lkas; //- para teste
//     emit lkasChanged();
// }

void System::setLkas(const QString &newLkas)
{
    if (m_lkas == newLkas)
        return;
    m_lkas = newLkas;
    qDebug() << "System: lkas alterado para: " << m_lkas;
    if (!m_updatingFromZMQ) { // Só envia ao ZMQ se não veio do ZMQ
        sendToZMQ("lkas", m_lkas);
    }
    emit lkasChanged();
}

void System::setLkasFromZMQ(const QString &newLkas)
{
    m_updatingFromZMQ = true; // Marca como vindo do ZMQ
    setLkas(newLkas);
    m_updatingFromZMQ = false; // Reseta a flag
}

QString System::autoPilot() const
{
    return m_autoPilot;
}

// void System::setAutoPilot(const QString &newAutoPilot)
// {
//     if (m_autoPilot == newAutoPilot)
//         return;
//     m_autoPilot = newAutoPilot;
//     qDebug() << "System: autoPilot alterado para: " << m_autoPilot; // - para teste
//     emit autoPilotChanged();
// }

void System::setAutoPilot(const QString &newAutoPilot)
{
    if (m_autoPilot == newAutoPilot)
        return;
    m_autoPilot = newAutoPilot;
    qDebug() << "System: autoPilot alterado para: " << m_autoPilot;
    if (!m_updatingFromZMQ) { // Só envia ao ZMQ se não veio do ZMQ
        sendToZMQ("autoPilot", m_autoPilot);
    }
    emit autoPilotChanged();
}

void System::setAutoPilotFromZMQ(const QString &newAutoPilot)
{
    m_updatingFromZMQ = true; // Marca como vindo do ZMQ
    setAutoPilot(newAutoPilot);
    m_updatingFromZMQ = false; // Reseta a flag
}

QString System::lineRight() const
{
    return m_lineRight;
}

void System::setLineRight(const QString &newLineRight)
{
    if (m_lineRight == newLineRight)
        return;
    m_lineRight = newLineRight;
    emit lineRightChanged();
}

QString System::lineLeft() const
{
    return m_lineLeft;
}

void System::setLineLeft(const QString &newLineLeft)
{
    if (m_lineLeft == newLineLeft)
        return;
    m_lineLeft = newLineLeft;
    emit lineLeftChanged();
}

void System::sendToZMQ(const QString &key, const QString &value)
{
    QString message = key + ":" + value; // Ex.: "lkas:true"
    zmq::message_t zmqMessage(message.toUtf8().constData(), message.size());
    m_zmqSocket.send(zmqMessage, zmq::send_flags::dontwait); // Envia sem bloquear
    qDebug() << "Enviado ao ZMQ:" << message;
}

QString System::gamePad() const
{
    return m_gamePad;
}

void System::setGamePad(const QString &newGamePad)
{

    if (m_gamePad == newGamePad) {
        sendToZMQ("button", m_gamePad);
        return;
    }
    m_gamePad = newGamePad;
    qDebug() << "button gamepad: " << m_gamePad;
    sendToZMQ("button", m_gamePad);
    emit gamePadChanged();
}

QString System::horn() const
{
    return m_horn;
}

void System::setHorn(const QString &newHorn)
{
    if (m_horn == newHorn)
        return;
    m_horn = newHorn;
    emit hornChanged();
}

QString System::lightsLow() const
{
    return m_lightsLow;
}

void System::setLightsLow(const QString &newLightsLow)
{
    if (m_lightsLow == newLightsLow)
        return;
    m_lightsLow = newLightsLow;
    emit lightsLowChanged();
}

QString System::lightSpark() const
{
    return m_lightSpark;
}

void System::setLightSpark(const QString &newLightSpark)
{
    if (m_lightSpark == newLightSpark)
        return;
    m_lightSpark = newLightSpark;
    emit lightSparkChanged();
}

QString System::isMoving() const
{
    return m_isMoving;
}

void System::setIsMoving(const QString &newIsMoving)
{
    if (m_isMoving == newIsMoving)
        return;
    m_isMoving = newIsMoving;
    emit isMovingChanged();
}
