import QtQuick 2.15

Rectangle {

    id: signsLeftBottomBar
    anchors {
        left: parent.left
        top: parent.top
        margins: 10
    }
    implicitHeight: signsLeftCenterPanel.implicitHeight

    signal toggleLight(string lightName, bool state)

    // Propriedades para controlar os estados individuais das luzes
    property bool minLightOn: false
    property bool headLightsOn: false
    property bool maxLightOn: false
    property bool lightAutoOn: false
    property bool holdOn: false
    property bool brakeOn: false
    property bool tractionControlOn: false
    property bool tirePressureOn: false


    Row {
        id: signsLeftCenterPanel
        // anchors.horizontalCenter: parent.horizontalCenter
        spacing: 5

        Repeater {
            model: ["minLight", "headLights", "maxLight", "lightAuto", "hold", "brake", "tractionControl", "tirePressure"]

            delegate: Rectangle {
                width: 40
                height: 40
                color: "transparent"
                clip: true

                Image {
                    source: "../assets/" + modelData + ".png"
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    smooth: true // Para renderização de alta qualidade
                    // opacity: 1
                    opacity: modelData === "minLight" ? (minLightOn ? 1 : 0.2) :
                             modelData === "headLights" ? (headLightsOn ? 1 : 0.2) :
                             modelData === "maxLight" ? (maxLightOn ? 1 : 0.2) :
                             modelData === "lightAuto" ? (lightAutoOn ? 1 : 0.2) :
                             modelData === "hold" ? (holdOn ? 1 : 0.2) :
                             modelData === "brake" ? (brakeOn ? 1 : 0.2) :
                             modelData === "tractionControl" ? (tractionControlOn ? 1 : 0.2) :       
                             modelData === "tirePressure" ? (tirePressureOn ? 1 : 0.2) : 0.2
                }
            }
        }
    }


    onToggleLight: function(lightName, state) {

        if (lightName === "headLights") {
            headLightsOn = state;
        } else if (lightName === "minLight") {
            minLightOn = state;
        } else if (lightName === "maxLight") {
            maxLightOn = state;
        } else if (lightName === "lightAuto") {
            lightAutoOn = state;
        } else if (lightName === "hold") {
            holdOn = state;
        } else if (lightName === "brake") {
            brakeOn =  state;
        } else if (lightName === "tractionControl") {
            tractionControlOn = state;
        } else if (lightName === "tirePressure") {
            tirePressureOn = state;
        }
    }

}
