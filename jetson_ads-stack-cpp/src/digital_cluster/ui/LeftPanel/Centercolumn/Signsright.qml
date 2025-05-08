import QtQuick 2.15

Column {
    id: signsRight
    anchors.horizontalCenter: parent.horizontalCenter
    spacing: 5

    signal gearSelected(string gearSelected)

    // Sinal para receber as luzes
    // signal toggleLight(string lightName, bool state)

    property string gear: "P"
    property bool showIcons: false // Indica se os ícones devem estar "ligados"

    property bool holdOn: systemHandler.holdLight === "true"                      // falta no systemHandler
    property bool brakeOn: systemHandler.brakeLight === "true"
    property bool tractionControlOn: systemHandler.tractionControlOn === "true"   // falta no systemHandler
    property bool tirePressureOn: systemHandler.tirePressureLight === "true"      // falta no systemHandler
    property bool autoPilot: systemHandler.autoPilot === "true"                   // falta no systemHandler

    Repeater {
        model: ["hold", "brake", "tractionControl", "tirePressure", "autopilot_1"]

        delegate: Rectangle {
            width: 40
            height: 40
            color: "transparent"
            clip: true

            Image {
                source: "../../assets/" + modelData + ".png"
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                smooth: true // Para renderização de alta qualidade
                // opacity: showIcons ? 1 : 0.2
                opacity: showIcons ? 1 : (modelData === "hold" ? (signsRight.holdOn ? 1 : 0.2) :
                                        modelData === "brake" ? (signsRight.brakeOn ? 1 : 0.2) :
                                        modelData === "tractionControl" ? (signsRight.tractionControlOn ? 1 : 0.2) :
                                        modelData === "tirePressure" ? (signsRight.tirePressureOn ? 1 : 0.2) :
                                        modelData === "autopilot_1" ? (signsRight.autoPilot ? 1: 0.2) : 0.2)
            }
        }
    }


    Timer {
            id: iconTimer
            interval: 2000 // Tempo em milissegundos (2 segundos "ligados")
            repeat: false // Executa apenas uma vez
            onTriggered: showIcons = false // Desliga os ícones
    }


    onGearSelected: function (selectedGear) {
        console.log("gear in signsright", selectedGear);
        if (selectedGear === "D" && gear === "P") {
            gear = selectedGear
            showIcons = true // Liga os ícones
            iconTimer.restart() // Reinicia o timer para desligar
        } else {
            gear = selectedGear;
        }
    }


    // onToggleLight: function(lightName, state) {
    //     if (lightName === "hold") {
    //         headLightsOn = state;
    //     } else if (lightName === "brake") {
    //         minLightOn = state;
    //     } else if (lightName === "tractionControl") {
    //         maxLightOn = state;
    //     } else if (lightName === "tirePressure") {
    //         lightAutoOn = state;
    //     }
    // }

}



