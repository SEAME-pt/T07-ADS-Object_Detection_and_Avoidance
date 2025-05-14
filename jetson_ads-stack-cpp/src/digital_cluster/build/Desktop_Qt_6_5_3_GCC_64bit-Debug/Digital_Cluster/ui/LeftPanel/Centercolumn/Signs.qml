import QtQuick 2.15

Column {
    id: signs
    anchors.horizontalCenter: parent.horizontalCenter
    spacing: 5
    // sinal para receber a gear
    signal gearSelected(string gearSelected)

    // Sinal para receber as luzes
    // signal toggleLight(string lightName, bool state)

    property string gear: "P"
    property bool showIcons: false // Indica se os ícones devem estar "ligados" inicialmente -teste de luzes

    // Propriedades para controlar os estados individuais das luzes
    property bool minlightOn: systemHandler.minLights === "true"
    property bool headLightsOn: systemHandler.headlights === "true"
    property bool maxLightOn: systemHandler.maxLights === "true"
    property bool lightAutoOn: systemHandler.autoLight === "true"


    Repeater {
        model: ["minLight", "headLights", "maxLight", "lightAuto"]

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
               // opacity: showIcons ? 1 : 0.3
                opacity: showIcons ? 1 : (modelData === "minLight" ? (signs.minlightOn ? 1 : 0.2) :
                                          modelData === "headLights" ? (signs.headLightsOn ? 1 : 0.2) :
                                          modelData === "maxLight" ? (signs.maxLightOn ? 1 : 0.2) :
                                          modelData === "lightAuto" ? (signs.lightAutoOn ? 1 : 0.2) : 0.2)
            }
        }
    }


    Timer {
            id: iconTimer
            interval: 2000 // Tempo em milissegundos (2 segundos "ligados")
            repeat: false // Executa apenas uma vez
            onTriggered: showIcons = false // Desliga os ícones
        }


    // // Funções para controlar os estados individuais das luzes - não utilizadas exemplo
    // function setMinlight(on) {
    //     minlightOn = on;
    // }


    onGearSelected: function (selectedGear) {
        console.log("gear in signs", selectedGear);
        if (selectedGear === "D" && gear === "P") {
            gear = selectedGear
            showIcons = true // Liga os ícones
            iconTimer.restart() // Reinicia o timer para desligar
        } else {
            gear = selectedGear;
        }
    }


    // onToggleLight: function(lightName, state) {
    //     if (lightName === "headLights") {
    //         headLightsOn = state;
    //     } else if (lightName === "minLight") {
    //         minLightOn = state;
    //     } else if (lightName === "maxLight") {
    //         maxLightOn = state;
    //     } else if (lightName === "lightAuto") {
    //         lightAutoOn = state;
    //     }
    // }

}



