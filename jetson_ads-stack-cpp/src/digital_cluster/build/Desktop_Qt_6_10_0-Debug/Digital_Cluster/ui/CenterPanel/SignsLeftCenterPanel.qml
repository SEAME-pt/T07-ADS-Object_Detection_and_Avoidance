import QtQuick 2.15

Rectangle {

    id: signsLeftBottomBar
    anchors {
        left: parent.left
        top: parent.top
        margins: 10
    }
    implicitHeight: signsLeftCenterPanel.implicitHeight

    Row {
        id: signsLeftCenterPanel
        // anchors.horizontalCenter: parent.horizontalCenter
        spacing: 5

        property bool minLightOn: systemHandler.minLights === "true"                    //falta no systemHandler
        property bool headLightsOn: systemHandler.headLights === "true"
        property bool maxLightOn: systemHandler.maxLights === "true"                    //falta no systemHandler
        property bool lightAutoOn: systemHandler.autoLight === "true"                   //falta no systemHandler
        property bool holdOn: systemHandler.holdLight === "true"                        //falta no systemHandler
        property bool brakeOn: systemHandler.brakeLight === "true"
        property bool tractionControlOn: systemHandler.tractionControlLight === "true"  //falta no systemHandler
        property bool tirePressureOn: systemHandler.tirePressureLight === "true"        //falta no systemHandler
        property bool autoPilot: systemHandler.autoPilot === "true"                     // falta no systemHandler

        Repeater {
            model: ["minLight", "headLights", "maxLight", "lightAuto", "hold", "brake", "tractionControl", "tirePressure", "autopilot_1"]

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
                    opacity: bottomBar.showIcons ? 1 : (modelData === "minLight" ? (signsLeftCenterPanel.minLightOn ? 1 : 0.2) :
                            modelData === "headLights" ? (signsLeftCenterPanel.headLightsOn ? 1 : 0.2) :
                            modelData === "maxLight" ? (signsLeftCenterPanel.maxLightOn ? 1 : 0.2) :
                            modelData === "lightAuto" ? (signsLeftCenterPanel.lightAutoOn ? 1 : 0.2) :
                            modelData === "hold" ? (signsLeftCenterPanel.holdOn ? 1 : 0.2) :
                            modelData === "brake" ? (signsLeftCenterPanel.brakeOn ? 1 : 0.2) :
                            modelData === "tractionControl" ? (signsLeftCenterPanel.tractionControlOn ? 1 : 0.2) :
                            modelData === "tirePressure" ? (signsLeftCenterPanel.tirePressureOn ? 1 : 0.2) :
                            modelData === "autopilot_1" ? (signsLeftCenterPanel.autoPilot ? 1 : 0.2) : 0.2)
                }
            }
        }
    }

}
