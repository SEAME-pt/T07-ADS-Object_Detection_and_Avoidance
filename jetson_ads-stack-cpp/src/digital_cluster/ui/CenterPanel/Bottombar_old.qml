import QtQuick
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects

Rectangle {
    id: bottomBar
    width: parent.width
    height: parent.height * 0.25
    color: "#2F2F2F"
    opacity: 0.7
    radius: 8
    anchors.bottom: parent.bottom
    layer.enabled: true

        SignsLeftCenterPanel {
            id: signsLeftBottomBar

            //  se o sinal não estiver implementado no systemhandler, mesmo colocando true não aparece ligado no display
            Component.onCompleted: {
                        let stateMinLights = systemHandler.minLights === "true" ? true : false; //falta no systemHandler
                        let stateHeadLights = systemHandler.headLights === "true" ? true : false;
                        let stateMaxLight = systemHandler.maxLights === "true" ? true : false; //falta no systemHandler
                        let stateAutolight = systemHandler.autoLights === "true" ? true : false; //falta no systemHandler
                        let stateHoldLight = systemHandler.holdLight === "true" ? true : false; //falta no systemHandler
                        let stateBrakeLights = systemHandler.brakeLight === "true" ? true : false;
                        let stateTractionControlLight = systemHandler.tractionControlLight === "true" ? true : false;
                        let stateTirePressureLight = systemHandler.tirePressureLight === "true" ? true : false;
                        toggleLight("minLight", stateMinLights); // Liga ou desliga o ícone - fazer o mesmo para os outros
                        toggleLight("headLights", stateHeadLights);
                        toggleLight("maxLight", stateMaxLight);
                        toggleLight("lightAuto", stateAutolight);
                        toggleLight("hold", stateHoldLight);
                        toggleLight("brake", stateBrakeLights);
                        toggleLight("tractionControl", stateTractionControlLight);
                        toggleLight("tirePressure", stateTirePressureLight);
                    }
        }


        Rectangle {
            id: hoursBottombar
            implicitWidth: hoursColumn.implicitWidth
            implicitHeight: hoursColumn.implicitHeight
            color: "transparent"
            anchors {
                horizontalCenter: parent.horizontalCenter
            }

            Column {
                id: hoursColumn
                spacing: 5

                Text {
                    text: "12:27 pm"
                    font.pixelSize: 20
                    color: "lightgray"
                }
            }
        }

        MediaBottomBar {
            id:mediaBottomBar
        }
    }
