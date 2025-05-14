import QtQuick 2.15
import QtQuick.Controls 2.15


Rectangle {
    id: centerPanel

    color: "#2F2F2F"
    radius: 8

    signal stopNav(bool stop)
    // signal gearSelected(string gear)

    property bool isCenterPanel: true // Identificador único

    property string gearCenterPanel: "P"


    Timer {
            id: iconTimer1
            interval: 2000 // Tempo em milissegundos (2 segundos "ligados")
            repeat: false // Executa apenas uma vez
            onTriggered: bottomBar.showIcons = false // Desliga os ícones
        }


    function updateGear(gear) {
        console.log("gear recebida no CenterPanel:", gear);
        if (gear === "D" && gearCenterPanel === "P") {
            gearCenterPanel = gear
            bottomBar.showIcons = true // Liga os ícones
            iconTimer1.restart() // Reinicia o timer para desligar
        } else {
            gearCenterPanel = gear;
        }
    }

    property bool turnLightLeftCenterPanelOn: systemHandler.turnLightLeft === "true" ? true : false
    property bool turnLightRightCenterPanelOn: systemHandler.turnLightRight === "true" ? true : false
    property bool emergencyCenterPanel: systemHandler.emergencyLights === "true" ? true : false // para teste colocar aqui true



    width: parent ? parent.width : 0
    height: parent ? parent.height : 0


    Loader {
        id: rightLoader
        anchors {
            fill: parent
        }
        source: "Navigation.qml"

    }


    Bottombar {
        id: bottomBar
    }


    Rectangle {
        id: speedCenterPanel
        // width: parent.width * 0.1
        // height: parent.height * 0.2
        implicitWidth: speedColumn.implicitWidth
        implicitHeight: speedColumn.implicitHeight
        anchors {
            top: parent.top
            right: parent.right
            rightMargin: 30
            topMargin: 20
        }
        color: "transparent"

        Column {
            id: speedColumn
            spacing: 3
            // anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            // Current Speed

            Text {
                id: speed
                // text: "42"
                text: systemHandler.speed
                color: "#2F2F2F"
                font.pixelSize: 50
                anchors.horizontalCenter: parent.horizontalCenter
            }

            Text {
                id: speedUnits
                text: "m/h"
                color: "#2F2F2F"
                font.pixelSize: 16
                anchors.horizontalCenter: parent.horizontalCenter
                opacity: 1
            }

        }
    }


    LeftColumnCenterPanel {
        id: leftCenterPanel
    }


    Rectangle {
        id: turnSignalLeftCenterPanel
        width: 50
        height: 50
        color: "transparent"
        clip: true
        visible: false
        anchors {
            top: parent.top
            left: parent.left
            leftMargin: parent.width * 0.17
            topMargin: parent.height * 0.08
        }

        Image {
            source: "../assets/turn_light_02.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            smooth: true // Para renderização de alta qualidade
            // para testes
            // MouseArea {
            //     anchors.fill: parent
            //     onClicked: {
            //         console.log("clicked for teste")
            //         blinkTimerLeft.running = false;
            //     }
            // }
        }

        Timer {
            id: blinkTimerLeftCenterPanel
            interval: 500 // 500ms, ou seja, pisca duas vezes por segundo
            running: turnLightLeftCenterPanelOn || emergencyCenterPanel // Inicia automaticamente
            repeat: true // Continua a piscar
            onTriggered: {
                turnSignalLeftCenterPanel.visible = !turnSignalLeftCenterPanel.visible;
            }
            onRunningChanged: {
                if (!running) {
                    turnSignalLeftCenterPanel.visible = false;
                }
            }
        }
    }


    Rectangle {
        id: turnSignalRightCenterPanel
        width: 50
        height: 50
        color: "transparent"
        rotation: 180
        visible: false
        clip: true
        anchors {
            top: parent.top
            right: parent.right
            rightMargin: parent.width * 0.17
            topMargin: parent.height * 0.08
        }

        Image {
            source: "../assets/turn_light_02.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            smooth: true // Para renderização de alta qualidade
        }

        Timer {
            id: blinkTimerRight
            interval: 500 // 500ms, ou seja, pisca duas vezes por segundo
            running: turnLightRightCenterPanelOn || emergencyCenterPanel // Inicia automaticamente
            repeat: true // Continua a piscar
            onTriggered: {
                turnSignalRightCenterPanel.visible = !turnSignalRightCenterPanel.visible
            }
            onRunningChanged: {
                if (!running) {
                    turnSignalRightCenterPanel.visible = false;
                }
            }
        }
    }


    Directions {
        id: indications
    }

    onStopNav: function(stop) {
        console.log("stop no center panel: ", stop);
        root.stopNav(stop);
    }


}




