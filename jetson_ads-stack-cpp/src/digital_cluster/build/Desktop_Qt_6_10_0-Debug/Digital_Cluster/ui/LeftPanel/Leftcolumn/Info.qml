import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {

    id: info
    anchors {
        // top: parent.top
        // bottom: parent.bottom
        // horizontalCenter: parent.horizontalCenter
        fill: parent
    }

    width: parent.width * 0.6 // 60% da largura do container
    color: "#3A3A3A"
    radius: 8


    Column {
        // anchors.centerIn: parent
        anchors.fill: parent
        anchors.top: parent.top // Defina como base o topo do `parent`
        // anchors.topMargin: 25 // Margem de 50 pixels a partir do topo
        spacing: 10

        Rectangle {
            height: 30 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

        padding: 5

        // Total de km
        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 15

            Rectangle {
                id: totalKmIcon
                width: 40
                height: 40
                color: "transparent"
                clip: true

                Image {
                    source: "../../assets/odometer.png"
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                }
            }

            Column {
                spacing: 2

                Text {
                    text: "TOTAL KM"
                    color: "#CCCCCC"
                    font.pixelSize: 14
                    font.bold: true
                    horizontalAlignment: Text.AlignLeft
                }

                Text {
                    text: "4200 Km"
                    color: "white"
                    font.pixelSize: 25
                    font.bold: true
                    horizontalAlignment: Text.AlignLeft
                }
            }
        }


        // Linha separadora
        Rectangle {
            width: parent.width * 0.8
            height: 1
            color: "#555555"
            anchors.horizontalCenter: parent.horizontalCenter
        }

        // Espaçamento específico para este ponto
        Rectangle {
            height: 10 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }


        // Trip
        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 30

            Rectangle {
                id: tripIcon
                width: 30
                height: 30
                color: "transparent"
                clip: true

                Image {
                    source: "../../assets/trip.png"
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                }
            }

            Column {
                spacing: 2

                Text {
                    text: "TRIP"
                    color: "#CCCCCC"
                    font.pixelSize: 14
                    font.bold: true
                    horizontalAlignment: Text.AlignLeft
                }

                Text {
                    text: "4200 Km"
                    color: "white"
                    font.pixelSize: 20
                    font.bold: true
                    horizontalAlignment: Text.AlignLeft
                }
            }
        }


        Rectangle {
            height: 40 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }


        Rectangle {
            id: batteryIcon
            width: 15
            height: 15
            color: "transparent"
            clip: true
            anchors.horizontalCenter: parent.horizontalCenter

            Image {
                source: "../../assets/battery.png"
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                smooth: true // Para renderização de alta qualidade
            }

        }

        ProgressBar {
            id: batteryLevel
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width * 0.3
            value: 0.7
            padding: 2

            background: Rectangle {
                // implicitWidth: 200 // ver Customization da ajuda
                implicitWidth: parent.width
                implicitHeight: 6
                color: "#e6e6e6"
                radius: 3
            }

            contentItem: Item {
                // implicitWidth: 200
                implicitWidth: parent.width
                implicitHeight: 4

                Rectangle {
                    width: batteryLevel.visualPosition * parent.width
                    height: parent.height
                    radius: 2
                    color: "#17a81a"
                }
            }
        }


        Text {
            id: batteryInfo
            text: "70%"
            // text:systemHandler.batteryPer +"%"
            color: "white"
            font.pixelSize: 17
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Rectangle {
            height: 30 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

        Gear {
            id: gear
            // onGearSelected: leftPanel.gearSelected(gear) // Propaga o sinal para o LeftPanel
            onGearSelected: function(selectedGear) {
                leftPanel.gearSelected(selectedGear); // Propaga o sinal com o parâmetro correto
                centerColumn.gearSelected(selectedGear);
            }
        }

    }


}
