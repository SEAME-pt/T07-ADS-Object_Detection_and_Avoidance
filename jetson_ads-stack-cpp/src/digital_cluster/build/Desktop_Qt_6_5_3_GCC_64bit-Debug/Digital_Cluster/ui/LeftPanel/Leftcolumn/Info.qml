import QtQuick 2.15
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects

Rectangle {
    id: info
    anchors.fill: parent
    radius: 8
    gradient: Gradient {
        GradientStop { position: 0.0; color: "#3A3A3A" }
        GradientStop { position: 0.5; color: "#000000" }
        GradientStop { position: 0.85; color: "#505050" }
        GradientStop { position: 1.0; color: "#252525" }
    }
    border.width: 1
    border.color: "#202020"
    Column {
        anchors.fill: parent
        anchors.top: parent.top
        spacing: 10
        padding: 5

        Rectangle {
            height: 30
            width: parent.width
            color: "transparent"
        }

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

        Rectangle {
            width: parent.width * 0.8
            height: 1
            color: "#555555"
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Rectangle {
            height: 10
            width: parent.width
            color: "transparent"
        }

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
            height: 40
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
                smooth: true
            }
        }

        ProgressBar {
            id: batteryLevel
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width * 0.3
            value: 0.7
            padding: 2

            background: Rectangle {
                implicitWidth: parent.width
                implicitHeight: 6
                color: "#e6e6e6"
                radius: 3
            }

            contentItem: Item {
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
            text: systemHandler.batteryPer + "%"
            color: "white"
            font.pixelSize: 17
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Rectangle {
            height: 30
            width: parent.width
            color: "transparent"
        }

        Gear {
            id: gear
            onGearSelected: function(selectedGear) {
                leftPanel.gearSelected(selectedGear)
                centerColumn.gearSelected(selectedGear)
            }
        }
    }
}
