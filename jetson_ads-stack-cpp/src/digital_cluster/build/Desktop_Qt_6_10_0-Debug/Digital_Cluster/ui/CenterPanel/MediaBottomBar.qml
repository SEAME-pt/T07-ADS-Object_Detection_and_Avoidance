import QtQuick 2.15
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects

Rectangle {
    id: mediaBottomBar
    // width: 120
    implicitWidth: musicRow.implicitWidth
    height: parent.height
    anchors {
        right: parent.right
        bottom: parent.bottom
        top:parent.top
        margins: 15
    }

    color: "transparent"
    // border.color: "white"
    radius: 10
    Row {
        id:musicRow
        spacing: 10
        anchors.centerIn: parent

        Column {
            spacing: 5

            Text {
                text: "Nirvana"
                font.pixelSize: 16
                font.bold: true
                color: "lightgray"
                anchors.right: parent.right
            }
            Text {
                id: titleSong
                text: "The man who sold the world"
                font.pixelSize: 10
                color: "white"
            }

            ProgressBar {
                id: progressSong

                anchors.horizontalCenter: parent.horizontalCenter
                // width: parent.width * 0.65
                implicitWidth: titleSong.width
                value: 0.7
                // padding: 2

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
                        width: progressSong.visualPosition * parent.width
                        height: parent.height
                        radius: 2
                        color: "#808080"
                    }
                }
            }

            Row {
                // spacing: 10
                anchors.horizontalCenter: parent.horizontalCenter
                width: progressSong.width

                // Start Time
                Text {
                    text: "1:25"
                    color: "white"
                    font.pixelSize: 12
                }

                Rectangle {
                    height: 10 // Espaçamento específico para este ponto
                    width: parent.width * 0.65
                    color: "transparent"
                }

                // End Time
                Text {
                    text: "4:34"
                    color: "white"
                    font.pixelSize: 12
                }
            }

        }
    }
}
