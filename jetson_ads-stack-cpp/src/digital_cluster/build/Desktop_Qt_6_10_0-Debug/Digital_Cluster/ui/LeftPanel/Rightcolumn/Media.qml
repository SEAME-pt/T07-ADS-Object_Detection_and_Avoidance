import QtQuick 2.15
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects


Rectangle {
    id: media
    color: "#1E1E1E"
    radius: 10

    Column {
        id:infoAlbum
        spacing: 10
        anchors.fill: parent
        anchors.margins: 15


        Rectangle {
            height: 20 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }


        Rectangle {
            id: albumCover
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width * 0.60
            height: width
            color: "transparent"
            clip: true

            Image {
                id:albumArt
                source: "../../assets/nirvana.jpg"
                anchors.fill: parent
                fillMode: Image.PreserveAspectCrop
                layer.enabled: true
                layer.effect: OpacityMask {
                    maskSource: mask
                }
            }

            Rectangle {
                id: mask
                width: albumArt.width
                height: albumArt.height
                radius: 10 // Define o raio dos cantos arredondados
                visible: true
                color: "black"
                opacity: 0.3
            }
        }


        Rectangle {
            height: 10 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }


        Text {
            id: songTitle
            text: " The man who sold the world"
            color: "white"
            font.pixelSize: 12
            anchors.horizontalCenter: parent.horizontalCenter
        }


        ProgressBar {
            id: progressSong
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width * 0.55
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
                    width: progressSong.visualPosition * parent.width
                    height: parent.height
                    radius: 2
                    color: "#808080"
                }
            }
        }


        Row {
            id: timersAlbum
            // spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter
            width: progressSong.width

            Text {
                id: startTime
                text: "1:25"
                color: "white"
                font.pixelSize: 12
            }

            Rectangle {
                height: 10 // Espaçamento específico para este ponto
                width: parent.width * 0.55
                color: "transparent"
            }

            Text {
                id: endTime
                text: "4:34"
                color: "white"
                font.pixelSize: 12
            }
        }
    }

    // Botão de voltar
    Rectangle {
        id: btnBack
        anchors {
            bottom: parent.bottom
        }

        width: parent.width
        height: 50
        // color: mouseAreaBack.containsMouse ? "#3A3A3A" : "transparent"
        color: "transparent"
        radius: 5

        Row {
            anchors.centerIn: parent
            spacing: 10

            Image {
                source: "../../assets/arrow_back.svg"
                width: 18
                height: 18
                anchors.verticalCenter: parent.verticalCenter
            }

            Text {
                text: "Back"
                color: "white"
                font.pixelSize: 18
                font.weight: Font.Medium
            }
        }

        MouseArea {
            id: mouseAreaBack
            anchors.fill: parent
            // hoverEnabled: true
            onClicked: {
                console.log("Back clicked");
                stackviewRightColumn.pop()
            }
        }
    }
}
