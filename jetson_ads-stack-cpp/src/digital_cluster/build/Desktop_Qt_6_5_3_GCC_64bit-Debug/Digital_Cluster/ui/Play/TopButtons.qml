import QtQuick 2.15

Row {
    id: topButtons
    anchors.horizontalCenter: parent.horizontalCenter
    anchors.bottom: cameraArea.top
    anchors.bottomMargin: 10
    spacing: 20

    Rectangle {
        width: 80
        height: 30
        color: "#666666"
        radius: 5

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "Select"
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("Select clicado")
                gamepadPage.buttonPressed("Select")
            }
        }
    }

    Rectangle {
        width: 80
        height: 30
        color: "#666666"
        radius: 5

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "Start"
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("Start clicado")
                gamepadPage.buttonPressed("Start")
            }
        }
    }

    Rectangle {
        width: 80
        height: 30
        color: "#666666"
        radius: 5

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "End Game"
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("End Game clicado");
                gamepadPage.buttonPressed("EndGame");
            }
        }
    }
}
