import QtQuick 2.15
import QtQuick.Controls 2.15

// Área do D-pad (esquerda)
Rectangle {
    id: dpad
    width: 150
    height: 150
    color: "#333333"
    radius: 10
    anchors.left: parent.left
    anchors.leftMargin: 50
    anchors.verticalCenter: parent.verticalCenter

    // Símbolos do D-pad (triângulos para indicar direções)
    // Cima (▲)
    Item {
        width: 50
        height: 50
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 10

        Text {
            text: "▲"
            color: "white"
            font.pixelSize: 30
            anchors.centerIn: parent
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("D-pad: Cima clicado")
                gamepadPage.buttonPressed("DpadUp")
            }
        }
    }

    // Baixo (▼)
    Item {
        width: 50
        height: 50
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10

        Text {
            text: "▼"
            color: "white"
            font.pixelSize: 30
            anchors.centerIn: parent
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("D-pad: Baixo clicado")
                gamepadPage.buttonPressed("DpadDown")
            }
        }
    }

    // Esquerda (◄)
    Item {
        width: 50
        height: 50
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 10

        Text {
            text: "◄"
            color: "white"
            font.pixelSize: 30
            anchors.centerIn: parent
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("D-pad: Esquerda clicado")
                gamepadPage.buttonPressed("DpadLeft")
            }
        }
    }

    // Direita (►)
    Item {
        width: 50
        height: 50
        anchors.verticalCenter: parent.verticalCenter
        anchors.right: parent.right
        anchors.rightMargin: 10

        Text {
            text: "►"
            color: "white"
            font.pixelSize: 30
            anchors.centerIn: parent
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("D-pad: Direita clicado")
                gamepadPage.buttonPressed("DpadRight")
            }
        }
    }
}
