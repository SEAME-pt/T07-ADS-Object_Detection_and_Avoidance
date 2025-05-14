import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: dpad
    width: 150
    height: 150
    color: "#333333"
    radius: 10
    anchors.left: parent.left
    anchors.leftMargin: 50
    anchors.verticalCenter: parent.verticalCenter

    // Estado para ver se o botão está pressionado
    property bool isPressed: false
    property string currentDirection: ""

    // Timer para atrasar o início da emissão contínua
    Timer {
        id: delayTimer
        interval: 200 // Atraso de 200ms para distinguir clique único de pressão contínua
        repeat: false
        onTriggered: {
            if (dpad.isPressed) {
                // Se o botão ainda está pressionado após o atraso, inicia a emissão contínua
                continuousTimer.start()
            }
        }
    }

    // Timer para emissão contínua
    Timer {
        id: continuousTimer
        interval: 100 // Emite o sinal a cada 100ms durante a pressão contínua
        repeat: true
        onTriggered: {
            if (dpad.currentDirection !== "") {
                console.log("D-pad: " + dpad.currentDirection + " (contínuo)")
                gamepadPage.buttonPressed("D-pad-" + dpad.currentDirection)
            }
        }
    }

    // Função para iniciar o D-pad
    function startDpad(direction) {
        dpad.isPressed = true
        dpad.currentDirection = direction
        delayTimer.start() // Inicia o Timer de atraso
    }

    // Função para parar o D-pad
    function stopDpad(direction) {
        if (dpad.isPressed && !delayTimer.running) {
            // Se o botão foi solto após o atraso (emissão contínua já começou), não faz nada extra
        } else if (dpad.isPressed) {
            // Se o botão foi solto antes do atraso (clique único), emite o sinal uma vez
            console.log("D-pad: " + direction + " (clique único)")
            gamepadPage.buttonPressed("D-pad-" + direction)
        }
        dpad.isPressed = false
        dpad.currentDirection = ""
        delayTimer.stop()
        continuousTimer.stop()
    }

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
            onPressed: {
                console.log("D-pad: Cima pressionado")
                dpad.startDpad("Up")
            }
            onReleased: {
                console.log("D-pad: Cima solto")
                dpad.stopDpad("Up")
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
            onPressed: {
                console.log("D-pad: Baixo pressionado")
                dpad.startDpad("Down")
            }
            onReleased: {
                console.log("D-pad: Baixo solto")
                dpad.stopDpad("Down")
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
            onPressed: {
                console.log("D-pad: Esquerda pressionado")
                dpad.startDpad("Left")
            }
            onReleased: {
                console.log("D-pad: Esquerda solto")
                dpad.stopDpad("Left")
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
            onPressed: {
                console.log("D-pad: Direita pressionado")
                dpad.startDpad("Right")
            }
            onReleased: {
                console.log("D-pad: Direita solto")
                dpad.stopDpad("Right")
            }
        }
    }
}
