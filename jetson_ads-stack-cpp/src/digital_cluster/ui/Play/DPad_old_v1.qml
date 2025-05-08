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

    // Timer para emitir o sinal continuamente enquanto o botão estiver pressionado
    Timer {
        id: dpadTimer
        interval: 100 // Emite o sinal a cada 100ms (ajuste conforme necessário)
        repeat: true
        property string currentDirection: ""

        onTriggered: {
            if (currentDirection !== "") {
                // console.log("D-pad: " + currentDirection + " (contínuo)")
                gamepadPage.buttonPressed("Dpad" + currentDirection)
            }
        }
    }

    // Função para iniciar o Timer com a direção correta
    function startDpadTimer(direction) {
        dpadTimer.currentDirection = direction
        dpadTimer.start()
    }

    // Função para parar o Timer
    function stopDpadTimer() {
        dpadTimer.currentDirection = ""
        dpadTimer.stop()
    }

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
            onPressed: {
                // console.log("D-pad: Cima pressionado")
                gamepadPage.buttonPressed("DpadUp") // Emite o sinal imediatamente
                dpad.startDpadTimer("Up") // Inicia o Timer para emissão contínua
            }
            onReleased: {
                // console.log("D-pad: Cima solto")
                dpad.stopDpadTimer() // Para o Timer
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
                // console.log("D-pad: Baixo pressionado")
                gamepadPage.buttonPressed("DpadDown")
                dpad.startDpadTimer("Down")
            }
            onReleased: {
                // console.log("D-pad: Baixo solto")
                dpad.stopDpadTimer()
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
                // console.log("D-pad: Esquerda pressionado")
                gamepadPage.buttonPressed("DpadLeft")
                dpad.startDpadTimer("Left")
            }
            onReleased: {
                // console.log("D-pad: Esquerda solto")
                dpad.stopDpadTimer()
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
                // console.log("D-pad: Direita pressionado")
                gamepadPage.buttonPressed("DpadRight") // resposta imediata ao pressionar o botão, sem qualquer atraso inicial
                dpad.startDpadTimer("Right")
            }
            onReleased: {
                // console.log("D-pad: Direita solto")
                dpad.stopDpadTimer()
            }
        }
    }
}
