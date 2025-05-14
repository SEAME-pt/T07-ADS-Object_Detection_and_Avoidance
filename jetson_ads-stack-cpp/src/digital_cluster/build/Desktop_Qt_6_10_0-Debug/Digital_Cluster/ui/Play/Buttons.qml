import QtQuick 2.15


// Botões de ação (A, B, X, Y) à direita
Item {
    width: 150
    height: 150
    anchors.right: parent.right
    anchors.rightMargin: 50
    anchors.verticalCenter: parent.verticalCenter

    // Botão Y (topo)
    Rectangle {
        id: buttonY
        width: 50
        height: 50
        color: isPressed ? "#FFD700" : "#FFCC00" // Amarelo mais claro quando pressionado
        radius: width / 2
        x: 50
        y: 0

        property bool isPressed: false

        Text {
            anchors.centerIn: parent
            color: "black"
            text: "Y"
            font.bold: true
            font.pixelSize: 20
        }

        MouseArea {
            anchors.fill: parent
            onPressed: {
                console.log("Y pressionado")
                buttonY.isPressed = true
                yDelayTimer.start()
            }
            onReleased: {
                console.log("Y solto")
                if (buttonY.isPressed && !yDelayTimer.running) {
                    // Emissão contínua já começou, não faz nada extra
                } else if (buttonY.isPressed) {
                    // Clique único
                    console.log("Y (clique único)")
                    gamepadPage.buttonPressed("Y")
                }
                buttonY.isPressed = false
                yDelayTimer.stop()
                yContinuousTimer.stop()
            }
        }

        Timer {
            id: yDelayTimer
            interval: 200
            repeat: false
            onTriggered: {
                if (buttonY.isPressed) {
                    yContinuousTimer.start()
                }
            }
        }

        Timer {
            id: yContinuousTimer
            interval: 100
            repeat: true
            onTriggered: {
                console.log("Y (contínuo)")
                gamepadPage.buttonPressed("Y")
            }
        }
    }

    // Botão X (esquerda)
    Rectangle {
        id: buttonX
        width: 50
        height: 50
        color: isPressed ? "#3399FF" : "#0066FF" // Azul mais claro quando pressionado
        radius: width / 2
        x: 0
        y: 50

        property bool isPressed: false

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "X"
            font.bold: true
            font.pixelSize: 20
        }

        MouseArea {
            anchors.fill: parent
            onPressed: {
                console.log("X pressionado")
                buttonX.isPressed = true
                xDelayTimer.start()
            }
            onReleased: {
                console.log("X solto")
                if (buttonX.isPressed && !xDelayTimer.running) {
                    // Emissão contínua já começou, não faz nada extra
                } else if (buttonX.isPressed) {
                    // Clique único
                    console.log("X (clique único)")
                    gamepadPage.buttonPressed("X")
                }
                buttonX.isPressed = false
                xDelayTimer.stop()
                xContinuousTimer.stop()
            }
        }

        Timer {
            id: xDelayTimer
            interval: 200
            repeat: false
            onTriggered: {
                if (buttonX.isPressed) {
                    xContinuousTimer.start()
                }
            }
        }

        Timer {
            id: xContinuousTimer
            interval: 100
            repeat: true
            onTriggered: {
                console.log("X (contínuo)")
                gamepadPage.buttonPressed("X")
            }
        }
    }

    // Botão B (direita)
    Rectangle {
        id: buttonB
        width: 50
        height: 50
        color: isPressed ? "#FF6666" : "#FF3333" // Vermelho mais claro quando pressionado
        radius: width / 2
        x: 100
        y: 50

        property bool isPressed: false

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "B"
            font.bold: true
            font.pixelSize: 20
        }

        MouseArea {
            anchors.fill: parent
            onPressed: {
                console.log("B pressionado")
                buttonB.isPressed = true
                bDelayTimer.start()
            }
            onReleased: {
                console.log("B solto")
                if (buttonB.isPressed && !bDelayTimer.running) {
                    // Emissão contínua já começou, não faz nada extra
                } else if (buttonB.isPressed) {
                    // Clique único
                    console.log("B (clique único)")
                    gamepadPage.buttonPressed("B")
                }
                buttonB.isPressed = false
                bDelayTimer.stop()
                bContinuousTimer.stop()
            }
        }

        Timer {
            id: bDelayTimer
            interval: 200
            repeat: false
            onTriggered: {
                if (buttonB.isPressed) {
                    bContinuousTimer.start()
                }
            }
        }

        Timer {
            id: bContinuousTimer
            interval: 100
            repeat: true
            onTriggered: {
                console.log("B (contínuo)")
                gamepadPage.buttonPressed("B")
            }
        }
    }

    // Botão A (baixo)
    Rectangle {
        id: buttonA
        width: 50
        height: 50
        color: isPressed ? "#33FF33" : "#00CC00" // Verde mais claro quando pressionado
        radius: width / 2
        x: 50
        y: 100

        property bool isPressed: false

        Text {
            anchors.centerIn: parent
            color: "white"
            text: "A"
            font.bold: true
            font.pixelSize: 20
        }

        MouseArea {
            anchors.fill: parent
            onPressed: {
                console.log("A pressionado")
                buttonA.isPressed = true
                aDelayTimer.start()
            }
            onReleased: {
                console.log("A solto")
                if (buttonA.isPressed && !aDelayTimer.running) {
                    // Emissão contínua já começou, não faz nada extra
                } else if (buttonA.isPressed) {
                    // Clique único
                    console.log("A (clique único)")
                    gamepadPage.buttonPressed("A")
                }
                buttonA.isPressed = false
                aDelayTimer.stop()
                aContinuousTimer.stop()
            }
        }

        Timer {
            id: aDelayTimer
            interval: 200
            repeat: false
            onTriggered: {
                if (buttonA.isPressed) {
                    aContinuousTimer.start()
                }
            }
        }

        Timer {
            id: aContinuousTimer
            interval: 100
            repeat: true
            onTriggered: {
                console.log("A (contínuo)")
                gamepadPage.buttonPressed("A")
            }
        }
    }
}

