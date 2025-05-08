import QtQuick 2.15


Rectangle{
    id:buttonsRight
    color: "transparent"
    width: 210
    height: 210
    anchors.right: parent.right
    anchors.rightMargin: 50
    anchors.verticalCenter: parent.verticalCenter

    // Botão Y (topo)
    Rectangle {
        width: 70
        height: 70
        color: "#FFCC00" // Amarelo
        radius: width / 2
        x: 70
        y: 0

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
                // console.log("Y pressionado")
                gamepadPage.buttonPressed("Y")
                yTimer.start() // Inicia o Timer
            }
            onReleased: {
                // console.log("Y solto")
                yTimer.stop() // Para o Timer
            }
        }

        Timer {
            id: yTimer
            interval: 100
            repeat: true
            onTriggered: {
                // console.log("Y (contínuo)")
                gamepadPage.buttonPressed("Y") // resposta imediata ao pressionar o botão, sem qualquer atraso inicial
            }
        }
    }

    // Botão X (esquerda)
    Rectangle {
        width: 70
        height: 70
        color: "#0066FF" // Azul
        radius: width / 2
        x: 0
        y: 70

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
                // console.log("X pressionado")
                gamepadPage.buttonPressed("X")
                xTimer.start() // Inicia o Timer
            }
            onReleased: {
                // console.log("X solto")
                xTimer.stop() // Para o Timer
            }
        }

        Timer {
            id: xTimer
            interval: 100
            repeat: true
            onTriggered: {
                // console.log("X (contínuo)")
                gamepadPage.buttonPressed("X") // resposta imediata ao pressionar o botão, sem qualquer atraso inicial
            }
        }
    }

    // Botão B (direita)
    Rectangle {
        width: 70
        height: 70
        color: "#FF3333" // Vermelho
        radius: width / 2
        x: 140
        y: 70

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
                // console.log("B pressionado")
                gamepadPage.buttonPressed("B")
                bTimer.start() // Inicia o Timer
            }
            onReleased: {
                // console.log("B solto")
                bTimer.stop() // Para o Timer
            }
        }

        Timer {
            id: bTimer
            interval: 100
            repeat: true
            onTriggered: {
                // console.log("B (contínuo)")
                gamepadPage.buttonPressed("B") // resposta imediata ao pressionar o botão, sem qualquer atraso inicial
            }
        }
    }

    // Botão A (baixo)
    Rectangle {
        width: 70
        height: 70
        color: "#00CC00" // Verde
        radius: width / 2
        x: 70
        y: 140

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
                // console.log("A pressionado")
                gamepadPage.buttonPressed("A")
                aTimer.start() // Inicia o Timer
            }
            onReleased: {
                // console.log("A solto")
                aTimer.stop() // Para o Timer
            }
        }

        Timer {
            id: aTimer
            interval: 100
            repeat: true
            onTriggered: {
                // console.log("A (contínuo)")
                gamepadPage.buttonPressed("A") // resposta imediata ao pressionar o botão, sem qualquer atraso inicial
            }
        }
    }
}

