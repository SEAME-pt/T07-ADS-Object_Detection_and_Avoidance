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
            onClicked: {
                console.log("Y clicado")
                gamepadPage.buttonPressed("Y")
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
            onClicked: {
                console.log("X clicado")
                gamepadPage.buttonPressed("X")
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
            onClicked: {
                console.log("B clicado")
                gamepadPage.buttonPressed("B")
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
            onClicked: {
                console.log("A clicado")
                gamepadPage.buttonPressed("A")
            }
        }
    }
}

