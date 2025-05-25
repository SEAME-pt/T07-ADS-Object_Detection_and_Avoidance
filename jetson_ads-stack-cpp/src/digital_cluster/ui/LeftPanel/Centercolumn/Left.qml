import QtQuick 2.15

Rectangle {
    id: left
    anchors {
        top: parent.top
        left: parent.left
        bottom: parent.bottom
        margins: 25
    }
    signal gearSelected(string gear)
    width: parent.width * 0.1
    color: "transparent"
    // border.color: "white" // Borda do retângulo principal
    // border.width: 2

    Column {
        anchors.fill: parent // O Column preenche o retângulo pai
        spacing: 0 // Sem espaçamento entre as linhas

        // Linha de cima: contém o speedLimit
        Rectangle {
            id: topRow
            width: parent.width
            height: 60 // Altura fixa para a linha de cima
            color: "transparent"
            // border.color: "white" // Borda para visualização
            // border.width: 2

            // Speed Limit
            Rectangle {
                id: speedLimit
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    verticalCenter: parent.verticalCenter // Centraliza verticalmente na linha
                }
                width: 60
                height: 60
                color: "white"
                radius: 50
                border.color: "red"
                border.width: 4
                Text {
                    text: "70"
                    font.pixelSize: 30
                    anchors.centerIn: parent
                }
            }
        }

        // Linha de baixo: contém os sinais
        Rectangle {
            id: bottomRow
            width: parent.width
            height: parent.height - topRow.height // Ocupa o espaço restante
            color: "transparent"
            // border.color: "white" // Borda para visualização
            // border.width: 2

            Signs {
                id: signs
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    bottom: parent.bottom // Alinha os sinais à parte inferior da linha de baixo
                }
            }
        }
    }

    onGearSelected: function (gear) {
        signs.gearSelected(gear);
    }
}
