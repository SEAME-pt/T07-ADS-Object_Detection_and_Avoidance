import QtQuick 2.15

Rectangle {
    id: right
    anchors {
        top: parent.top
        right: parent.right
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

        // Linha de cima: retângulo vazio com altura 60
        Rectangle {
            id: topRow
            width: parent.width
            height: 60 // Altura fixa para corresponder à linha de cima do lado esquerdo
            color: "transparent"
            // border.color: "white" // Borda para visualização
            // border.width: 2
        }

        // Linha de baixo: contém os sinais
        Rectangle {
            id: bottomRow
            width: parent.width
            height: parent.height - topRow.height // Ocupa o espaço restante
            color: "transparent"
            // border.color: "white" // Borda para visualização
            // border.width: 2

            Signsright {
                id: signsRight
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    bottom: parent.bottom // Alinha os sinais à parte inferior da linha de baixo
                }
            }
        }
    }

    onGearSelected: function (gear) {
        signsRight.gearSelected(gear);
    }
}
