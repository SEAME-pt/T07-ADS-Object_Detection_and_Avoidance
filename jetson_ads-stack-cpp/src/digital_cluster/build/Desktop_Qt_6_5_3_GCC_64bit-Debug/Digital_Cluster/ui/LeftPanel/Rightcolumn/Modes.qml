import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: modes
    color: "#1E1E1E"
    radius: 10

    property ListModel modesModel

    Column {
        id: menuModes
        anchors.fill: parent
        anchors.margins: 15
        spacing: 8

        Repeater {
            // model: [
            //     { name: "Lane Keep Assist"},
            //     { name: "Autopilot"},
            //     { name: "Colors"}
            // ]

            model: modes.modesModel

            delegate: Rectangle {
                id: menuItemModes
                // width: rightColumn.width
                width: parent.width
                height: 50
                // color: "transparent"
                color: model.selected ? "#1A1A1A" : "transparent"
                radius: 8

                // Component.onCompleted: {
                //         console.log("Delegate criado para:", model.name, "índice:", index);
                //     }

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 15
                    anchors.rightMargin: 15
                    spacing: 20

                    Image {
                        source: model.icon
                        width: 18
                        height: 18
                        anchors.verticalCenter: parent.verticalCenter
                        fillMode: Image.PreserveAspectFit
                        smooth: true
                        mipmap: true
                    }

                    Text {
                        text: model.name
                        // color: "#4A90E2"
                        color: model.selected ? "white" : "#4A90E2"
                        font.pixelSize: 18
                        // font.family: "Arial"
                        font.weight: Font.Medium
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }

                // MouseArea {
                //     id: mouseArea
                //     anchors.fill: parent
                //     onClicked: {
                //         console.log("Clic:", model.name);
                //         // Atualiza diretamente no modelo global
                //         modes.modesModel.setProperty(index, "selected", !model.selected)
                //     }
                // }

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    onClicked: {
                        // console.log("Clic: " + model.name + ", índice: " + index);
                        // Desseleciona todos os itens do modelo, retira o último que é o playmode desta equação
                        if (model.name === "Play") {
                            modes.modesModel.setProperty(index, "selected", !model.selected)
                            mainWindow.updatePlayMode();
                        } else {
                            for (var i = 0; i < modes.modesModel.count - 1; i++) {
                                modes.modesModel.setProperty(i, "selected", false);
                            }
                            // Seleciona apenas o item clicado
                            modes.modesModel.setProperty(index, "selected", true);
                            mainWindow.updateSelectedMode();
                        }

                    }
                }

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


// No seu código QML, você precisa usar modes.modesModel em vez de apenas model no MouseArea porque o model dentro do delegate do Repeater é um escopo local que representa apenas os dados do item atual sendo renderizado (por exemplo, model.name, model.selected, model.icon). Ele não dá acesso direto ao ListModel completo da propriedade modesModel definida no Rectangle raiz (modes). Vamos entender isso passo a passo:
// Contexto do model no Repeater

//     O Repeater usa a propriedade model: modes.modesModel para iterar sobre os elementos do ListModel definido em Main.qml (ou onde quer que esteja o globalModesModel).
//     Dentro do delegate, o model é automaticamente fornecido pelo Repeater como uma referência aos dados de cada elemento individual da lista (ex.: { name: "Normal", selected: false, icon: "..." }).
//     Esse model local é limitado ao escopo do delegate. Ele permite acessar propriedades como model.name, model.selected ou model.icon para o item atual, mas não é o ListModel em si, apenas uma "visão" dos dados daquele índice.

// Por que modes.modesModel?

//     modes.modesModel é a propriedade definida no Rectangle raiz (id: modes), que contém o ListModel completo passado de Main.qml.
//     Quando você precisa manipular o ListModel globalmente (como no for para desselecionar todos os itens ou ao usar setProperty em índices diferentes), você deve referenciar o ListModel original, que é modes.modesModel.
//     Usar apenas model no MouseArea não funciona porque ele não tem métodos como setProperty ou count — esses pertencem ao ListModel, não aos dados do item individual.

