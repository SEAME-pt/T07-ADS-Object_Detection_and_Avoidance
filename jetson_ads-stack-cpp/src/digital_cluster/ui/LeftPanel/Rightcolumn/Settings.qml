import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: settings
    color: "#1E1E1E"
    radius: 10

    property ListModel model

    //Debug da passagem do modelo global que vem da Main
    Component.onCompleted: {
            console.log("Modelo em Settings:", model.count, "itens");
            console.log("Primeiro item do modelo:", model.get(0).name);
    }

    Column {
        id: menuSettings
        anchors.fill: parent
        anchors.margins: 15
        spacing: 8

        Repeater {
            // model: [
            //     { name: "Lane Keep Assist"},
            //     { name: "Autopilot"},
            //     { name: "Colors"}
            // ]

            model: settings.model

            delegate: Rectangle {
                id: menuItem
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

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    onClicked: {
                        // console.log("Clic:", model.name);
                        // Atualiza diretamente no modelo global
                        // settings.model.setProperty(index, "selected", !model.selected) - só tenho de enviar para o systemHandler
                        if (model.name === "Lkas") {
                             var newValueLkas = !model.selected;
                             systemHandler.lkas = newValueLkas ? "true" : "false"; /// muda no cpp - system Handler
                         } else if (model.name === "Autopilot") {
                            var newValueAutoPilot = !model.selected;
                            systemHandler.autoPilot = newValueAutoPilot ? "true" : "false";
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
