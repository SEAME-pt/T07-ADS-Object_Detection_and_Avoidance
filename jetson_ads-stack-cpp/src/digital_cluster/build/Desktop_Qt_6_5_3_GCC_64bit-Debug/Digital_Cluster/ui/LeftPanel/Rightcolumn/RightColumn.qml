import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {

    id: rightColumn1
    width: parent.height * 0.618 - 10// 20% da largura do container retirar os 10 da margin , ver coluna da esquerda
    anchors {
        top: parent.top
        bottom: parent.bottom
    }
    clip: true
    // color: "#3A3A3A"
    // color: "white"
    radius: 8
    gradient: Gradient {
        GradientStop { position: 0.0; color: "#3A3A3A" }
        GradientStop { position: 0.5; color: "#000000" }
        GradientStop { position: 0.85; color: "#505050" }
        GradientStop { position: 1.0; color: "#252525" }
    }
    border.width: 1
    border.color: "#202020"

    signal startNavigation(bool start)

    property ListModel settingsModel
    property ListModel modesModel

    Item{

        id: stackViewLoader
        anchors {
           fill: parent
           margins: 0
        }
        // color: "transparent" // o Item não precisa de color porque não tem background

        StackView {
            id: stackviewRightColumn
            anchors.fill: parent
            anchors.margins: 0
            initialItem: "List.qml"

            // destroyOnPop: false // não funciona
            // retira as animações de push e pop do stackview
            // popEnter: null
            // popExit: null
            // pushEnter: null
            // pushExit: null
        }

    }

    onStartNavigation: function(start) {
        // console.log("start no right column", start);
        leftPanel.startNavigation(start);
    }


    // Component {
    //         id: settingsPageComponent
    //         Settings {
    //             // Passa o modelo global para a página
    //             model: parent.parent.parent.globalSettingsModel
    //         }
    // }


    Component {
        id: settingsPageComponent
        Settings {
            model: rightColumn.settingsModel // Acesso direto via ID raiz
        }
    }

    Component {
        id: modesPageComponent
        Modes {
            modesModel: rightColumn.modesModel // Acesso direto via ID raiz
        }
    }

    // Função para empurrar Settings.qml (pode ser chamada de List.qml ou outro lugar)
    function showSettings() {
        stackviewRightColumn.push(settingsPageComponent);
    }

    function showModes() {
        stackviewRightColumn.push(modesPageComponent);
    }

}
