import QtQuick
import QtQuick.Controls 2.15
import "ui/LeftPanel"
import "ui/CenterPanel"
import "ui/Main"
import "ui/Play"

Window {
    id: mainWindow
    width: 1280
    height: 400
    // width: Screen.width
    // height: Screen.height
    visible: true
    // visibility: Window.FullScreen
    title: qsTr("Cluster v2")

    // Permite fechar ao pressionar "Esc", o Focusscope ouve os eventos do teclado
    FocusScope {
            anchors.fill: parent // O FocusScope cobre toda a janela
            focus: true // Garante que o FocusScope capture os eventos de teclado

            Keys.onPressed:  function test (event) {
                if (event.key === Qt.Key_Escape) {
                    Qt.quit(); // Fecha a aplicação
                }
            }
    }

    // declarar o model como propriedade raiz
    property ListModel globalSettingsModel: ListModel {
        ListElement { name: "Lkas"; icon: "../../assets/lkas.svg"; selected: false }
        ListElement { name: "Autopilot"; icon: "../../assets/autoSteer.svg"; selected: false }
        //ListElement { name: "Colors"; selected: false }
    }


    // muda o valor do selected no listmodel
    property bool lkasSignal: systemHandler.lkas === "true"
    onLkasSignalChanged: {
        globalSettingsModel.setProperty(0, "selected", lkasSignal)
    }

    property bool autoPilotSignal: systemHandler.autoPilot === "true"
    onAutoPilotSignalChanged: {
        globalSettingsModel.setProperty(1, "selected", autoPilotSignal)
    }

    property string selectedMode: "Normal"
    property bool playMode: false

    property ListModel globalModesModel: ListModel {
        ListElement { name: "Normal"; icon: "../../assets/normalMode.svg"; selected: true }
        ListElement { name: "Eco"; icon: "../../assets/ecoMode.svg"; selected: false }
        ListElement { name: "Sport"; icon: "../../assets/sportMode.svg"; selected: false }
        ListElement { name: "Play"; icon: "../../assets/playMode.svg"; selected: false }
    }


    function updateSelectedMode() {
        for (var i = 0; i < globalModesModel.count; i++) {
            if (globalModesModel.get(i).selected) {
                selectedMode = globalModesModel.get(i).name;
                // console.log("Modo selecionado atualizado:", selectedMode);
                break;
            }
        }
    }


    function updatePlayMode() {
        if (globalModesModel.get(3).selected) {
            playMode = true;
            console.log("Modo play atualizado:", playMode);
        } else {
            playMode = false;
            console.log("Modo play atualizado:", playMode);
        }
    }

    // Component.onCompleted: {
    //     // updateSelectedMode(); // Define o modo inicial
    //     updatePlayMode();
    //     console.log("Modo na mainWindow: ", selectedMode);
    // }


    // Rectangle {
    //     id: root
    //     width: 1280
    //     height: 400
    //     color: "black" // Background color of the cluster

    //     signal gearSelected(string gear)
    //     signal startNavigation(bool start)
    //     signal stopNav(bool stop)

    //     property string gearRoot: "P" // tem P no Gear ao inicio

    //     property string rootSelectedMode: mainWindow.selectedMode


    //     LeftPanel {
    //         id: leftPanel
    //         property bool isCenterPanelOn
    //         leftPanelSelectedMode: mainWindow.selectedMode

    //         // declarar a propriedade correspondente no leftPanel
    //         settingsModel: mainWindow.globalSettingsModel // para passar o ListModel global para o left panel
    //         modesModel: mainWindow.globalModesModel
    //         onGearSelected: function(gear) {
    //             if (gear === "D") {
    //                 console.log("Driving mode activated");
    //                 // centerPanel.push(centerPanel.carDrivingPage);
    //             } else {
    //                 console.log("Stopped mode activated");
    //                 // centerPanel.push(centerPanel.carStoppedPage);
    //             }
    //         }

    //     }


    //     StackView {
    //         id: stackview
    //         anchors.right: parent.right
    //         anchors.top: parent.top
    //         anchors.bottom: parent.bottom
    //         anchors.margins: 10
    //         width: parent.width - (parent.height * 0.618 + 10)
    //         initialItem: Item {width: 0; height: 0}
    //         pushEnter: Transition {
    //                 PropertyAnimation {
    //                     property: "x"
    //                     from: stackview.width // Começa fora da tela (direita)
    //                     to: 0                 // Fica visível na posição final
    //                     duration: 1000
    //                     easing.type: Easing.OutCubic
    //                 }
    //         }

    //     }

    //     onGearSelected: function(gear) {
    //         console.log("gear no root: ", gear);
    //         // stackview.gearSelected(gear);
    //         // Se o CenterPanel estiver visível, passa a marcha
    //        if (stackview.currentItem && stackview.currentItem.isCenterPanel) {
    //            stackview.currentItem.updateGear(gear);
    //        }
    //     }

    //     onStartNavigation: function (start) {
    //         // console.log("start no root: ", start);
    //         if(start) {
    //             leftPanel.isCenterPanelOn = true;
    //             // centerLoader.source = "ui/CenterPanel/CenterPanel.qml";
    //             stackview.push("ui/CenterPanel/CenterPanel.qml");
    //         }
    //     }

    //     onStopNav: function(stop) {
    //         console.log("stop no root: ", stop);
    //         if (stop) {
    //             leftPanel.isCenterPanelOn = false;
    //             stackview.pop();
    //         }
    //     }

    // }

    StackView {
        id: stackViewMain
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        // anchors.margins: 10
        anchors.left: parent.left
        // initialItem: Item {width: 0; height: 0}
        initialItem: "ui/Main/Root.qml"
        pushEnter: Transition {
            PropertyAnimation {
                property: "x"
                from: stackViewMain.width // Começa fora da tela (direita)
                to: 0                 // Fica visível na posição final
                duration: 1000
                easing.type: Easing.OutCubic
            }
        }
    }

    onPlayModeChanged: function() {
        if (playMode){
            stackViewMain.push("ui/Play/Gamepad.qml")
        } else {
            stackViewMain.pop();
        }
    }

    //teste c++ -> QML
    Rectangle {
        Component.onCompleted: {
            console.log("systemHandler está acessível no QML:", typeof systemHandler);
        }
    }

}
