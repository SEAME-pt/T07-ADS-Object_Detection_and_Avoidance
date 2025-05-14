import QtQuick 2.15
import QtQuick.Controls 2.15
import "../LeftPanel"
import "../CenterPanel"

Rectangle {
    id: root
    width: 1280
    height: 400
    color: "black" // Background color of the cluster

    signal gearSelected(string gear)
    signal startNavigation(bool start)
    signal stopNav(bool stop)

    property string gearRoot: "P" // tem P no Gear ao inicio

    property string rootSelectedMode: mainWindow.selectedMode


    LeftPanel {
        id: leftPanel
        property bool isCenterPanelOn
        leftPanelSelectedMode: mainWindow.selectedMode

        // declarar a propriedade correspondente no leftPanel
        settingsModel: mainWindow.globalSettingsModel // para passar o ListModel global para o left panel
        modesModel: mainWindow.globalModesModel
        onGearSelected: function(gear) {
            if (gear === "D") {
                console.log("Driving mode activated");
                // centerPanel.push(centerPanel.carDrivingPage);
            } else {
                console.log("Stopped mode activated");
                // centerPanel.push(centerPanel.carStoppedPage);
            }
        }

    }

    StackView {
        id: stackview
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: 10
        width: parent.width - (parent.height * 0.618 + 10)
        initialItem: Item {width: 0; height: 0}
        pushEnter: Transition {
                PropertyAnimation {
                    property: "x"
                    from: stackview.width // Começa fora da tela (direita)
                    to: 0                 // Fica visível na posição final
                    duration: 1000
                    easing.type: Easing.OutCubic
                }
        }

    }

    onGearSelected: function(gear) {
        console.log("gear no root: ", gear);
        // stackview.gearSelected(gear);
        // Se o CenterPanel estiver visível, passa a marcha
       if (stackview.currentItem && stackview.currentItem.isCenterPanel) {
           stackview.currentItem.updateGear(gear);
       }
    }

    onStartNavigation: function (start) {
        // console.log("start no root: ", start);
        if(start) {
            leftPanel.isCenterPanelOn = true;
            // centerLoader.source = "ui/CenterPanel/CenterPanel.qml";
            stackview.push("../CenterPanel/CenterPanel.qml");
        }
    }

    onStopNav: function(stop) {
        console.log("stop no root: ", stop);
        if (stop) {
            leftPanel.isCenterPanelOn = false;
            stackview.pop();
        }
    }

}

