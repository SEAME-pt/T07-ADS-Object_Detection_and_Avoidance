import QtQuick 2.15
import QtQuick.Controls 2.15
import "./Leftcolumn"
import "./Rightcolumn"
import "./Centercolumn"

// Left panel
Rectangle {
    id: leftPanel
    width: isCenterPanelOn ? (parent.height * 0.618 - 10) : parent.width // não deve ser necessário
    // width: (parent.height * 0.618 - 10)
    // color: "#272727"
    color: "transparent"
    radius: 8
    anchors {
        bottom: parent.bottom
        left: parent.left
        top: parent.top
        right: isCenterPanelOn ? stackview.left : parent.right // verificar se funciona, é necessário para aplicar a margin à dir
        margins: 10
    }

    signal gearSelected(string gear)
    signal startNavigation(bool start)

    property ListModel settingsModel
    property ListModel modesModel
    property string leftPanelSelectedMode

    // Component.onCompleted: {
    //     console.log("Modo na leftPanel: ", leftPanelSelectedMode);
    // }

    Row {
        id: rowLayout
        spacing: 10 // Espaçamento entre as colunas
        anchors.fill: parent

        // Coluna da esquerda
        LeftColumn {
            id: leftColumn
            visible: true
        }


        CenterColumn {
            id: centerColumn
            // visible: true
            centerColumnSelectedMode: leftPanel.leftPanelSelectedMode
            visible: isCenterPanelOn ? false : true
        }


        RightColumn {
            id: rightColumn
            // visible: true
            settingsModel: leftPanel.settingsModel
            modesModel: leftPanel.modesModel
            visible: isCenterPanelOn ? false : true
        }
    }

    onGearSelected: function(gear) {
        root.gearSelected(gear);
    }

    onStartNavigation: function(start) {
        // console.log("start no left panel", start);
        root.startNavigation(start);
    }
}
