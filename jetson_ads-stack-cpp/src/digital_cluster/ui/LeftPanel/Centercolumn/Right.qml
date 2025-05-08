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
    // height: parent.height
    color: "transparent"


    Column {

        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 10

        //Speed Limit

        Rectangle {
            height: 85 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

        Signsright {
            id: signsRight
        }

        Rectangle {
            height: 10 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }
    }

    onGearSelected: function (gear) {
        // console.log("gear in left", gear);
        signsRight.gearSelected(gear);
    }

}
