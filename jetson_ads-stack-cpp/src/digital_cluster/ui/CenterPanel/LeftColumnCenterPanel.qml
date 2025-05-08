import QtQuick 2.15

Rectangle {

    id: leftCenterPanel
    anchors {
        top: parent.top
        left: parent.left
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
            id: speedLimit

            anchors {
                horizontalCenter: parent.horizontalCenter
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

        Rectangle {
            height: 20 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

        Rectangle {
            height: 10 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }


    }

    // onGearSelected: function (gear) {
    //     // console.log("gear in left", gear);
    //     signs.gearSelected(gear);
    // }

}
