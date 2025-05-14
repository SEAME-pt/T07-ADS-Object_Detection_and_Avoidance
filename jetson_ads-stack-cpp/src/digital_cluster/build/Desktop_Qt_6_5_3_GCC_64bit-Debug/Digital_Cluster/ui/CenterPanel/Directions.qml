import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {

    id: indications
    implicitWidth: textDirections.implicitWidth + 10
    implicitHeight: textDirections.implicitHeight + 10
    radius: 12
    color: "black"
    opacity: 0.8

    anchors {
        top: parent.top
        left: parent.left
        leftMargin: parent.width * 0.25
        topMargin: 20
    }

    // border.color: "white"
    // border.width: 2

    Column {

        id: textDirections
        anchors.fill: parent
        anchors.margins: 5
        spacing: 8

        Text {
            text: "2.7 Km"
            font.pixelSize: 25
            color: "white"
            font.bold: true
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Text {
            text: "Take the exit toward\nCeuta St"
            font.pixelSize: 15
            color: "white"
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
            wrapMode: Text.WordWrap
        }

    }
}
