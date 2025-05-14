import QtQuick 2.15

Rectangle {
    id: leftColumn
    // width: parent.width * 0.2 - 10// 20% da largura do container retira os 10 do spacing
    // width: isCenterPanelOn ? parent.width : parent.height * 0.618 - 10
    width: parent.height * 0.618 - 10
    // height: parent.height
    anchors {
        top: parent.top
        bottom: parent.bottom
    }

    color: "#3A3A3A"
    // border.color: "white" // Borda para identificar o retângulo
    // border.width: 2
    radius: 8


    Loader {
            id: leftLoader
            anchors.fill: parent

            source: "Info.qml"

    }

}
