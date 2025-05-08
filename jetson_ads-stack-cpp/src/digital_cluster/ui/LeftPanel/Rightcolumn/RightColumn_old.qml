import QtQuick 2.15

Rectangle {
    id: rightColumn
    width: parent.height * 0.618 - 10// 20% da largura do container retirar os 10 da margin , ver coluna da esquerda
    anchors {
        top: parent.top
        bottom: parent.bottom
    }
    clip: true
    color: "#3A3A3A"
    radius: 8


    Rectangle {
        id: loaderContainer
        anchors {
            bottom: listContainer.top
            top: parent.top
            left: parent.left
            right: parent.right
            margins: 10
        }
        color: "transparent"

        Loader {
                id: rightLoader
                anchors.fill: parent
                source: "Navigation_right_Mapview.qml"

        }
    }

    Rectangle {
        id: listContainer
        // color: "transparent"
        color: "#1E1E1E"
        // color: "red"
        anchors {
            bottom: parent.bottom
            // bottomMargin: 10
            left: parent.left
            right: parent.right
            margins: 10
        }
        radius: 8
        height: parent.height * 0.15
        clip: true

        List {
            id: menuList
        }
    }
}
