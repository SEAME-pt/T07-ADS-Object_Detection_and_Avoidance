import QtQuick 2.15

Rectangle {
    id: menuList

    anchors.fill: parent
    color: "transparent"

    ListView {
    id: optionsList

    anchors.centerIn: parent
    width: parent.width
    height: 50 // Ajuste de altura para caber os itens


    snapMode: ListView.SnapOneItem
    orientation: ListView.Vertical

        model: ListModel {
            ListElement { name: "navigation" ; page: "Navigation_right_Mapview.qml" }
            ListElement { name: "modes" ; page: "modes.qml" }
            ListElement { name: "media" ; page: "Media.qml"}
            ListElement { name: "settings" ; page: "settings.qml"}
        }

        delegate: Item {
            width: parent.width
            height: 50
            // height: Math.abs(optionsList.view.currentIndex - index) === 0 ? 80 : 50 // Aumenta o item no centro
            // opacity: Math.abs(optionsList.currentIndex - index) === 0 ? 1 : 0.5 // Ajusta a opacidade do item

            Text {
                anchors.centerIn: parent
                text: model.name
                // color: "white"
                color: index === optionsList.currentIndex ? "#4A90E2" : "white"
                opacity: index === optionsList.currentIndex ? 1 : 0.2
                font.pixelSize: index === optionsList.currentIndex ? 18 : 10
                // font.pixelSize: 18
            }

            MouseArea {
                id: mouseArea
                anchors.fill: parent
                onClicked: {
                    optionsList.currentIndex = index; // Atualiza o índice do item selecionado
                    rightLoader.source = model.page;
                }
            }
        }

        onContentYChanged: {
            let centerIndex = Math.round(contentY / 50); // Calcula o índice central baseado na altura de cada item
            if (centerIndex !== currentIndex) {
                currentIndex = centerIndex;
            }
            // console.log("Center index: ", centerIndex, model.get(centerIndex).name)
        }
    }
}
