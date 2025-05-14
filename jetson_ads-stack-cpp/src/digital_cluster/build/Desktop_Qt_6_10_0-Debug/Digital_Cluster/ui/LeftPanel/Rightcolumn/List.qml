import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: menuRight
    // anchors.fill: parent
    // clip: true
    color: "#3A3A3A"
    radius: 10

    Column {
        id: menuList
        anchors.fill: parent
        anchors.margins: 15
        spacing: 8

        Repeater {
            model: [
                { name: "Settings", icon: "../../assets/settings_r.svg", page: "Settings.qml" },
                { name: "Modes", icon: "../../assets/modes_r.svg", page: "Modes.qml" },
                { name: "Navigation", icon: "../../assets/navigation_r.svg", page: "Navigation_menu.qml" },
                { name: "Media", icon: "../../assets/media_r.svg", page: "Media.qml" }
            ]

            delegate: Rectangle {
                id: menuItem
                // width: rightColumn.width
                width: parent.width
                height: 60
                color: "transparent"
                radius: 8
                property bool hovered: false

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 15
                    anchors.rightMargin: 15
                    spacing: 20

                    Image {
                        source: modelData.icon
                        width: 18
                        height: 18
                        anchors.verticalCenter: parent.verticalCenter
                        fillMode: Image.PreserveAspectFit
                        smooth: true
                        mipmap: true
                    }

                    Text {
                        text: modelData.name
                        color: "#4A90E2"
                        font.pixelSize: 18
                        // font.family: "Arial"
                        font.weight: Font.Medium
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    hoverEnabled: true

                    onEntered: menuItem.hovered = true
                    onExited: menuItem.hovered = false
                    onClicked: {
                        console.log("Clic:", modelData.name)
                        if (modelData.name === "Settings") {
                            // Acessa a função showSettings no RightColumn
                            rightColumn1.showSettings();
                        } else if (modelData.name === "Modes") {
                            rightColumn1.showModes();
                        } else {
                            stackviewRightColumn.push(modelData.page)
                        }
                    }
                }
            }
        }
    }
}

