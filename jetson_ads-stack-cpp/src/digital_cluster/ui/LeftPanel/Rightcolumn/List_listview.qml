import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: rightColumn
    // width: parent.height * 0.618 - 10
    anchors {
        // top: parent.top
        // bottom: parent.bottom
        fill: parent
    }
    clip: true
    color: "#3A3A3A"
    radius: 10

    ListView {
        id: menuList
        anchors.fill: parent
        anchors.margins: 15
        // clip: true
        spacing: 8

        model: ListModel {
            ListElement { name: "Settings"; icon: "../../assets/settings_r.svg"; page: "Settings.qml" }
            ListElement { name: "Modes"; icon: "../../assets/modes_r.svg"; page: "Modes.qml" }
            ListElement { name: "Navigation"; icon: "../../assets/navigation_r.svg"; page: "Navigation_right_Mapview.qml" }
            ListElement { name: "Media"; icon: "../../assets/media_r.svg"; page: "Media.qml" }
        }

        delegate: Rectangle {
            id: menuItem
            width: parent.width
            height: 60
            // color: (menuList.currentIndex === index ? "#D32F2F" : "transparent")
            color: "transparent"
            radius: 8

            property bool hovered: false

            Behavior on color {
                ColorAnimation { duration: 150 }
            }

            Row {
                anchors.fill: parent
                anchors.leftMargin: 15
                anchors.rightMargin: 15
                spacing: 20

                Image {
                    source: icon
                    width: 28
                    height: 28
                    anchors.verticalCenter: parent.verticalCenter
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    mipmap: true
                }

                Text {
                    text: name
                    // color: menuList.currentIndex === index ? "#4A90E2" : "white"
                    color: "#4A90E2"
                    font.pixelSize: 18
                    font.family: "Arial"
                    font.weight: Font.Medium
                    anchors.verticalCenter: parent.verticalCenter
                    Behavior on color {
                        ColorAnimation { duration: 150 }
                    }
                }
            }


            MouseArea {
                id: mouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    menuList.currentIndex = index
                    console.log("index right column: ", menuList.currentIndex);
                    console.log("typeof rightloader right column: ", typeof rightLoader);
                    if (typeof rightLoader !== "undefined") {
                        rightLoader.source = page
                    }
                }

                // onEntered: {
                //     console.log("Mouse entered the area")
                //     menuItem.hovered = true  // Update custom property on enter
                //     }

                // onExited: {
                //     console.log("Mouse exited the area")
                //     menuItem.hovered = false
                // }
            }

            // Rectangle {
            //     anchors.fill: parent
            //     color: "transparent"
            //     radius: 8
            //     // border.color: mouseArea.containsMouse ? "#555555" : "transparent"
            //     border.color: menuItem.hovered ? "#555555" : "red"
            //     border.width: 4
            //     opacity: 1  //0.3
            // }



        }
    }
}
