import QtQuick 2.15

Rectangle {
    id: menuRight
    color: "transparent"
    anchors.fill: parent

    Timer {
        id: delayTimer
        interval: 50 // Atraso para permitir o cálculo do layout
        running: true
        repeat: false
        onTriggered: {
            console.log("Parent teste width:", parent.width)
            console.log("Parent teste height:", parent.height)
        }
    }


    ListModel {
        id: nameModel
        ListElement {file: "../assets/trip.svg"
            name: "trip"; page: "Trip.qml" }
        ListElement {file: "../assets/navigation.svg"
            name: "navigation"; page: "Navigation.qml" }
        ListElement {file: "../assets/modes.svg"
            name: "modes"; page: "Modes.qml" }
        ListElement {file: "../assets/settings.svg"
            name: "settings"; page: "Settings.qml" }
    }


    Component {
        id: nameDelegate

        Column {
            opacity: PathView.opacity
            z: PathView.z
            scale: PathView.scale

            Image {
                anchors.horizontalCenter: delegateText.horizontalCenter
                source: model.file; width: 60; height: 60; smooth: true
                fillMode: Image.PreserveAspectFit
                MouseArea {
                    anchors.fill: parent
                    onClicked: rightLoader.source = page
                }
            }

            Text {
                id: delegateText
                text: model.name; font.pixelSize: 24
                color: "white"
            }


        }
    }


    PathView {

        anchors.centerIn: parent

        height: parent.height * 0.3 // O menu ocupa 30% da altura do pai
        width: parent.width * 0.3

        model: nameModel
        delegate: nameDelegate
        focus: true


        path: Path {
            // Front
            startX: 150; startY: 100
            PathAttribute { name: "opacity"; value: 1.0 }
            PathAttribute { name: "scale"; value: 1.0 }
            PathAttribute { name: "z"; value: 0 }

            // Left
            PathCubic { x: 50; y: 50; control1X: 100; control1Y: 100
                control2X: 50; control2Y: 75 }
            PathAttribute { name: "opacity"; value: 0.5 }
            PathAttribute { name: "scale"; value: 0.5 }
            PathAttribute { name: "z"; value: -1 }

            // Top
            PathCubic { x: 150; y: 20; control1X: 50; control1Y: 35
                control2X: 100; control2Y: 20 }
            PathAttribute { name: "opacity"; value: 0.25 }
            PathAttribute { name: "scale"; value: 0.25 }
            PathAttribute { name: "z"; value: -2 }

            // Right
            PathCubic { x: 250; y: 50; control1X: 200; control1Y: 20
                control2X: 250; control2Y: 35 }
            PathAttribute { name: "opacity"; value: 0.5 }
            PathAttribute { name: "scale"; value: 0.5 }
            PathAttribute { name: "z"; value: -1 }

            PathCubic { x: 150; y: 100; control1X: 250; control1Y: 75
                control2X: 200; control2Y: 100 }
        }

        highlight: Rectangle {
            radius: 4
            width: 2; height: 2
            color: "lightblue"
        }

        Keys.enabled: true
        Keys.onLeftPressed: decrementCurrentIndex()
        Keys.onRightPressed: incrementCurrentIndex()
    }
}
