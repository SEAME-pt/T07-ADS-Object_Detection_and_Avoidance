import QtQuick
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects

Rectangle {
    id: bottomBar
    width: parent.width
    height: parent.height * 0.25
    color: "#2F2F2F"
    opacity: 0.7
    radius: 8
    anchors.bottom: parent.bottom
    layer.enabled: true

    property bool showIcons: false

        SignsLeftCenterPanel {
            id: signsLeftBottomBar

        }

        Rectangle {
            id: hoursBottombar
            implicitWidth: hoursColumn.implicitWidth
            implicitHeight: hoursColumn.implicitHeight
            color: "transparent"
            anchors {
                horizontalCenter: parent.horizontalCenter
            }

            Column {
                id: hoursColumn
                spacing: 5

                Text {
                    text: "12:27 pm"
                    font.pixelSize: 20
                    color: "lightgray"
                }

                Rectangle {
                    id: btnBack


                    width: parent.width
                    height: 50
                    // color: mouseAreaBack.containsMouse ? "#3A3A3A" : "transparent"
                    color: "transparent"
                    radius: 5

                    Row {
                        anchors.centerIn: parent
                        spacing: 10

                        Image {
                            source: "../assets/stop.svg"
                            width: 17
                            height: 17
                            anchors.verticalCenter: parent.verticalCenter
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            mipmap: true
                            opacity: 0.5
                        }

                        Text {
                            text: "Stop"
                            color: "white"
                            font.pixelSize: 17
                            font.weight: Font.Medium
                            anchors.verticalCenter: parent.verticalCenter
                            opacity: 0.5
                        }
                    }

                    MouseArea {
                        id: mouseAreaBack
                        anchors.fill: parent
                        // hoverEnabled: true
                        property bool _stopNav: false
                        onClicked: {
                            console.log("Stop clicked");
                            _stopNav = !_stopNav;
                            centerPanel.stopNav(_stopNav);
                        }
                    }
                }
            }
        }

        MediaBottomBar {
            id:mediaBottomBar
        }
    }
