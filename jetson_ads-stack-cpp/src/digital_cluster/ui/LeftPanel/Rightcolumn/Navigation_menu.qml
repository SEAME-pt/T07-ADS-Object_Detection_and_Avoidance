// import QtQuick 2.15
// import QtQuick.Controls 2.15

// Rectangle {
//     id: navigationMenu
//     color: "transparent"
//     radius: 10

//     Column {
//         anchors.fill: parent
//         anchors.margins: 20
//         spacing: 20

//         // Container do campo de texto
//         Rectangle {
//             id: inputContainer
//             width: parent.width
//             height: 50
//             color: "#2D2D2D"
//             radius: 5

//             Image {
//                 source: "../../assets/search.svg"
//                 width: 28
//                 height: 28
//                 anchors.verticalCenter: parent.verticalCenter
//                 fillMode: Image.PreserveAspectFit
//                 smooth: true
//                 mipmap: true
//             }

//             TextInput {
//                 id: locationInput
//                 anchors.fill: parent
//                 anchors.margins: 8
//                 verticalAlignment: Text.AlignVCenter
//                 color: "white"
//                 font.pixelSize: 18

//             }
//         }

//         // Botão de navegação
//         Rectangle {
//             id: btnStart
//             width: parent.width
//             height: 50
//             color: "transparent"
//             radius: 5

//             Text {
//                 text: "Start"
//                 anchors.centerIn: parent
//                 color: "white"
//                 font.pixelSize: 18
//                 font.weight: Font.Medium
//             }

//             MouseArea {
//                 anchors.fill: parent
//                 onClicked: {
//                     // Implementar lógica de navegação
//                     console.log("Start Navigation clicked");
//                 }
//             }
//         }

//         // Botão de voltar
//         Rectangle {
//             id: btnBack
//             width: parent.width
//             height: 50
//             color: "transparent"

//             Text {
//                 text: "back"
//                 anchors.centerIn: parent
//                 color: "white"
//                 font.pixelSize: 18
//                 font.weight: Font.Medium
//             }

//             MouseArea {
//                 anchors.fill: parent
//                 onClicked: {
//                     // Implementar lógica para voltar
//                     console.log("Back clicked");
//                     stackviewRightColumn.pop()
//                 }
//             }
//         }

//     }
// }


import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: navigationMenu
    color: "#1E1E1E"
    radius: 10

    // signal startNavigation(bool start)

    Column {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20

        // Container do campo de texto
        Rectangle {
            id: inputContainer
            width: parent.width
            height: 50
            color: "white" //"#2D2D2D"
            radius: 8
            clip: true

            Image {
                id: searchIcon
                source: "../../assets/search_black.svg"
                // color: "#2D2D2D"
                width: 18
                height: 18
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                anchors.leftMargin: 10
                fillMode: Image.PreserveAspectFit
                smooth: true
                mipmap: true
            }

            TextInput {
                    id: locationInput
                    anchors.left: searchIcon.right  // O texto começa à direita da imagem
                    anchors.right: parent.right    // O texto vai até o fim do container
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    anchors.leftMargin: 5          // Espaço entre a imagem e o texto
                    verticalAlignment: Text.AlignVCenter
                    color: "black"
                    font.pixelSize: 18
                    clip: true
                }


        }

        // Botão de navegação
        Rectangle {
            id: btnStart
            width: parent.width
            height: 50
            // color: mouseAreaStart.containsMouse ? "#3A3A3A" : "transparent"
            color: "transparent"
            radius: 5

            Row {
                anchors.centerIn: parent
                spacing: 10

                Image {
                    source: "../../assets/race_flag.svg"
                    width: 24
                    height: 24
                    anchors.verticalCenter: parent.verticalCenter
                }

                Text {
                    text: "Start"
                    color: "white"
                    font.pixelSize: 18
                    font.weight: Font.Medium
                }
            }

            MouseArea {
                id: mouseAreaStart
                anchors.fill: parent
                // hoverEnabled: true
                property bool _startNav: false
                onClicked: {
                    console.log("Start Navigation clicked");
                    _startNav = !_startNav;
                    // navigationMenu.startNavigation(_startNav);
                    rightColumn.startNavigation(_startNav);
                    stackviewRightColumn.pop();
                }
            }
        }



    }

    // Botão de voltar
    Rectangle {
        id: btnBack
        anchors {
            bottom: parent.bottom
        }

        width: parent.width
        height: 50
        // color: mouseAreaBack.containsMouse ? "#3A3A3A" : "transparent"
        color: "transparent"
        radius: 5

        Row {
            anchors.centerIn: parent
            spacing: 10

            Image {
                source: "../../assets/arrow_back.svg"
                width: 18
                height: 18
                anchors.verticalCenter: parent.verticalCenter
            }

            Text {
                text: "Back"
                color: "white"
                font.pixelSize: 18
                font.weight: Font.Medium
            }
        }

        MouseArea {
            id: mouseAreaBack
            anchors.fill: parent
            // hoverEnabled: true
            onClicked: {
                console.log("Back clicked");
                stackviewRightColumn.pop()
            }
        }
    }
}


