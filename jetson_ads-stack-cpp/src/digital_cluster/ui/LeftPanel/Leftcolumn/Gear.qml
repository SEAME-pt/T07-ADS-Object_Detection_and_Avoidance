import QtQuick 2.15

Row {
    id: rowGears
    anchors.horizontalCenter: parent.horizontalCenter
    spacing: 5
    // anchors.centerIn: parent

    signal gearSelected(string gear)
    property string selectedGear: "P" // Armazena o item selecionado no inicio

    Repeater {
        model: ["P", "R", "N", "D"]
        delegate: Rectangle {
            width: 20
            height: 20
            // color: gear === "D" ? "green" : "white"
            opacity: modelData === rowGears.selectedGear ? 1 : 0.5
            color: "transparent"
            // radius: 5

            Text {
                text: modelData
                color: "white"
                font.pixelSize: 15
                anchors.centerIn: parent
            }

            MouseArea {
                anchors.fill: parent
                onClicked: {
                    rowGears.gearSelected(modelData)
                    rowGears.selectedGear = modelData
                    console.log("gear on gear.qml: ", modelData)

                }
            }
        }
    }
}

