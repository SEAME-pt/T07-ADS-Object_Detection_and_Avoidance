import QtQuick 2.15

Rectangle {

    id: left
    anchors {
        top: parent.top
        left: parent.left
        bottom: parent.bottom
        margins: 25
    }

    signal gearSelected(string gear)
    width: parent.width * 0.1
    // height: parent.height
    color: "transparent"


    Column {

        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 10

        //Speed Limit
        Rectangle {
            id: speedLimit

            anchors {
                horizontalCenter: parent.horizontalCenter
            }
            width: 60
            height: 60
            color: "white"
            radius: 50
            border.color: "red"
            border.width: 4
            Text {
                text: "70"
                font.pixelSize: 30
                anchors.centerIn: parent
            }
        }

        Rectangle {
            height: 20 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

        Signs {
            id: signs

            // atenção só fornece os estados quando corre no inicio, não é dinâmico
            // Component.onCompleted: {
            //             // Converte a string recebida para um valor booleano
            //             // var state = system.headLights === "true";
            //             // let stateHeadLights = false; //true liga os headLights
            //             let stateHeadLights = systemHandler.headLights === "true" ? true : false;
            //             toggleLight("headLights", stateHeadLights); // Liga ou desliga o ícone
            //         }

        }

        Rectangle {
            height: 10 // Espaçamento específico para este ponto
            width: parent.width
            color: "transparent"
        }

    }

    onGearSelected: function (gear) {
        // console.log("gear in left", gear);
        signs.gearSelected(gear);
    }

}
