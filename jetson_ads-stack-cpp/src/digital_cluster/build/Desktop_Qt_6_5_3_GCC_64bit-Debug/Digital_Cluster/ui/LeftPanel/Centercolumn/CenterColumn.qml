import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: centerColumn

    // width: parent.width * 0.5 // 60% da largura do container
    width: parent.width - leftColumn.width - rightColumn.width - 20
    color: "#3A3A3A"
    // color: "blue"
    radius: 8
    anchors {
        top: parent.top
        bottom: parent.bottom
    }

    signal gearSelected(string gear)

    property bool turnLightLeftOn: systemHandler.turnLightLeft === "true" ? true : false
    property bool turnLightRightOn: systemHandler.turnLightRight === "true" ? true : false
    property bool emergency: systemHandler.emergencyLights === "true" ? true : false

    property string centerColumnSelectedMode

    // Component.onCompleted: {
    //     console.log("Modo na centerColumn: ", centerColumnSelectedMode);
    // }


    Text {
        id: speed

        anchors {
            top: parent.top
            topMargin: 10
            horizontalCenter: parent.horizontalCenter
        }

        //text: "42"
        // text: systemHandler.speed
        text: Math.round(Number(systemHandler.speed)) // Exibe a velocidade atual sem casas decimais
        color: "white"
        font.pixelSize: 80

    }


    Route {
        id: route
        routeSelectedMode: centerColumn.centerColumnSelectedMode
    }


    Text {
        id: speedUnits
        anchors {
            top: speed.bottom
            horizontalCenter: parent.horizontalCenter
        }
        text: "m/h"
        color: "white"
        font.pixelSize: 16
    }


    Left {
        id: left
    }


    Right {
        id: right
    }


    // Image {
    //        id: carRender
    //        source: "../../assets/car.png"
    //        anchors.bottom: parent.bottom
    //        anchors.bottomMargin: 10
    //        anchors.horizontalCenter: parent.horizontalCenter
    //        width: parent.width * .25
    //        fillMode: Image.PreserveAspectFit
    // }

    property bool isChangingLanes: false // activa a animaçao das linhas

    Image {
        id: carRender
        source: "../../assets/car.png"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.15
        fillMode: Image.PreserveAspectFit

        // Deslocação horizontal do carro
        property real targetOffset: {
            if (route.laneLeft) return -parent.width * 0.08;  // Move para a esquerda
            if (route.laneRight) return parent.width * 0.08;  // Move para a direita
            return 0;                                         // Volta ao centro
        }

        // Inclinação do carro (em graus)
        property real targetRotation: {
            if (route.laneLeft) return -10;   // Inclina a frente para a esquerda (negativo)
            if (route.laneRight) return 10;   // Inclina a frente para a direita (positivo)
            return 0;                         // Sem inclinação (centro)
        }

        // Aplica o movimento horizontal com uma animação suave
        anchors.horizontalCenterOffset: targetOffset

        // Aplica a rotação com uma animação suave
        rotation: targetRotation

        // Animação para o movimento horizontal
        Behavior on anchors.horizontalCenterOffset {
            NumberAnimation {
                duration: 300  // Duração da animação em milissegundos (300ms para um movimento suave)
                easing.type: Easing.InOutQuad  // Tipo de easing para um movimento natural
            }
        }

        // Animação para a rotação
        Behavior on rotation {
            NumberAnimation {
                duration: 300  // Mesma duração do movimento horizontal para sincronia
                easing.type: Easing.InOutQuad  // Mesmo tipo de easing para consistência
            }
        }
    }


    // Timer para simular a mudança de laneLeft e laneRight
    Timer {
        id: laneChangeTimer
        interval: 2000  // Muda a cada 2 segundos (ajustável)
        repeat: true
        running: isChangingLanes
        property int phase: 0  // Controla a fase da simulação

        onTriggered: {
            // Garantir que ambas as propriedades sejam false antes de mudar
            route.laneLeft = false;
            route.laneRight = false;

            if (phase === 0) { // Fase 0: Carro na faixa esquerda (laneLeft: true)
                route.laneLeft = true;
                phase = 1;
            } else if (phase === 1) { // Fase 1: Carro no centro (ambas false)
                // Já definimos ambas como false acima
                phase = 2;
            } else if (phase === 2) { // Fase 2: Carro na faixa direita (laneRight: true)
                route.laneRight = true;
                phase = 3;
            } else if (phase === 3) { // Fase 3: Carro no centro (ambas false)
                // Já definimos ambas como false acima
                phase = 0; // Reinicia o ciclo
            }
        }
    }


    Rectangle {
        id: turnSignalLeft
        width: 50
        height: 50
        color: "transparent"
        clip: true
        visible: false
        anchors {
            top: parent.top
            left: parent.left
            leftMargin: parent.width * 0.17
            topMargin: parent.height * 0.1
        }

        Image {
            source: "../../assets/turn_light_02.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            smooth: true // Para renderização de alta qualidade
            // para testes
            // MouseArea {
            //     anchors.fill: parent
            //     onClicked: {
            //         console.log("clicked for teste")
            //         blinkTimerLeft.running = false;
            //     }
            // }
        }

        Timer {
                id: blinkTimerLeft
                interval: 500 // 500ms, ou seja, pisca duas vezes por segundo
                running: turnLightLeftOn || emergency // Inicia automaticamente
                repeat: true // Continua a piscar
                onTriggered: {
                    turnSignalLeft.visible = !turnSignalLeft.visible;
                }
                onRunningChanged: {
                   if (!running) {
                       turnSignalLeft.visible = false;
                   }
                }
        }
    }


    Rectangle {
        id: turnSignalRight
        width: 50
        height: 50
        color: "transparent"
        rotation: 180
        visible: false
        clip: true
        anchors {
            top: parent.top
            right: parent.right
            rightMargin: parent.width * 0.17
            topMargin: parent.height * 0.1
        }

        Image {
            source: "../../assets/turn_light_02.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            smooth: true // Para renderização de alta qualidade
        }

        Timer {
                id: blinkTimerRight
                interval: 500 // 500ms, ou seja, pisca duas vezes por segundo
                running: turnLightRightOn || emergency // Inicia automaticamente
                repeat: true // Continua piscando
                onTriggered: {
                    turnSignalRight.visible = !turnSignalRight.visible
                }
                onRunningChanged: {
                   if (!running) {
                       turnSignalLeft.visible = false;
                   }
                }
        }
    }

    // teste de verificação - c++ -> QML
    // Component.onCompleted: {
    //     console.log("systemHandler está acessível no QML centerColumn:", systemHandler.speed);
    // }

    onGearSelected: function (gear) {
        left.gearSelected(gear);
        right.gearSelected(gear);
    }

}


// anchors.horizontalCenterOffset é uma propriedade que define um deslocamento (offset) em relação ao ponto de ancoragem definido por anchors.horizontalCenter.
//     Ela é usada para ajustar a posição do elemento a partir do centro horizontal definido por anchors.horizontalCenter, permitindo que você mova o elemento para a esquerda ou para a direita sem alterar o ponto de ancoragem principal.

// Como funciona?

//     Primeiro, você define o ponto de ancoragem com anchors.horizontalCenter (por exemplo, alinhando o centro do elemento ao centro do pai).
//     Depois, você usa anchors.horizontalCenterOffset para adicionar um deslocamento em pixels a partir desse ponto de ancoragem.
//     Um valor positivo de anchors.horizontalCenterOffset move o elemento para a direita; um valor negativo move para a esquerda.

// NumberAnimation é uma das animações mais simples e eficientes em QML. Ele é otimizado para interpolar valores numéricos, o que o torna ideal para animar propriedades como posições (x, y), deslocamentos (anchors.horizontalCenterOffset), tamanhos (width, height), opacidade (opacity), etc.
//     Como você só precisa animar um único valor numérico (o deslocamento horizontal), NumberAnimation é a escolha mais direta.

// Suporte a Behavior:

//     O Behavior em QML é usado para aplicar uma animação automaticamente sempre que a propriedade associada muda. Ele funciona bem com animações como NumberAnimation, porque o Behavior espera uma animação que possa interpolar o valor da propriedade de forma contínua.
//     NumberAnimation é compatível com Behavior, permitindo que o movimento do carro seja animado automaticamente toda vez que targetOffset (e, consequentemente, anchors.horizontalCenterOffset) mudar.

// Controle de Duração e Easing:

//     NumberAnimation oferece propriedades como duration e easing.type, que permitem controlar a velocidade e o estilo da animação:
//         duration: 300 define que a animação leva 300 milissegundos para completar.
//         easing.type: Easing.InOutQuad faz com que o movimento comece e termine de forma suave, com uma aceleração e desaceleração naturais.
//     Essas propriedades são essenciais para criar uma animação que pareça natural, como o movimento do carro para a esquerda ou direita.
