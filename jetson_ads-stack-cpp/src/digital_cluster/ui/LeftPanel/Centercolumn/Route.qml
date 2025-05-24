import QtQuick 2.15

Rectangle {
    id: route
    anchors.fill: parent
    color: "transparent" // Fundo transparente para o componente

    // Propriedades para detectar se as linhas laterais foram acionadas
    property bool laneRight: systemHandler.lineRight === "true" // Estado da linha direita
    property bool laneLeft: systemHandler.lineLeft === "true"   // Estado da linha esquerda
    property string routeSelectedMode                           // Modo selecionado para a rota

    // Propriedades para as barras de aceleração (trapézios)
    property real accelerationProgress: 0                       // Progresso da barra (0 a 1)
    property color accelerationColor: {                         // Cor da barra com base no modo
        switch (mainWindow.selectedMode) {
            case "Normal": return "#0000FF" // Azul
            case "Eco": return "#00FF00"    // Verde
            case "Sport": return "#FF0000"  // Vermelho
            default: return "#0000FF"       // Padrão (azul)
        }
    }
    property color trapezoidBorderColor: "#2E2E2E"             // Cor do contorno da barra
    property bool isPressingSpace: false                       // Estado da tecla de espaço

    // Propriedades para a animação da linha tracejada
    property real offset: 0                                    // Deslocamento animado da linha tracejada

    // Propriedades para as linhas laterais e velocidade
    property int lateralLineWidth: 8                           // Espessura das linhas laterais
    property real maxSpeed: 100                                // Velocidade máxima
    property real smoothingFactor: 0.5                         // Fator de suavização para transições
    property real simulatedSpeed: 0                            // Velocidade simulada do carro
    property bool isSimulatingRunning: false                   // Controle da simulação de velocidade

    // Propriedades para controle de temporização das linhas laterais
    property bool timeElapsed: false                           // Estado de ativação da linha esquerda
    property bool timeElapsedRight: false                      // Estado de ativação da linha direita
    property int startDelay: 150                               // Delay inicial para ativação (ms)

    // Coordenadas para a estrada central
    property int leftMarginUpRight: width * 0.45               // Margem superior direita
    property int rightMarginUpLeft: width * 0.55               // Margem superior esquerda
    property int leftMarginDownRight: width * 0.30             // Margem inferior direita
    property int rightMarginDownLeft: width * 0.70             // Margem inferior esquerda

    // Temporizador para simular variação da velocidade
    // Timer {
    //     id: speedSimulationTimer
    //     interval: 1                                           // Intervalo de atualização (ms)
    //     repeat: true                                          // Repetição contínua
    //     running: isSimulatingRunning                          // Ativo quando simulação está em execução
    //     property int phase: 0                                 // Fase atual da simulação
    //     property real targetSpeed: 0                          // Velocidade alvo
    //     property int holdCounter: 0                           // Contador para pausas

    //     onTriggered: {
    //         if (phase === 0) { // Fase 0: Acelera até 60
    //             targetSpeed = 60
    //             simulatedSpeed += 1
    //             if (simulatedSpeed >= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 1
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 1) { // Fase 1: Desacelera até 40
    //             targetSpeed = 40
    //             simulatedSpeed -= 1
    //             if (simulatedSpeed <= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 2
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 2) { // Fase 2: Acelera até 80
    //             targetSpeed = 80
    //             simulatedSpeed += 1
    //             if (simulatedSpeed >= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 3
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 3) { // Fase 3: Mantém 80 por 1 segundo
    //             holdCounter++
    //             if (holdCounter >= 10) { // 10 * 100ms = 1s
    //                 phase = 4
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 4) { // Fase 4: Desacelera até 60
    //             targetSpeed = 60
    //             simulatedSpeed -= 1
    //             if (simulatedSpeed <= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 5
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 5) { // Fase 5: Desacelera até 30
    //             targetSpeed = 30
    //             simulatedSpeed -= 1
    //             if (simulatedSpeed <= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 6
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 6) { // Fase 6: Acelera até 40
    //             targetSpeed = 40
    //             simulatedSpeed += 1
    //             if (simulatedSpeed >= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 7
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 7) { // Fase 7: Desacelera até 0
    //             targetSpeed = 0
    //             simulatedSpeed -= 1
    //             if (simulatedSpeed <= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 8
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 8) { // Fase 8: Acelera até 50
    //             targetSpeed = 50
    //             simulatedSpeed += 1
    //             if (simulatedSpeed >= targetSpeed) {
    //                 simulatedSpeed = targetSpeed
    //                 phase = 9
    //                 holdCounter = 0
    //             }
    //         } else if (phase === 9) { // Fase 9: Mantém 50 por 1 segundo
    //             holdCounter++
    //             if (holdCounter >= 10) { // 10 * 100ms = 1s
    //                 phase = 0
    //                 holdCounter = 0
    //             }
    //         }
    //     }
    // }

    Timer {
        interval: 16
        repeat: true
        running: true
        onTriggered: {
            var speedValue = Number(systemHandler.speed);
            var targetProgress = Math.min(1, Math.max(0, speedValue / maxSpeed));
            accelerationProgress += (targetProgress - accelerationProgress) * smoothingFactor;
            console.log("Velocidade:", speedValue, "Progresso:", accelerationProgress);
            routeCanvas.requestPaint();
        }
    }

    // Temporizador para suavizar o progresso da barra de aceleração
    Timer {
        interval: 16                                          // Aproximadamente 60 FPS
        repeat: true                                          // Repetição contínua
        running: true                                         // Sempre ativo
        onTriggered: {
            var targetProgress = isSimulatingRunning ? Math.min(1, Math.max(0, simulatedSpeed / maxSpeed)) : 0
            accelerationProgress += (targetProgress - accelerationProgress) * smoothingFactor
            routeCanvas.requestPaint()                        // Redesenha o canvas
        }
    }

    // Temporizador para animar a linha tracejada
    Timer {
        interval: 16                                          // Aproximadamente 60 FPS
        running: Math.round(Number(systemHandler.speed)) > 0  // Ativo quando há velocidade
        repeat: true                                          // Repetição contínua
        onTriggered: {
            offset += 2
            if (offset >= route.totalHeight) {
                offset = 0                                   // Reinicia o deslocamento
            }
            routeCanvas.requestPaint()                       // Redesenha o canvas
        }
    }

    // Temporizador para delay da linha esquerda
    Timer {
        id: delayTimerLeft
        interval: startDelay                                  // Delay inicial (ms)
        running: false                                       // Inicia quando ativado
        repeat: false                                         // Executa uma vez
        onTriggered: {
            changeColorTimerLeft.start()                     // Inicia o timer de mudança de cor
        }
    }

    // Temporizador para mudar a cor da linha esquerda
    Timer {
        id: changeColorTimerLeft
        interval: 300                                         // Tempo de ativação (ms)
        running: false                                        // Inicia quando ativado
        repeat: false                                         // Executa uma vez
        onTriggered: {
            timeElapsed = true                                   // Ativa a mudança de cor
            routeCanvas.requestPaint()                           // Redesenha o canvas
        }
    }

    // Observa mudanças na linha esquerda
    onLaneLeftChanged: {
        if (laneLeft) {
            delayTimerLeft.start()                           // Inicia o delay quando ativado
        } else {
            delayTimerLeft.stop()                            // Para o delay
            changeColorTimerLeft.stop()                      // Para a mudança de cor
            timeElapsed = false                              // Reseta o estado
        }
    }

    // Temporizador para delay da linha direita
    Timer {
        id: delayTimerRight
        interval: startDelay                                  // Delay inicial (ms)
        running: false                                        // Inicia quando ativado
        repeat: false                                         // Executa uma vez
        onTriggered: {
            changeColorTimerRight.start()                    // Inicia o timer de mudança de cor
        }
    }

    // Temporizador para mudar a cor da linha direita
    Timer {
        id: changeColorTimerRight
        interval: 300                                         // Tempo de ativação (ms)
        running: false                                        // Inicia quando ativado
        repeat: false                                         // Executa uma vez
        onTriggered: {
            timeElapsedRight = true                          // Ativa a mudança de cor
            routeCanvas.requestPaint()                           // Redesenha o canvas
        }
    }

    // Observa mudanças na linha direita
    onLaneRightChanged: {
        if (laneRight) {
            delayTimerRight.start()                          // Inicia o delay quando ativado
        } else {
            delayTimerRight.stop()                           // Para o delay
            changeColorTimerRight.stop()                     // Para a mudança de cor
            timeElapsedRight = false                         // Reseta o estado
        }
    }

    // Manipula pressionamento de teclas
    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_Space) {
            isPressingSpace = true                           // Ativa ao pressionar espaço
        } else if (event.key === Qt.Key_Escape) {
            Qt.quit()                                        // Sai ao pressionar escape
        }
    }

    // Manipula liberação de teclas
    Keys.onReleased: function(event) {
        if (event.key === Qt.Key_Space) {
            isPressingSpace = false                          // Desativa ao liberar espaço
        }
    }

    // Canvas para desenhar a estrada e as barras
    Canvas {
        id: routeCanvas
        anchors.fill: parent
        renderTarget: Canvas.FramebufferObject            // Otimiza para animações suaves

        onPaint: {
            var ctx = routeCanvas.getContext("2d")
            ctx.clearRect(0, 0, width, height)               // Limpa o canvas

            // Desenha o trapézio esquerdo (aceleração)
            ctx.save()
            var trapezoidHeight = height * 0.67              // Altura do trapézio
            var baseWidth = width * 0.1                      // Largura da base
            var topWidth = baseWidth * 0.6                   // Largura do topo
            var barX = leftMarginDownRight - baseWidth       // Posição X da base
            var barY = height / 3                            // Posição Y do topo

            var baseLeftX = barX
            var baseRightX = barX + baseWidth
            var topLeftX = leftMarginUpRight - topWidth
            var topRightX = leftMarginUpRight

            // Contorno do trapézio esquerdo
            ctx.strokeStyle = trapezoidBorderColor
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(baseLeftX, height)
            ctx.lineTo(baseRightX, height)
            ctx.lineTo(topRightX, barY)
            ctx.lineTo(topLeftX, barY)
            ctx.closePath()
            ctx.stroke()

            // Preenchimento do trapézio esquerdo
            var fillHeight = trapezoidHeight * accelerationProgress
            var fillBaseY = height
            var fillTopY = fillBaseY - fillHeight
            var fillTopLeftX = topLeftX + ((baseLeftX - topLeftX) * (1 - accelerationProgress))
            var fillTopRightX = topRightX + ((baseRightX - topRightX) * (1 - accelerationProgress))

            ctx.fillStyle = accelerationColor
            ctx.beginPath()
            ctx.moveTo(baseLeftX, fillBaseY)
            ctx.lineTo(baseRightX, fillBaseY)
            ctx.lineTo(fillTopRightX, fillTopY)
            ctx.lineTo(fillTopLeftX, fillTopY)
            ctx.closePath()
            ctx.fill()
            ctx.restore()

            // Desenha o trapézio direito (inverso)
            ctx.save()
            var invertedTrapezoidHeight = height * 0.67
            var invertedBaseWidth = width * 0.1
            var invertedTopWidth = invertedBaseWidth * 0.6
            var invertedBarX = rightMarginDownLeft
            var invertedBarY = height / 3

            var invertedBaseLeftX = invertedBarX
            var invertedBaseRightX = invertedBarX + invertedBaseWidth
            var invertedTopLeftX = rightMarginUpLeft
            var invertedTopRightX = rightMarginUpLeft + invertedTopWidth

            // Contorno do trapézio direito
            ctx.strokeStyle = trapezoidBorderColor
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(invertedBaseLeftX, height)
            ctx.lineTo(invertedBaseRightX, height)
            ctx.lineTo(invertedTopRightX, invertedBarY)
            ctx.lineTo(invertedTopLeftX, invertedBarY)
            ctx.closePath()
            ctx.stroke()

            // Preenchimento do trapézio direito
            var invertedFillHeight = invertedTrapezoidHeight * accelerationProgress
            var invertedFillBaseY = height
            var invertedFillTopY = invertedFillBaseY - invertedFillHeight
            var invertedFillTopLeftX = invertedTopLeftX + ((invertedBaseLeftX - invertedTopLeftX) * (1 - accelerationProgress))
            var invertedFillTopRightX = invertedTopRightX + ((invertedBaseRightX - invertedTopRightX) * (1 - accelerationProgress))

            ctx.fillStyle = accelerationColor
            ctx.beginPath()
            ctx.moveTo(invertedBaseLeftX, invertedFillBaseY)
            ctx.lineTo(invertedBaseRightX, invertedFillBaseY)
            ctx.lineTo(invertedFillTopRightX, invertedFillTopY)
            ctx.lineTo(invertedFillTopLeftX, invertedFillTopY)
            ctx.closePath()
            ctx.fill()
            ctx.restore()

            // Desenha a estrada central (trapézio preto)
            ctx.save()
            ctx.fillStyle = "#2F2F2F"
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height)
            ctx.lineTo(rightMarginDownLeft, height)
            ctx.lineTo(rightMarginUpLeft, height / 3)
            ctx.lineTo(leftMarginUpRight, height / 3)
            ctx.closePath()
            ctx.fill()
            ctx.restore()

            // Desenha a linha lateral esquerda
            ctx.save()
            var gradientLeft = ctx.createLinearGradient(width * 0.30, height / 3, width * 0.30, height)
            gradientLeft.addColorStop(0, "#2E2E2E")     // Tom escuro no topo
            gradientLeft.addColorStop(1, "#8A8A8A")     // Tom claro na base
            ctx.strokeStyle = (laneLeft && timeElapsed) ? "#4A90E2" : gradientLeft
            ctx.lineWidth = lateralLineWidth
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height)
            ctx.lineTo(leftMarginUpRight, height / 3)
            ctx.stroke()
            ctx.restore()

            // Desenha a linha lateral direita
            ctx.save()
            var gradientRight = ctx.createLinearGradient(width * 0.7, height / 3, width * 0.7, height)
            gradientRight.addColorStop(0, "#2E2E2E")    // Tom escuro no topo
            gradientRight.addColorStop(1, "#8A8A8A")    // Tom claro na base
            ctx.strokeStyle = (laneRight && timeElapsedRight) ? "#4A90E2" : gradientRight
            ctx.lineWidth = lateralLineWidth
            ctx.beginPath()
            ctx.moveTo(rightMarginDownLeft, height)
            ctx.lineTo(rightMarginUpLeft, height / 3)
            ctx.stroke()
            ctx.restore()
        }
    }

    // Ativa o foco ao clicar no componente
    MouseArea {
        anchors.fill: parent
        onClicked: route.focus = true
    }
}
