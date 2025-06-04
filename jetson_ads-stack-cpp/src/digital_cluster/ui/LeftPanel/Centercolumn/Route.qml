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
    property bool isPressingSpace: false                       // Estado da tecla de espaço

    // Propriedades para a animação da linha tracejada
    property real offset: 0                                    // Deslocamento animado da linha tracejada

    // Propriedades para as linhas laterais e velocidade
    property int lateralLineWidth: 3                           // Espessura das linhas laterais
    property real maxSpeed: 100                                // Velocidade máxima
    property real smoothingFactor: 0.5                         // Fator de suavização para transições
    property real simulatedSpeed: 0                            // Velocidade simulada do carro
    property bool isSimulatingRunning: false                   // Controle da simulação de velocidade

    // Propriedades para controle de temporização das linhas laterais
    property bool timeElapsed: false                           // Estado de ativação da linha esquerda
    property bool timeElapsedRight: false                      // Estado de ativação da linha direita
    property int startDelay: 150                               // Delay inicial para ativação (ms)

    // Coordenadas para a estrada central
    property int leftMarginUpRight: width * 0.40               // Margem superior direita
    property int rightMarginUpLeft: width * 0.60               // Margem superior esquerda
    property int leftMarginDownRight: width * 0.30             // Margem inferior direita
    property int rightMarginDownLeft: width * 0.70             // Margem inferior esquerda

    Timer {
        interval: 16
        repeat: true
        running: true
        onTriggered: {
            var speedValue = Number(systemHandler.speed);
            var targetProgress = Math.min(1, Math.max(0, speedValue / maxSpeed));
            accelerationProgress += (targetProgress - accelerationProgress) * smoothingFactor;
            //console.log("Velocidade:", speedValue, "Progresso:", accelerationProgress);
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
            var baseWidth = width * 0.075                      // Largura da base
            var topWidth = baseWidth * 0.6                   // Largura do topo
            var barX = leftMarginDownRight - baseWidth       // Posição X da base
            var barY = height / 3                            // Posição Y do topo

            var baseLeftX = barX
            var baseRightX = barX + baseWidth
            var topLeftX = leftMarginUpRight - topWidth
            var topRightX = leftMarginUpRight

            // Preenchimento do trapézio esquerdo com gradiente
            var fillHeight = trapezoidHeight * accelerationProgress
            var fillBaseY = height-1
            var fillTopY = fillBaseY - fillHeight
            var fillTopLeftX = topLeftX + ((baseLeftX - topLeftX) * (1 - accelerationProgress))
            var fillTopRightX = topRightX + ((baseRightX - topRightX) * (1 - accelerationProgress))

            var gradientLeftTrapezoid = ctx.createLinearGradient(barX + baseWidth / 2, fillTopY, barX + baseWidth / 2, fillBaseY)
            gradientLeftTrapezoid.addColorStop(0, "transparent") // Preto no topo
            gradientLeftTrapezoid.addColorStop(1, accelerationColor) // Cor atual na base
            ctx.fillStyle = gradientLeftTrapezoid
            ctx.beginPath()
            ctx.moveTo(baseLeftX, fillBaseY)
            ctx.lineTo(baseRightX, fillBaseY)
            ctx.lineTo(fillTopRightX, fillTopY)
            ctx.lineTo(fillTopLeftX, fillTopY)
            ctx.closePath()
            ctx.fill()
            ctx.restore()

            // Desenha o trapézio direito (aceleração)
            ctx.save()
            var invertedTrapezoidHeight = height * 0.67
            var invertedBaseWidth = width * 0.075
            var invertedTopWidth = invertedBaseWidth * 0.6
            var invertedBarX = rightMarginDownLeft
            var invertedBarY = height / 3

            var invertedBaseLeftX = invertedBarX
            var invertedBaseRightX = invertedBarX + invertedBaseWidth
            var invertedTopLeftX = rightMarginUpLeft
            var invertedTopRightX = rightMarginUpLeft + invertedTopWidth

            // Preenchimento do trapézio direito com gradiente
            var invertedFillHeight = invertedTrapezoidHeight * accelerationProgress
            var invertedFillBaseY = height-1
            var invertedFillTopY = invertedFillBaseY - invertedFillHeight
            var invertedFillTopLeftX = invertedTopLeftX + ((invertedBaseLeftX - invertedTopLeftX) * (1 - accelerationProgress))
            var invertedFillTopRightX = invertedTopRightX + ((invertedBaseRightX - invertedTopRightX) * (1 - accelerationProgress))

            var gradientRightTrapezoid = ctx.createLinearGradient(invertedBarX + invertedBaseWidth / 2, invertedFillTopY, invertedBarX + invertedBaseWidth / 2, invertedFillBaseY)
            gradientRightTrapezoid.addColorStop(0, "transparent") // Preto no topo
            gradientRightTrapezoid.addColorStop(1, accelerationColor) // Cor atual na base
            ctx.fillStyle = gradientRightTrapezoid
            ctx.beginPath()
            ctx.moveTo(invertedBaseLeftX, invertedFillBaseY)
            ctx.lineTo(invertedBaseRightX, invertedFillBaseY)
            ctx.lineTo(invertedFillTopRightX, invertedFillTopY)
            ctx.lineTo(invertedFillTopLeftX, invertedFillTopY)
            ctx.closePath()
            ctx.fill()
            ctx.restore()

            // Desenha a estrada central (trapézio com gradiente)
            ctx.save()
            var gradientRoad = ctx.createLinearGradient(width * 0.5, height / 3, width * 0.5, height)
            gradientRoad.addColorStop(0, "#262626")          // Preto no topo
            gradientRoad.addColorStop(1, "#000000")          // Cor atual na base
            ctx.fillStyle = gradientRoad
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height-1)
            ctx.lineTo(rightMarginDownLeft, height-1)
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
