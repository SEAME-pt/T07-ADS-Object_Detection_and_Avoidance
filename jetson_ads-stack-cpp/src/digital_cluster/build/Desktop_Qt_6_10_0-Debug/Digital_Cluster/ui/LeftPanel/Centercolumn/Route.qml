import QtQuick 2.15

Rectangle {
    id: route
    anchors.fill: parent
    color: "transparent"

    //Bool vars para indicar qual a linha calcada
    property bool laneRight: systemHandler.lineRight === "true" // criar o laneright no System
    // property bool laneRight: false
    property bool laneLeft: systemHandler.lineLeft === "true" // criar o laneleft no System
    // property bool laneLeft: false

    // Component.onCompleted: {
    //     console.log("systemHandler.lineLeft inicial:", systemHandler.lineLeft, typeof systemHandler.lineLeft)
    //     console.log("laneLeft inicial:", laneLeft)
    // }

    property string routeSelectedMode

    // Component.onCompleted: {
    //     console.log("Modo na route: ", routeSelectedMode);
    // }

    // Linha tracejada (simulação de estrada)
    property int lineWidth: 8
    property int dashHeight: 50
    property int gapHeight: 40
    property int totalHeight: dashHeight + gapHeight

    // Street Up Corners
    property int leftMarginUpLeft: width * 0.45
    property int leftMarginUpRight: width * 0.475
    property int rightMarginUpLeft: width * 0.525
    property int rightMarginUpRight: width * 0.55

    // Street Down Corners
    property int leftMarginDownLeft: width * 0.2
    property int leftMarginDownRight: width * 0.35
    property int rightMarginDownLeft: width * 0.65
    property int rightMarginDownRight: width * 0.80

    // Propriedades da barra de aceleração
    property real accelerationProgress: 0               // Progresso da barra (0 a 1)
    // property color accelerationColor: "#0000FF"         // Cor do preenchimento interno da barra

    property color accelerationColor: {
        switch (mainWindow.selectedMode) {
            case "Normal": return "#0000FF" // Azul
            case "Eco": return "#00FF00"    // Verde
            case "Sport": return "#FF0000"  // Vermelho
            default: return "#0000FF"       // Padrão (azul)
        }
    }
    property color trapezoidBorderColor: "#2E2E2E"      // Cor do contorno da barra

    property bool isPressingSpace: false

    property real offset: 0                             // Deslocamento animado

    // Nova propriedade para a espessura das linhas laterais
    property int lateralLineWidth: 8                    // Espessura das linhas laterais (ajustável)

    property real maxSpeed: 100                         // Velocidade máxima (ajustável)

    // Nova propriedade para controlar a suavidade da transição
    property real smoothingFactor: 0.5                  // Fator de suavização (0 a 1)


    // Nova propriedade para simular a velocidade do carro - teste só
    property real simulatedSpeed: 0                     // Velocidade simulada (substitui systemHandler.speed)
    property bool isSimulatingRunning: false


    // Temporizador para simular a variação da velocidade do carro
    Timer {
        id: speedSimulationTimer
        interval: 1                                   // A cada 100ms (ajustável)
        repeat: true
        running: isSimulatingRunning                    // Sempre ativo para simular a velocidade
        property int phase: 0                           // Fase da simulação
        property real targetSpeed: 0                    // Velocidade alvo para a fase atual
        property int holdCounter: 0                     // Contador para manter a velocidade constante

        onTriggered: {
            if (phase === 0) { // Fase 0: Sobe para 60
                targetSpeed = 60;
                simulatedSpeed += 1; // Aumenta 1 a cada 100ms
                if (simulatedSpeed >= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 1; // Passa para a próxima fase
                    holdCounter = 0;
                }
            } else if (phase === 1) { // Fase 1: Desce para 40
                targetSpeed = 40;
                simulatedSpeed -= 1; // Diminui 1 a cada 100ms
                if (simulatedSpeed <= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 2; // Passa diretamente para a próxima fase (sem espera)
                    holdCounter = 0;
                }
            } else if (phase === 2) { // Fase 2: Sobe para 80
                targetSpeed = 80;
                simulatedSpeed += 1; // Aumenta 1 a cada 100ms
                if (simulatedSpeed >= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 3; // Passa para a fase de manter
                    holdCounter = 0;
                }
            } else if (phase === 3) { // Fase 3: Mantém 80 por 1 segundo
                holdCounter++;
                if (holdCounter >= 10) { // 10 * 100ms = 1 segundo
                    phase = 4; // Passa para a próxima fase
                    holdCounter = 0;
                }
            } else if (phase === 4) { // Fase 4: Desce para 60
                targetSpeed = 60;
                simulatedSpeed -= 1; // Diminui 1 a cada 100ms
                if (simulatedSpeed <= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 5; // Passa diretamente para a próxima fase (sem espera)
                    holdCounter = 0;
                }
            } else if (phase === 5) { // Fase 5: Desce para 30
                targetSpeed = 30;
                simulatedSpeed -= 1; // Diminui 1 a cada 100ms
                if (simulatedSpeed <= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 6; // Passa diretamente para a próxima fase (sem espera)
                    holdCounter = 0;
                }
            } else if (phase === 6) { // Fase 6: Sobe para 40
                targetSpeed = 40;
                simulatedSpeed += 1; // Aumenta 1 a cada 100ms
                if (simulatedSpeed >= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 7; // Passa diretamente para a próxima fase (sem espera)
                    holdCounter = 0;
                }
            } else if (phase === 7) { // Fase 7: Desce para 0
                targetSpeed = 0;
                simulatedSpeed -= 1; // Diminui 1 a cada 100ms
                if (simulatedSpeed <= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 8; // Passa para a próxima fase
                    holdCounter = 0;
                }
            } else if (phase === 8) { // Fase 8: Sobe para 50
                targetSpeed = 50;
                simulatedSpeed += 1; // Aumenta 1 a cada 100ms
                if (simulatedSpeed >= targetSpeed) {
                    simulatedSpeed = targetSpeed;
                    phase = 9; // Passa para a fase de manter
                    holdCounter = 0;
                }
            } else if (phase === 9) { // Fase 9: Mantém 50 por 1 segundo
                holdCounter++;
                if (holdCounter >= 10) { // 10 * 100ms = 1 segundo
                    phase = 0; // Volta para a primeira fase (reinicia o ciclo)
                    holdCounter = 0;
                }
            }
        }
    }

    // Temporizador para controle suave do progresso baseado na velocidade -teste para ver se funciona
    Timer {
        interval: 16                                    // Aproximadamente 60 FPS
        repeat: true
        running: true                                  // Sempre ativo para monitorar a velocidade
        onTriggered: {
            // Converter systemHandler.speed para número
            // var currentSpeed = Math.round(Number(systemHandler.speed)) || 0 - assim esta nos numeros
            // var currentSpeed = Number(systemHandler.speed) || 0; usar com o system

            var targetProgress = isSimulatingRunning ? Math.min(1, Math.max(0, simulatedSpeed / maxSpeed)) : 0;
            // console.log("current speed: ", simulatedSpeed);

            // Mapear a velocidade atual para o intervalo de 0 a 1
            // let targetProgress = Math.min(1, Math.max(0, currentSpeed / maxSpeed)); -usar com o system

            // Suavizar a transição de accelerationProgress para targetProgress
            accelerationProgress += (targetProgress - accelerationProgress) * smoothingFactor;

            // Redesenhar o Canvas para atualizar as barras
            routeCanvas.requestPaint();
        }
    }

    // Temporizador para controle suave do progresso - pode-se alterar para controlar o systemHandler.speed np if considerando se for >0 e <0
    // Timer {
    //     interval: 16                                    // Aproximadamente 60 FPS
    //     repeat: true
    //     running: isPressingSpace || accelerationProgress > 0
    //     onTriggered: {
    //         if (isPressingSpace) {
    //             accelerationProgress = Math.min(1, accelerationProgress + 0.01);
    //         } else {
    //             accelerationProgress = Math.max(0, accelerationProgress - 0.01);
    //         }
    //         routeCanvas.requestPaint();
    //     }
    // }

    // Temporizador para deslocamento da linha tracejada
    Timer {
        interval: 16                                    // Aproximadamente 60 FPS
        running: Math.round(Number(systemHandler.speed)) > 0
        repeat: true
        onTriggered: {
            offset += 2;
            if (offset >= route.totalHeight) {
                offset = 0;                             // Reinicia o deslocamento
            }
            routeCanvas.requestPaint();
        }
    }


    // Controlar ligar a linha só a partir de qualquer
    // 1. Criar uma variável de controlo (se não existir)
    property bool timeElapsed: false
    property bool timeElapsedRight: false

    property int startDelay: 150 // Delay inicial em milissegundos (1 segundo, por exemplo)

    Timer {
        id: delayTimerLeft
        interval: startDelay
        running: false // Não inicia automaticamente
        repeat: false // Garante que só roda uma vez por ativação
        onTriggered: {
            changeColorTimerLeft.start() // Inicia o timer principal após o delay
        }
    }

    Timer {
        id: changeColorTimerLeft
        interval: 300 // Tempo principal
        running: false // Não inicia automaticamente
        repeat: false // Garante que só roda uma vez
        onTriggered: {
            timeElapsed = true
            routeCanvas.requestPaint()
        }
    }

    // Observa mudanças em laneLeft
    onLaneLeftChanged: {
        // console.log("laneLeft mudou para:", laneLeft)
        if (laneLeft) { // Só inicia se laneLeft mudar para true
            delayTimerLeft.start()
        } else { // faz reset quando laneLeft volta a false, se necessário
            delayTimerLeft.stop()
            changeColorTimerLeft.stop()
            timeElapsed = false // Opcional: reseta o estado
        }
    }


    Timer {
        id: delayTimerRight
        interval: startDelay
        running: false // Não inicia automaticamente
        repeat: false // Garante que só roda uma vez por ativação
        onTriggered: {
            changeColorTimerRight.start() // Inicia o timer principal após o delay
        }
    }

    Timer {
        id: changeColorTimerRight
        interval: 300 // Tempo principal
        running: false // Não inicia automaticamente
        repeat: false // Garante que só roda uma vez
        onTriggered: {
            timeElapsedRight = true
            routeCanvas.requestPaint()
        }
    }

    // Observa mudanças em laneLeft
    onLaneRightChanged: {
        if (laneRight) { // Só inicia se laneLeft mudar para true
            delayTimerRight.start()
        } else { // faz reset quando laneLeft volta a false, se necessário
            delayTimerRight.stop()
            changeColorTimerRight.stop()
            timeElapsedRight = false // Opcional: reseta o estado
        }
    }



    Keys.onPressed: function pressSpace(event){
        if (event.key === Qt.Key_Space) {
            isPressingSpace = true;
        } else if (event.key === Qt.Key_Escape) {
            Qt.quit();
        }
    }

    Keys.onReleased: function releaseSpace(event){
        if (event.key === Qt.Key_Space) {
            isPressingSpace = false;
        }
    }


    Canvas {
        id: routeCanvas
        anchors.fill: parent
        renderTarget: Canvas.FramebufferObject          // Necessário para animações suaves

        onPaint: {
            var ctx = routeCanvas.getContext("2d");
            ctx.clearRect(0, 0, width, height);

            // === Configurações do trapézio esquerdo (aceleração) ===
            ctx.save(); // Salvar o estado do contexto
            var trapezoidHeight = height*0.67;         // Altura fixa do trapézio
            var baseWidth = width * 0.1;                // Largura fixa da base
            var topWidth = baseWidth * 0.6;             // Largura fixa do topo (60% da base)
            var barX = leftMarginDownRight - baseWidth; // Posicionar à esquerda da borda da estrada
            var barY = height / 3;                      // Margem inferior (topo do trapézio)

            // Coordenadas do trapézio esquerdo (aceleração)
            var baseLeftX = barX;
            var baseRightX = barX + baseWidth;
            var topLeftX = leftMarginUpRight - topWidth; // Alinhar com a borda superior da estrada
            var topRightX = leftMarginUpRight;           // Alinhar com a borda superior da estrada

            // === Desenhar o contorno do trapézio esquerdo (aceleração) ===
            ctx.strokeStyle = trapezoidBorderColor;
            ctx.lineWidth = 1;                          // Contorno fino - para retirar o contorno no trapezio mexer aqui - 0
            ctx.beginPath();
            ctx.moveTo(baseLeftX, height);              // Base esquerda
            ctx.lineTo(baseRightX, height);             // Base direita
            ctx.lineTo(topRightX, barY);                // Topo direito
            ctx.lineTo(topLeftX, barY);                 // Topo esquerdo
            ctx.closePath();
            ctx.stroke();

            // === Preenchimento do trapézio esquerdo (aceleração) ===
            var fillHeight = trapezoidHeight * accelerationProgress; // Altura proporcional ao progresso
            var fillBaseY = height;                                  // Base do trapézio no fundo
            var fillTopY = fillBaseY - fillHeight;                   // Altura dinâmica do preenchimento

            // Interpolação das coordenadas para manter a inclinação correta
            var fillTopLeftX = topLeftX + ((baseLeftX - topLeftX) * (1 - accelerationProgress));
            var fillTopRightX = topRightX + ((baseRightX - topRightX) * (1 - accelerationProgress));

            // Preenchimento do trapézio esquerdo (aceleração)
            ctx.fillStyle = accelerationColor;
            ctx.beginPath();
            ctx.moveTo(baseLeftX, fillBaseY);  // Base esquerda
            ctx.lineTo(baseRightX, fillBaseY); // Base direita
            ctx.lineTo(fillTopRightX, fillTopY);  // Topo direito ajustado
            ctx.lineTo(fillTopLeftX, fillTopY);   // Topo esquerdo ajustado
            ctx.closePath();
            ctx.fill();
            ctx.restore(); // Restaurar o estado do contexto

            // === Configurações do trapézio direito (inverso) ===
            ctx.save(); // Salvar o estado do contexto
            var invertedTrapezoidHeight = height * 0.67;         // Altura fixa do trapézio
            var invertedBaseWidth = width * 0.1;                 // Largura fixa da base
            var invertedTopWidth = invertedBaseWidth * 0.6;      // Largura fixa do topo (60% da base)
            var invertedBarX = rightMarginDownLeft;              // Posicionar à direita da borda da estrada
            var invertedBarY = height / 3;                       // Margem inferior (topo do trapézio)

            // Coordenadas do trapézio direito (invertido)
            var invertedBaseLeftX = invertedBarX;                // Alinhar com a borda inferior da estrada
            var invertedBaseRightX = invertedBarX + invertedBaseWidth;
            var invertedTopLeftX = rightMarginUpLeft;            // Alinhar com a borda superior da estrada
            var invertedTopRightX = rightMarginUpLeft + invertedTopWidth; // Alinhar com a borda superior da estrada

            // === Desenhar o contorno do trapézio direito (invertido) ===
            ctx.strokeStyle = trapezoidBorderColor;
            ctx.lineWidth = 1;                          // Contorno fino
            ctx.beginPath();
            ctx.moveTo(invertedBaseLeftX, height);           // Base esquerda
            ctx.lineTo(invertedBaseRightX, height);          // Base direita
            ctx.lineTo(invertedTopRightX, invertedBarY);     // Topo direito
            ctx.lineTo(invertedTopLeftX, invertedBarY);      // Topo esquerdo
            ctx.closePath();
            ctx.stroke();

            // === Preenchimento do trapézio direito (invertido) ===
            var invertedFillHeight = invertedTrapezoidHeight * accelerationProgress; // Altura proporcional ao progresso
            var invertedFillBaseY = height;                                          // Base do trapézio no fundo
            var invertedFillTopY = invertedFillBaseY - invertedFillHeight;           // Altura dinâmica do preenchimento

            // Interpolação das coordenadas para manter a inclinação correta
            var invertedFillTopLeftX = invertedTopLeftX + ((invertedBaseLeftX - invertedTopLeftX) * (1 - accelerationProgress));
            var invertedFillTopRightX = invertedTopRightX + ((invertedBaseRightX - invertedTopRightX) * (1 - accelerationProgress));

            // Preenchimento do trapézio direito (invertido)
            ctx.fillStyle = accelerationColor;
            ctx.beginPath();
            ctx.moveTo(invertedBaseLeftX, invertedFillBaseY);  // Base esquerda
            ctx.lineTo(invertedBaseRightX, invertedFillBaseY); // Base direita
            ctx.lineTo(invertedFillTopRightX, invertedFillTopY);  // Topo direito ajustado
            ctx.lineTo(invertedFillTopLeftX, invertedFillTopY);   // Topo esquerdo ajustado
            ctx.closePath();
            ctx.fill();
            ctx.restore(); // Restaurar o estado do contexto

            // Estrada central (trapézio preto)
            ctx.save();
            ctx.fillStyle = "#2F2F2F"
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height);    // Base esquerda
            ctx.lineTo(rightMarginDownLeft, height);    // Base direita
            ctx.lineTo(rightMarginUpLeft, height / 3);  // Ponto superior direito
            ctx.lineTo(leftMarginUpRight, height / 3);  // Ponto superior esquerdo
            ctx.closePath();
            ctx.fill()
            ctx.restore();

            // Linha lateral esquerda (agora uma linha mais fina)
            ctx.save();
            var gradientLeft = ctx.createLinearGradient(width * 0.30, height / 3, width * 0.30, height)
            gradientLeft.addColorStop(0, "#2E2E2E")     // Tom mais escuro no topo
            gradientLeft.addColorStop(1, "#8A8A8A")     // Tom mais claro na base
            // ctx.strokeStyle = gradientLeft
            // ctx.strokeStyle = laneLeft === true ? "#4A90E2" : gradientLeft  // muda a cor da linha se calacar a linha
            ctx.strokeStyle = (laneLeft && timeElapsed) ? "#4A90E2" : gradientLeft
            ctx.lineWidth = lateralLineWidth
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height);    // Base direita da estrada central
            ctx.lineTo(leftMarginUpRight, height / 3);  // Ponto superior direito da estrada central
            ctx.stroke()
            ctx.restore();

            // Linha lateral direita (agora uma linha mais fina)
            ctx.save();
            var gradientRight = ctx.createLinearGradient(width * 0.7, height / 3, width * 0.7, height)
            gradientRight.addColorStop(0, "#2E2E2E")    // Tom mais escuro no topo
            gradientRight.addColorStop(1, "#8A8A8A")    // Tom mais claro na base
            // ctx.strokeStyle = gradientRight
            // ctx.strokeStyle = laneRight === true ? "#4A90E2" : gradientRight  // muda a cor da linha se calacar a linha
            ctx.strokeStyle = (laneRight && timeElapsedRight) ? "#4A90E2" : gradientRight
            ctx.lineWidth = lateralLineWidth
            ctx.beginPath()
            ctx.moveTo(rightMarginDownLeft, height);    // Base esquerda da estrada central
            ctx.lineTo(rightMarginUpLeft, height / 3);  // Ponto superior esquerdo da estrada central
            ctx.stroke()
            ctx.restore();

            // Linha tracejada com gradiente no topo
            ctx.save();
            // Criar o gradiente para o último terço da tela
            var overallGradient = ctx.createLinearGradient(0, height / 3, 0, height);
            overallGradient.addColorStop(0, "#2E2E2E"); // Branco no início do gradiente
            overallGradient.addColorStop(1, "#8A8A8A"); // Cinza claro no fim do gradiente
            ctx.fillStyle = overallGradient;

            const centerX = width / 2 - lineWidth / 2;  // Posição central
            for (let y = height / 4 + offset; y < height; y += totalHeight) {
                ctx.fillRect(centerX, y, lineWidth, dashHeight);
            }

            var lineGradient = ctx.createLinearGradient(0, height / 4, 0, height);
            lineGradient.addColorStop(0, "#2E2E2E");    // Branco no início do gradiente
            lineGradient.addColorStop(1, "#2E2E2E");    // Cinza claro no fim do gradiente
            ctx.fillStyle = lineGradient;
            ctx.fillRect(width/2-(lineWidth/2), height/4, lineWidth, dashHeight);
            ctx.restore();
        }
    }

    MouseArea {
        anchors.fill: parent
        onClicked: route.focus = true
    }
}
