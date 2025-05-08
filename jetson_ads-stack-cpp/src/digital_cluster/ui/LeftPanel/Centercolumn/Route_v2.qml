import QtQuick 2.15

Rectangle {
    id: route
    anchors.fill: parent
    color: "transparent"

    //Bool vars para indicar qual a linha calcada
    property bool laneRight: systemHandler.laneRight === "true" // criar o laneright no System
    // property bool laneRight: true
    property bool laneLeft: systemHandler.laneLeft === "true" // criar o laneleft no System
    // property bool laneLeft: true

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
    property color accelerationColor: "#0000FF"         // Cor do preenchimento interno da barra
    property color trapezoidBorderColor: "#2E2E2E"      // Cor do contorno da barra

    property bool isPressingSpace: false

    property real offset: 0                             // Deslocamento animado

    // Nova propriedade para a espessura das linhas laterais
    property int lateralLineWidth: 8                    // Espessura das linhas laterais (ajustável)

    // Temporizador para controle suave do progresso
    Timer {
        interval: 16                                    // Aproximadamente 60 FPS
        repeat: true
        running: isPressingSpace || accelerationProgress > 0
        onTriggered: {
            if (isPressingSpace) {
                accelerationProgress = Math.min(1, accelerationProgress + 0.01);
            } else {
                accelerationProgress = Math.max(0, accelerationProgress - 0.01);
            }
            routeCanvas.requestPaint();
        }
    }

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

    Keys.onPressed: function pressSpace(event){
        if (event.key === Qt.Key_Space) {
            isPressingSpace = true;
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
            ctx.strokeStyle = laneLeft === true ? "#4A90E2" : gradientLeft
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
            ctx.strokeStyle = laneRight === true ? "#4A90E2" : gradientRight
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

            // desenha a linha central
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
