import QtQuick 2.15

Rectangle {
    id: route
    anchors.fill: parent
    color: "transparent"

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
        // running: true
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
            var trapezoidHeight = height*0.67;         // Altura fixa do trapézio
            var baseWidth = width * 0.1;                // Largura fixa da base
            var topWidth = baseWidth * 0.6;             // Largura fixa do topo (60% da base)
            var barX = width * 0.1;                     // Posição horizontal do trapézio
            var barY = height / 3;                      // Margem inferior (topo do trapézio)

            // Coordenadas do trapézio esquerdo (aceleração)
            var baseLeftX = barX;
            var baseRightX = barX + baseWidth;
            var topLeftX = leftMarginUpLeft - 20;       // Ajuste da posição do topo esquerdo
            var topRightX = leftMarginUpLeft;           // Ajuste da posição do topo direito

            // === Desenhar o contorno do trapézio esquerdo (aceleração) ===
            ctx.strokeStyle = trapezoidBorderColor;
            ctx.lineWidth = 0;
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


            // === Configurações do trapézio direito (inverso) ===
            var invertedTrapezoidHeight = height * 0.67;         // Altura fixa do trapézio
            var invertedBaseWidth = width * 0.1;                 // Largura fixa da base
            var invertedTopWidth = invertedBaseWidth * 0.6;      // Largura fixa do topo (60% da base)
            var invertedBarX = width * 0.9;                      // Posição horizontal do trapézio (à direita)
            var invertedBarY = height / 3;                       // Margem inferior (topo do trapézio)

            // Coordenadas do trapézio direito (invertido)
            var invertedBaseLeftX = invertedBarX - invertedBaseWidth;  // Base esquerda ajustada
            var invertedBaseRightX = invertedBarX;                     // Base direita ajustada
            var invertedTopLeftX = rightMarginUpRight;              // Ajuste da posição do topo esquerdo
            var invertedTopRightX = rightMarginUpRight + 20; // Ajuste da posição do topo direito

            // === Desenhar o contorno do trapézio direito (invertido) ===
            ctx.strokeStyle = trapezoidBorderColor;
            ctx.lineWidth = 0;
            ctx.beginPath();
            ctx.moveTo(invertedBaseLeftX, height);           // Base esquerda
            ctx.lineTo(invertedBaseRightX, height);          // Base direita
            ctx.lineTo(invertedTopRightX, invertedBarY);     // Topo direito
            ctx.lineTo(invertedTopLeftX, invertedBarY);      // Topo esquerdo
            ctx.closePath();
            ctx.stroke();



            // // Estrada central (trapézio preto)
            ctx.fillStyle = "#2F2F2F"
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height);    // Base esquerda
            ctx.lineTo(rightMarginDownLeft, height);    // Base direita
            ctx.lineTo(rightMarginUpLeft, height / 3);  // Ponto superior direito
            ctx.lineTo(leftMarginUpRight, height / 3);  // Ponto superior esquerdo
            ctx.closePath();
            ctx.fill()

            // // Lateral esquerda com inclinação ajustada
            var gradientLeft = ctx.createLinearGradient(width * 0.30, height / 3, width * 0.30, height)
            gradientLeft.addColorStop(0, "#2E2E2E")     // Tom mais escuro no topo
            gradientLeft.addColorStop(1, "#8A8A8A")     // Tom mais claro na base
            ctx.fillStyle = gradientLeft
            ctx.beginPath()
            ctx.moveTo(leftMarginDownRight, height);    // Base direita
            ctx.lineTo(leftMarginUpRight, height / 3);  // Ponto superior direito
            ctx.lineTo(leftMarginUpLeft, height / 3);   // Ponto superior esquerdo
            ctx.lineTo(leftMarginDownLeft, height);     // Base esquerda
            ctx.closePath()
            ctx.fill()

            // // Lateral direita com inclinação ajustada
            var gradientRight = ctx.createLinearGradient(width * 0.7, height / 3, width * 0.7, height)
            gradientRight.addColorStop(0, "#2E2E2E")    // Tom mais escuro no topo
            gradientRight.addColorStop(1, "#8A8A8A")    // Tom mais claro na base
            ctx.fillStyle = gradientRight
            ctx.beginPath()
            ctx.moveTo(rightMarginDownRight, height);   // Base direita
            ctx.lineTo(rightMarginUpRight, height / 3); // Ponto superior direito
            ctx.lineTo(rightMarginUpLeft, height / 3);  // Ponto superior esquerdo
            ctx.lineTo(rightMarginDownLeft, height);    // Base esquerda
            ctx.closePath()
            ctx.fill()

            // // Linha tracejada com gradiente no topo
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
        }

    }

    // focus: true // Necessário para capturar eventos de

    MouseArea {
        anchors.fill: parent
        onClicked: route.focus = true
    }
}
