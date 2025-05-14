import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15

Rectangle {
    id: gamepadPage
    width: 1280
    height: 400
    color: "#1C2526" // Fundo escuro, similar ao cluster

    // Propriedade para o modo selecionado (caso precise vincular a outras partes)
    property string currentMode: "Normal"

    // Sinal para ações dos botões (opcional)
    signal buttonPressed(string buttonName)

    DPad{
        id:dpad
    }

    // Área central para a câmera
    // Rectangle {
    //     id: cameraArea
    //     width: 600
    //     height: 300
    //     color: "#555555" // Placeholder para a câmera
    //     border.color: "white"
    //     border.width: 2
    //     anchors.centerIn: parent

    //     Text {
    //         anchors.centerIn: parent
    //         color: "white"
    //         text: "Área da Câmera"
    //         font.pixelSize: 20
    //     }

        //Comando -gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink


        // Você pode substituir por um componente de vídeo ou imagem, ex.:
        // Image { source: "caminho/para/imagem.jpg"; anchors.fill: parent }
        // ou
        // Video { source: "caminho/para/video.mp4"; anchors.fill: parent }

        // Video {
        //     id: cameraFeed
        //     source: "rtsp://seu_endereço_de_video" // Substitua pelo endereço do stream da câmera
        //     anchors.fill: parent
        //     // autoPlay: true

        //     // Substitua autoPlay por uma chamada ao play()
        //     Component.onCompleted: {
        //         play() // Inicia a reprodução automaticamente
        //     }

        //     // Adicione logs para depurar erros
        //     onErrorChanged: {
        //         console.log("Erro no vídeo:", errorString)
        //     }
        //     onPlaybackStateChanged: {
        //         console.log("Estado de reprodução:", playbackState)
        //     }
        // }
    // }

    Rectangle {
        id: cameraArea
        width: 600
        height: 300
        color: "#555555"
        border.color: "white"
        border.width: 2
        anchors.centerIn: parent

        Image {
            id: cameraFeed
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            cache: false // Evita cache para garantir atualização
            source: "frame.jpg" // caminho para o frame.jpg

            Timer {
                interval: 100 // Atualiza a cada 100ms (10 FPS)
                running: true
                repeat: true
                onTriggered: {
                    // Força a recarga da imagem
                    cameraFeed.source = ""
                    cameraFeed.source = "frame.jpg"
                }
            }

            onStatusChanged: {
                if (status === Image.Error) {
                    console.log("Erro ao carregar frame:", cameraFeed.source)
                }
            }
        }
    }

    TopButtons {
        id: topButtons
    }

    // Botões Select e Start (acima da câmera)
    // Row {
    //     anchors.horizontalCenter: parent.horizontalCenter
    //     anchors.bottom: cameraArea.top
    //     anchors.bottomMargin: 10
    //     spacing: 20

    //     Rectangle {
    //         width: 80
    //         height: 30
    //         color: "#666666"
    //         radius: 5

    //         Text {
    //             anchors.centerIn: parent
    //             color: "white"
    //             text: "Select"
    //         }

    //         MouseArea {
    //             anchors.fill: parent
    //             onClicked: {
    //                 console.log("Select clicado")
    //                 gamepadPage.buttonPressed("Select")
    //             }
    //         }
    //     }

    //     Rectangle {
    //         width: 80
    //         height: 30
    //         color: "#666666"
    //         radius: 5

    //         Text {
    //             anchors.centerIn: parent
    //             color: "white"
    //             text: "Start"
    //         }

    //         MouseArea {
    //             anchors.fill: parent
    //             onClicked: {
    //                 console.log("Start clicado")
    //                 gamepadPage.buttonPressed("Start")
    //             }
    //         }
    //     }
    // }

    Buttons{
        id:buttonsRight
    }

    onButtonPressed: function(button) {
        console.log("Button pressed GamePad: ", button);
        // systemHandler.gamePad = button;
        if (button === "EndGame") {
            mainWindow.playMode = false;
            globalModesModel.get(3).selected = false;
        } else {
            systemHandler.gamePad = button;
        }

        // console.log("button gamepad no gamepad: ", systemHandler.gamePad);
    }

}
