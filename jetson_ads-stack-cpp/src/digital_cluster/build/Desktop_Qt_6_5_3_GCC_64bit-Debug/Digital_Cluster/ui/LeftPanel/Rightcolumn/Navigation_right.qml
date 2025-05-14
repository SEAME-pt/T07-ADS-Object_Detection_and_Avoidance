import QtQuick 2.15
import QtLocation
import QtPositioning
import Qt5Compat.GraphicalEffects


Rectangle {
    id: mapContainer

    anchors {
        fill: parent
    }

    // radius: 30 // Define o raio dos cantos arredondados
    clip: true // Garante que o conteúdo do mapa respeite os cantos arredondados
    // border.color: "red"
    // border.width: 8


    Plugin {
        id: mapPlugin
        name: "osm"
    }

    PositionSource {
            id: positionSource
            // active: true  // Ativa a obtenção de posição automaticamente
            // onPositionChanged: {
            //     if (mapCircle) {
            //         mapCircle.center = position.coordinate
            //     }
            // }
        }

    Map {
        id: map
        anchors.fill: parent
        plugin: mapPlugin
        center: QtPositioning.coordinate(41.14, -8.61) //Porto
        zoomLevel: 25
        copyrightsVisible: false
        color: "black"
        property geoCoordinate startCentroid

        PinchHandler {
            id: pinch
            target: null
            onActiveChanged: if (active) {
                                 map.startCentroid = map.toCoordinate(pinch.centroid.position, false)
                             }
            onScaleChanged: (delta) => {
                                map.zoomLevel += Math.log2(delta)
                                map.alignCoordinateToPoint(map.startCentroid, pinch.centroid.position)
                            }
            onRotationChanged: (delta) => {
                                   map.bearing -= delta
                                   map.alignCoordinateToPoint(map.startCentroid, pinch.centroid.position)
                               }
            grabPermissions: PointerHandler.TakeOverForbidden
        }
        WheelHandler {
            id: wheel
            // workaround for QTBUG-87646 / QTBUG-112394 / QTBUG-112432:
            // Magic Mouse pretends to be a trackpad but doesn't work with PinchHandler
            // and we don't yet distinguish mice and trackpads on Wayland either
            acceptedDevices: Qt.platform.pluginName === "cocoa" || Qt.platform.pluginName === "wayland"
                             ? PointerDevice.Mouse | PointerDevice.TouchPad
                             : PointerDevice.Mouse
            rotationScale: 1/120
            property: "zoomLevel"
        }
        DragHandler {
            id: drag
            target: null
            onTranslationChanged: (delta) => map.pan(-delta.x, -delta.y)
        }
        Shortcut {
            enabled: map.zoomLevel < map.maximumZoomLevel
            sequence: StandardKey.ZoomIn
            onActivated: map.zoomLevel = Math.round(map.zoomLevel + 1)
        }
        Shortcut {
            enabled: map.zoomLevel > map.minimumZoomLevel
            sequence: StandardKey.ZoomOut
            onActivated: map.zoomLevel = Math.round(map.zoomLevel - 1)
        }

        property MapCircle mapCircle

        Component.onCompleted: {
            // Cria o círculo no mapa
            mapCircle = Qt.createQmlObject('import QtLocation; MapCircle {}',  map, "dynamicMapCircle") // cria um componente
            // mapCircle.center = positionSource.position.coordinate
            mapCircle.center = QtPositioning.coordinate(41.18, -8.61)
            mapCircle.radius = 10.0  // Define o raio (em metros)
            /*mapCircle.color = "rgba(0, 255, 0, 0.4)"*/  // Verde translúcido
            mapCircle.border.width = 3
            mapCircle.border.color = "green"
            addMapItem(mapCircle)
            map.center = mapCircle.center
            // console.log("Nova posição:", positionSource.position.coordinate)
            // Centraliza o mapa na posição inicial (se obtida)
            // if (positionSource.position.coordinate.isValid) {
            //     center = positionSource.position.coordinate
            // }
        }


    }

    // color: "orange"



}

