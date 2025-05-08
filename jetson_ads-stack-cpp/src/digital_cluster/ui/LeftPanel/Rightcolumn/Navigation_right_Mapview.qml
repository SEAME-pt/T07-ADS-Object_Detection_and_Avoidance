import QtQuick 2.15
import QtLocation
import QtPositioning
import Qt5Compat.GraphicalEffects
import QtQuick.Controls

Item {
    id: container
    anchors.fill: parent
    clip: true

    Rectangle {
        id: clipMask
        anchors.fill: parent
        radius: 8
        color: "transparent"
        // border.color: "red"
        border.color: "#1E1E1E"
        border.width: 9
        z: 1
    }

    MapView {
        id: view
        anchors.fill: parent
        z: 0 // Coloca o MapView atrás da máscara
        anchors {
           left: parent.left
           right: parent.right
           top: parent.top
           bottom: parent.bottom

           // Simula o padding
           leftMargin: 7
           rightMargin: 7
           topMargin: 7
           bottomMargin: 7
        }

        map {
            plugin: Plugin {
                name: "osm"
                parameters: [
                            PluginParameter { name: "osm.mapping.host"; value: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" }
                        ]
            }

            copyrightsVisible: false
            // zoomLevel: (maximumZoomLevel - minimumZoomLevel) / 2
            zoomLevel: 25
            tilt: 50 // inclinação do mapa
            // colorOverlay: Qt.rgba(0, 0, 0, 0.5) // escurecer o mapa utilizar com Map
            center {
                latitude: 41.14848
                longitude: -8.61301
            }
        }
    }

    MapItemView {
         id: mapItem
         parent: view.map
         // visible: false

         model: ListModel {
             id: markersModel
             ListElement {
                 latitude: 41.14848
                 longitude: -8.61301
                 icon: "../../assets/red_marker.png" // Marcador inicial
             }
         }

         delegate: MapQuickItem {
             id: marker
             parent: view.map
             coordinate: QtPositioning.coordinate(latitude, longitude)
             anchorPoint.x: markerImage.width / 2
             anchorPoint.y: markerImage.height
             sourceItem: Image {
                 id: markerImage
                 source: icon
                 width: 32
                 height: 32
             }
         }
    }
}


