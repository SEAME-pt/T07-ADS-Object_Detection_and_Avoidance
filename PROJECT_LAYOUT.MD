jetson-ads-stack-cpp/
├── README.md
├── LICENSE
├── .gitignore
├── CMakeLists.txt
├── setup.sh
├── third_party/
│   └── yolov8_trt/         # submodule or compiled inference engine (TensorRT)
│
├── include/
│   ├── core/
│   │   ├── camera_input.hpp
│   │   ├── display_output.hpp
│   │   └── utils.hpp
│   ├── lane_detection/
│   │   ├── lane_detector.hpp
│   │   └── lane_visualizer.hpp
│   ├── object_detection/
│   │   ├── yolov8_detector.hpp
│   │   ├── tracker.hpp
│   │   └── class_filter.hpp
│   ├── control/
│   │   ├── mpc_controller.hpp
│   │   └── actuator_interface.hpp
│   └── integration/
│       └── perception_fusion.hpp
│
├── src/
│   ├── main.cpp                  # Entry point
│   ├── core/
│   │   ├── camera_input.cpp
│   │   ├── display_output.cpp
│   │   └── utils.cpp
│   ├── lane_detection/
│   │   ├── lane_detector.cpp
│   │   └── lane_visualizer.cpp
│   ├── object_detection/
│   │   ├── yolov8_detector.cpp
│   │   ├── tracker.cpp
│   │   └── class_filter.cpp
│   ├── control/
│   │   ├── mpc_controller.cpp
│   │   └── actuator_interface.cpp
│   └── integration/
│       └── perception_fusion.cpp
│
├── models/
│   ├── yolov8.engine            # TensorRT optimized engine
│
├── config/
│   ├── config.yaml              # runtime config: detection enabled, thresholds, camera ID
│   └── mpc_params.yaml
│
├── scripts/
│   ├── convert_onnx_to_trt.sh
│   └── run_main.sh
│
├── data/
│   ├── calibration/
│   ├── test_videos/
│   └── logs/
│
├── tests/
│   ├── test_lanes.cpp
│   ├── test_objects.cpp
│   ├── test_fusion.cpp
│   └── CMakeLists.txt
│
├── docker/
│   ├── Dockerfile.jetson
│   └── docker-compose.yml
│
└── docs/
    ├── architecture_diagram.png
    ├── object_detection_flow.md
    └── sprint_notes/
        ├── sprint_1.md
        └── ...
