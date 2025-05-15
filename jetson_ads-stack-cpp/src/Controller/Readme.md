# Controller Application Documentation

## Overview
The `Controller` application is designed to handle joystick input and control the `JetCar`. It supports various button actions, axis movements, and an autonomous driving mode.

## Features
- **Joystick control**: Adjust steering and motor speed using axis inputs.
- **Autonomous mode toggle**: Pressing the `HOME` button switches between manual and autonomous modes.
- **GPIO-based lighting system**: Control different light settings using predefined button mappings.

## Architecture Decision Record (ADR)
### **Why SDL2?**
The application uses **SDL2** for handling joystick input due to:
- **Cross-platform compatibility**: SDL2 works across different operating systems.
- **Robust event handling**: SDL2 provides efficient event-driven input management.
- **Performance optimization**: SDL2 is lightweight and optimized for real-time applications.

## Dependencies
To install the required dependencies on your Jetson Nano, run the following commands:
```sh
sudo apt update
sudo apt install libgpiod-dev libgpiod1

sudo ln -s /usr/lib/aarch64-linux-gnu/libgpiod.so.1 /usr/lib/aarch64-linux-gnu/libgpiod.so.2

sudo apt install joystick libsdl2-dev libsdl2-2.0-0
```

## Adding a New Action to the Controller
To add a new action to the controller, follow these steps:

1. **Define the Action Function**
   Create a function that implements the desired behavior. Example:
   ```cpp
   void customAction() {
       std::cout << "Custom action triggered!" << std::endl;
   }
   ```

2. **Map the Function to a Button**
   In `main()`, use `setButtonAction()` to bind the function to a button:
   ```cpp
   controller.setButtonAction(BTN_X, {nullptr, customAction});
   ```
   This example triggers `customAction` when the `X` button is released.

3. **Compile and Run**
   Recompile the application and test the new action.
   ```sh
   make
   ./ControllerExec
   ```

## Autonomous Mode Activation
The `HOME` button toggles autonomous mode by calling `changeMode`:
```cpp
changeModeActions.onRelease = [&](){
    changeMode(controller.getMode(), controller, jetCar);
};
```
This function switches between manual (`MODE_JOYSTICK`) and autonomous (`MODE_AUTONOMOUS`) modes, stopping the vehicle before transitioning.

## GPIO Pin Assignments
| Function       | GPIO Pin |
|---------------|---------|
| Brake Lights  | 17      |
| Low Beam     | 18      |
| High Beam    | 19      |
| Left Signal  | 20      |
| Right Signal | 21      |

## Button Mapping
| Button   | Action |
|----------|--------|
| `HOME`   | Toggle Autonomous Mode |
| `A`      | Customizable Action |
| `B`      | Customizable Action |
| `X`      | Customizable Action |
| `Y`      | Customizable Action |

## Cross-Compilation & Deployment to Jetson Nano
To cross-compile the application and deploy it to the Jetson Nano, follow these steps:

1. **Run the QEMU emulator and Docker Compose:**
   ```sh
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   docker compose up
   ```

2. **Access the build container:**
   ```sh
   docker exec -it t07-ads_lane-detection-controller-1 /bin/bash
   ```

3. **Build the project inside the container:**
   ```sh
   cd build
   cmake ..
   make
   ```

4. **Copy the executable to the Jetson Nano:**
   ```sh
   scp ControllerExec team07@10.21.221.47:/home/team07
   ```
   Repeat the process for `mainCarServer` if needed.

## Conclusion
This documentation provides an overview of how to extend the `Controller` application, the rationale behind using SDL2, and details on autonomous mode activation. Additionally, it outlines the steps to set up the Jetson Nano, cross-compile the application, and deploy it to the target device. By following these steps, you can customize and expand the controller’s functionality efficiently.

