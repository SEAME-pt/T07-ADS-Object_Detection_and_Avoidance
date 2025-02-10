# Game Controller Handling with SDL2

This project uses SDL2 to capture events from a game controller (gamepad) and map specific actions to the controller's buttons and axes. It is built using C++ and CMake, and it requires the SDL2 library for game controller functionality.

## Dependencies

### 1. **Install SDL2**

To compile and run this project, the SDL2 library needs to be installed. SDL2 can be installed in different ways depending on the operating system.

#### On Ubuntu (Linux)

```bash
sudo apt update
sudo apt install libsdl2-dev
```

### 2. Install CMake

If you don't have CMake installed, you can install it based on your operating system.
#### On Ubuntu (Linux)

```bash
sudo apt update
sudo apt install cmake
```

## How to Compile the Project
### 1. Create a Build Directory

First, create a separate build directory for the compilation files. Run the following commands in the terminal (inside the project directory):
```bash
mkdir build
cd build
```
### 2. Run CMake

Within the build directory, run the cmake command to generate the build files:
```bash
cmake ..
```
This will configure the project and check if all the necessary dependencies are installed.
### 3. Compile the Project

After running CMake, compile the project with the make command:
```bash
make
```
This will compile all the source files and generate the project executable.
### 4. Run the Project

After successful compilation, run the generated executable. On Linux/macOS:
```bash
sudo ./your_executable_name
```

## Example of Usage

Here is a simple main.cpp to demonstrate how to use the Controller class.
main.cpp

```cpp
#include "Controller.hpp"

void onPressExample() {
    std::cout << "Button Pressed!" << std::endl;
}

void onReleaseExample() {
    std::cout << "Button Released!" << std::endl;
}

int main() {
    try {
        Controller controller;

        // Create actions for a button
        Actions buttonActions;
        buttonActions.onPress = onPressExample;
        buttonActions.onRelease = onReleaseExample;

        // Map the actions to the A button (SDL_CONTROLLER_BUTTON_A)
        controller.setButtonAction(SDL_CONTROLLER_BUTTON_A, buttonActions);

        // Start listening for gamepad events
        controller.listen();

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

## Architecture Decision Record (ADR)
Why Choose SDL2 over joystick.h?
Context

When implementing controller support for a game or application, we have multiple options for handling game controller input. Two of the most common choices are SDL2 (Simple DirectMedia Layer) and joystick.h (typically part of the Linux joystick interface or custom libraries). This ADR explains why SDL2 was chosen for this project over joystick.h.
Decision

We decided to use SDL2 over joystick.h for the following reasons:

1. Cross-Platform Support: SDL2 is a cross-platform library that supports game controller input on multiple platforms, including Linux, Windows, and macOS. It abstracts away platform-specific complexities, making it easier to handle controllers across different operating systems. On the other hand, joystick.h is generally Linux-specific and would require additional effort to support other platforms.

2. Ease of Use: SDL2 provides a higher-level API for game controller input that is easy to use and well-documented. It provides functions like SDL_GameControllerOpen, SDL_GameControllerGetButton, and SDL_GameControllerGetAxis, which simplify the process of handling game controllers compared to the lower-level joystick interface.

3. Active Community and Documentation: SDL2 has a large and active community, which means that there are a lot of resources and documentation available for troubleshooting and best practices. joystick.h is less commonly used, and while it may be suitable for specific use cases, it lacks the same level of community support and documentation.

4. Compatibility with Gamepads: SDL2 is explicitly designed for gamepad support and handles a variety of game controllers, including modern ones with multiple buttons and axes. In contrast, joystick.h may not handle modern controllers as gracefully, and additional effort might be required to manage different controller types.

Consequences

By choosing SDL2, we gain cross-platform compatibility, a higher-level API for handling game controllers, and a well-supported framework. The trade-off is that SDL2 is a larger dependency compared to joystick.h, but the advantages in ease of use and flexibility far outweigh this drawback for this project.

Conclusion

This project provides a simple and extensible way to handle game controller input using SDL2. With the ability to map buttons and axes to specific actions, it is easy to create complex gamepad interactions. SDL2's cross-platform support and ease of use make it a great choice for game controller handling, and this documentation provides everything needed to compile, run, and extend the project.


## Key Points:
- **Install Dependencies:** Instructions for installing SDL2, CMake, and necessary compilers.
- **Project Structure:** Explains how the project is structured with the `Controller` class, which handles gamepad events.
- **Compilation and Execution:** Detailed instructions on how to compile and run the project.
- **Example Code:** A simple example showing how to use the `Controller` class to map actions to buttons.
- **ADR:** Justifies the choice of SDL2 over `joystick.h` for cross-platform support, ease of use, and compatibility with modern game controllers.

