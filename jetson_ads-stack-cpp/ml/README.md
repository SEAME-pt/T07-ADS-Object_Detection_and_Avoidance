# ADS Project - Object Detection and Avoidance System - Level 2 Autonomy: Autonomous Lane Change and Collision Prevention
## Objects in sight, insights in mind

- [ADS Project - Object Detection and Avoidance System - Level 2 Autonomy: Autonomous Lane Change and Collision Prevention](#ads-project---object-detection-and-avoidance-system---level-2-autonomy-autonomous-lane-change-and-collision-prevention)
  - [Description](#description)
  - [Forewords](#forewords)
  - [Objective / Goal of the project](#objective--goal-of-the-project)
  - [Mandatory Part](#mandatory-part)
  - [Common Instruction](#common-instruction)
  - [Skills](#skills)
  - [Evaluation](#evaluation)
  - [Submission](#submission)

</br>


## Description

This project is designed to give you a hands-on experience in developing an object detection and avoidance system for vehicles, using both virtual simulations and real-world hardware implementation. Utilizing platforms like CARLA, Gazebo, or AirSim, you will create a system that not only detects obstacles in a vehicle's path but also performs autonomous maneuvers to avoid potential collisions while ensuring passenger safety.

</br>

## Forewords

The history of object detection and avoidance in automotive technology is a testament to the advances in computer vision and sensor integration. What began as a rudimentary alert system has now evolved into sophisticated autonomous interventions. This project channels the essence of these innovations, offering a glimpse into the future of vehicular autonomy where safety and technology go hand in hand.

</br>

## Objective / Goal of the project

- To develop a robust simulation-based object detection that can detect obstacles and execute safe lane changes.
- To apply the most appropriate algorithm for collision avoidance.
- To generate alerts for drivers during autonomous maneuvers and ensure smooth reintegration into traffic.
- To successfully implement and test the system on a PiRacer model car, using real cameras and sensors.

</br>

## Mandatory Part

1. Program the vehicle to detect objects using camera and sensor data and to determine safe avoidance strategies.
2. Implement visual and auditory warning systems for alerting during autonomous interventions.
3. Design and apply dynamic path-planning for steering and braking to facilitate safe object avoidance and lane changes.
4. Transfer the simulation model to a PiRacer, incorporating real hardware sensors and testing in a controlled environment.
5. Implement the same tasks for road signs detection and recognition (optional).

</br>

## Common Instruction

- Follow software development best practices, including version control and modular coding.
- Test each subsystem individually before full-system integration.
- Ensure all team members are familiar with both the simulation environment and the real-world hardware.
- In simulation, prioritize the integrity of the virtual environment to prevent system crashes.
- During real-world testing, use a controlled environment free from external interference.
- Always monitor the PiRacer during testing to manually override in case of malfunctions.

</br>

## Skills

- Proficiency in the use of simulation software for autonomous systems.
- Advanced knowledge of computer vision techniques and sensor data fusion.
- Practical experience in embedded systems programming and hardware integration.
- Critical problem-solving skills in dynamic and unpredictable scenarios.

</br>

## Evaluation
In this project, every team must host ONE final submission demo & presentation (max. 30 mins) in front of all the other teams. Each team must find a way to organize this presentation making sure that all the other teams can be present and participate actively (Please work out what date/time works the best for every team). The date and time of each team's presentation must be communicated to staff well in advance (at least a week in advance). It is presenting team's responsibility to make sure that all the forms are filled in **immediately** after the presentation.

This project has two evaluation forms:
1. For evaluators (the audience) - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLScDmOPBC_sEXiWNCGvxTPrTVGHdmdt0VY5Joz9OgMV29-1Cyg/viewform?usp=sf_link) to evaluate the presenting team's final project submission
2. For evaluatee (the presentor) - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSd5s9SclsQ5rz3D86K3csErFntdp-XOJAieVntfY5DDh4ubow/viewform?usp=sf_link) for general feedback on your workings on this project.

</br>

## Submission

1. Code: The source code of the project, including all necessary files and libraries. The code should be well-documented, readable, and organized in a logical manner.
2. Technical documentation: Detailed technical documentation that provides an overview of the object detection and avoidance system, including the background information, projects goals, objectives, technical requirements, software architecture, and design.
3. Test results: Detailed test results that demonstrate the performance and accuracy of the object detection and avoidance system. This should include test data and results, and visualizations such as graphs that help to illustrate the performance of the system.
4. User manual: A comprehensive user manual that provides instructions on how to use the autonomous vehicle, including how to set up the sensors and other components, how to control the vehicle, and how to monitor its performance.
5. Presentation: A presentation that summarizes the project and highlights the key results and contributions of the students. This presentation can be in the form of a slide deck, video, or other format as appropriate.
6. Final report: A final report that summarizes the project and provides a detailed overview of the work that was completed, the results achieved, and the challenges encountered. The report should also include a discussion of future work that could be done to extend or improve the autonomous lane change and collision prevention features.

</br>

# References

Here are some open source references and descriptions that could be used in the project:

1. OpenCV: OpenCV is a popular open-source computer vision library that provides a wide range of tools and algorithms for image and video processing. Participants could use OpenCV for pre-processing the video footage, extracting features, and identifying the objects.
    Link: [https://opencv.org/](https://opencv.org/)

2. ROS (Robot Operating System): ROS is an open-source software framework for robotics that provides a wide range of tools and libraries for building robotic systems. Participants could use ROS for integrating the obstacle avoidance algorithm into the PiRacer vehicle with lidar and for testing and validating the algorithm in real-world conditions.
    Link: https://www.ros.org/


These references are just examples and participants are encouraged to explore other open-source tools and resources that may be more suitable for their specific needs and requirements. Participants should be prepared to research and evaluate different open-source tools and resources, and to make informed decisions about which tools and resources to use for their projects.