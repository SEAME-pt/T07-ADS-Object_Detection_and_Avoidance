# ADS Project - Lane Keeping Assist - Level 1 Autonomy ADAS feature  
## Your lane, your pace!
</br>


- [ADS Project - Lane Keeping Assist - Level 1 Autonomy ADAS feature](#ads-project---lane-keeping-assist---level-1-autonomy-adas-feature)
  - [Introduction](#introduction)
  - [Background Information](#background-information)
  - [Objectives](#objectives)
  - [Mandatory Part](#mandatory-part)
  - [Common Instructions](#common-instructions)
  - [Skills](#skills)
  - [Evaluation](#evaluation)
  - [Submission](#submission)

</br>


## Introduction

In this project, you will be exposed to the intersection of virtual simulations and real-world applications. You will delve deep into the mechanisms of the Lane Keeping Assist System (LKA), a pivotal Level 1 autonomous driving feature. Using advanced simulation platforms and actual hardware, you will design, test, and implement an LKAS that can operate both virtually and in the real world. 
</br>


## Background Information

The dream of self-driving cars has been around for decades, long before they became a technical reality. From the early radio-controlled cars showcased in the 1930s World's Fair to the futuristic vehicles in science fiction, autonomous driving has always captured human imagination. The Lane Keeping Assist System is one of the first steps in making this dream come true. It's not just a technical marvel; LKAS plays a crucial role in ensuring safer roads by reducing lane departure incidents.  
</br>


## Objectives

1. Design a virtual LKAS capable of detecting and tracking lane markings using simulated sensors.
2. Implement feedback mechanisms to alert virtual drivers of unintentional lane departures.
3. Simulate corrective interventions, such as steering or braking adjustments, to ensure the virtual vehicle remains in its lane.
4. Translate the virtual LKAS system to a real-world application using the PiRacer, ensuring it operates effectively with real sensors and environments.

</br>


## Mandatory Part

1. Establish a comprehensive simulation environment using platforms like CARLA or Gazebo. This setup should include a detailed test track and a vehicle embedded with all necessary simulated sensors.
2. Develop algorithms to identify lane markings and determine the vehicle's position relative to them.
3. Implement a feedback system to alert the driver if the vehicle drifts from its lane, using both visual and auditory signals.
4. Create interventions in the simulation to redirect the vehicle back to its lane.
5. Transition from simulation to real-world application by setting up the LKAS on the PiRacer. Integrate necessary sensors and calibrate them to function in real environments.

</br>

## Common Instructions

- Always backup your data and code before making major changes.
- When transitioning to real-world testing, always test in controlled environments, ensuring all interventions are gradual and predictable to prevent abrupt movements or potential damage.
- Wear appropriate safety gear when working with hardware components.

</br>

## Skills

- Proficiency in simulation platforms such as CARLA or Gazebo.
- Understanding of sensor data processing and algorithmic design for lane detection.
- Integration skills, transitioning from a virtual to a real-world system.
- Calibration and testing skills for real-world hardware systems.
- Proficiency in ML and DL algorithms
- Proficiency in Linear Algebra

</br>


## Evaluation
In this project, every team must host ONE final submission demo & presentation (max. 30 mins) in front of all the other teams. Each team must find a way to organize this presentation making sure that all the other teams can be present and participate actively (Please work out what date/time works the best for every team). The date and time of each team's presentation must be communicated to staff well in advance (at least a week in advance). It is presenting team's responsibility to make sure that all the forms are filled in **immediately** after the presentation.

This project has two evaluation forms:
1. For evaluators (the audience) - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSe7AKrza228fzdDNgevTw4Gsz-ARlWcbtQmXn8JAEYaj24mzw/viewform?usp=sf_link) to evaluate the presenting team's final project submission
2. For evaluatee (the presentor) - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfYipLAaXFaAm23ZdW8ruXCfQDOXikLYwhxXwpve9C5kZoZvw/viewform?usp=sf_link) for general feedback on your workings on this project.

</br>

## Submission

1. Code: The source code of the project, including all necessary files and libraries. The code should be well-documented, readable, and organized in a logical manner.
2. Technical documentation: Detailed technical documentation that provides an overview of the project, including the background information, goals, objectives, technical requirements, software architecture, and design.
3. Test results: Detailed test results that demonstrate the performance and accuracy of the autonomous lane detection system. This should include test data and results, as well as any graphs or visualizations that help to show the performance of the system.
4. User manual: A comprehensive user manual that provides instructions on how to use the autonomous vehicle, including how to set up the sensors and other components, how to control the vehicle, and how to monitor its performance.
5. Presentation: A presentation that summarizes the project and highlights the key results and contributions of the students. This presentation can be in the form of a slide deck, video, or other format as appropriate.
6. Final report: A final report that summarizes the project and provides a detailed overview of the work that was completed, the results achieved, and the challenges encountered. The report should also include a discussion of future work that could be done to extend or improve the autonomous lane detection system.

</br>


# References

Here are some open source references and descriptions that could be used in the Road Surface Segmentation using PiRacer project:

1. OpenCV: OpenCV is a popular open-source computer vision library that provides a wide range of tools and algorithms for image and video processing. Participants could use OpenCV for pre-processing the video footage, extracting features, and identifying the lane markings.
    Link: [https://opencv.org/](https://opencv.org/)

2. TensorFlow: TensorFlow is an open-source machine learning framework that provides a wide range of tools for training deep neural networks. Participants could use TensorFlow for training a deep neural network for identifying extracted lane markings.
    Link: https://www.tensorflow.org/




These references are just examples and participants are encouraged to explore other open-source tools and resources that may be more suitable for their specific needs and requirements. Participants should be prepared to research and evaluate different open-source tools and resources, and to make informed decisions about which tools and resources to use for their projects.
