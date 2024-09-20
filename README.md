
# Drug Detection System

## Overview

This project implements a drug detection system using the YOLOv5 deep learning model for real-time object detection. It identifies drugs from video input (such as a webcam feed) and automatically saves detected images. A Telegram bot is integrated to send alerts whenever drugs are detected, enhancing the system’s functionality with real-time notifications.

## Files

### drug_detector.py
This is the main script that performs real-time drug detection using the YOLOv5 model. The script captures video frames, processes them using the pre-trained model, and checks for objects that exceed a confidence threshold. It includes functionality to:
- Save images with detected drugs.
- Send notifications via a Telegram bot.
- Display real-time bounding boxes and labels on detected objects.

Key features:
- Uses YOLOv5 for object detection.
- Configured to save images every 10 frames, if drugs are detected.
- Confidence threshold is set at 45%, which can be modified based on the use case.
- The detection results are rendered directly on the video feed for display.

### telegram_sender.py
This script is responsible for sending alert messages via a Telegram bot whenever a drug is detected in the video feed. It connects to the Telegram API and sends the saved image of the detected drug along with the detection details.

## Dependencies

Ensure you have the following libraries installed:

- torch: Required for loading and using the YOLOv5 model.
- opencv-python: For video capture and image processing.
- torchvision: For additional support in PyTorch-based image processing.
- python-telegram-bot: To send alerts via the Telegram API.
  
You can install these dependencies using the following command:
bash
pip install torch opencv-python torchvision python-telegram-bot


## Setup and Usage

1. *Clone the Repository*:
   Clone this repository using the following command:
   bash
   git clone https://github.com/Iniyan2003/Drug-Detection.git
   

2. *Install Dependencies*:
   Ensure that you have all the necessary libraries installed by running:
   bash
   pip install torch opencv-python torchvision python-telegram-bot
   

3. *Set up Telegram Bot*:
   - Set up a Telegram bot using the BotFather.
   - Obtain the bot token and configure it in the telegram_sender.py file.
   - Update the chat ID to ensure notifications are sent to the correct recipient.

4. *Run the Detection Script*:
   Start the detection system by running:
   bash
   python drug_detector.py
   
   The system will begin capturing video, and drug detection will occur in real-time. Detected images will be saved in the detected_images folder.

5. *Exit the System*:
   To stop the video capture and exit the program, press the 'q' key.

## How it Works

- The system captures video from a camera source (default is 0, i.e., the primary webcam).
- YOLOv5 processes each frame to detect objects, focusing on those with a confidence score above 0.45.
- Every 10 frames, if a drug is detected, an image is saved, and a Telegram notification is sent.
- Bounding boxes and labels are drawn on detected objects, and the video stream is displayed with real-time updates.

## Customization

- *Model Customization*: If you have a custom YOLOv5 model (best.pt), replace the model path in the drug_detector.py script.
- *Confidence Threshold*: Adjust the confidence threshold by modifying the line if confidence > 0.45 to suit your needs.
- *Frame Capture Frequency*: Change the frequency of saved frames by updating the frame_count % 10 == 0 condition.

## Future Improvements

- Expand the object detection classes for multi-class drug identification.
- Integrate more robust error handling for video capture and notification failures.

## Acknowledgements
- The project leverages the [YOLOv5 model](https://github.com/ultralytics/yolov5), known for its fast and accurate object detection capabilities.
