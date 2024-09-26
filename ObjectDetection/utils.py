import cv2
import numpy as np
from tflite_support.task import processor
import paho.mqtt.client as mqtt

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

# MQTT setup
broker = "192.168.8.195"
port = 1883
topic = "videos/playlist"
username = "pi"
password = "kasun1234"

# Global variable to store the previous category name
previous_category_name = None

def send_mqtt_message(message):
    client = mqtt.Client()
    client.username_pw_set(username, password)
    client.connect(broker, port)
    client.publish(topic, message)
    client.disconnect()

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualized.

    Returns:
      Image with bounding boxes.
    """
    global previous_category_name  # Declare the global variable

    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x, _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # Check if the category has changed and print only if it has
        if category_name and category_name != previous_category_name:
            print(category_name)
            # Optionally send the MQTT message here if needed
            send_mqtt_message(category_name)
            previous_category_name = category_name  # Update the previous category name

    return image
