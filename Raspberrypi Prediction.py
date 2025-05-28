import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import sys
import subprocess
import os
import time
import board
import busio
import RPi.GPIO as GPIO  # For button input
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from adafruit_ssd1306 import SSD1306_I2C

# ✅ Initialize I2C OLED display
i2c = busio.I2C(board.SCL, board.SDA)
display = SSD1306_I2C(128, 64, i2c)
display.fill(0)
display.show()

# ✅ Initialize GPIO for Button
BUTTON_PIN = 17  # GPIO 17 (Pin 11)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Enable Pull-up

# ✅ Load the trained model
MODEL_PATH = "/home/ansh/Desktop/project/trained_model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Error: Model file not found.")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define class labels
class_labels = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 
                'Herbicide Growth Damage', 'Leaf Hopper Jassids', 'Leaf Redding']

# ✅ Function to display text on OLED
def show_message_on_oled(message):
    display.fill(0)
    font = ImageFont.load_default()
    image = Image.new('1', (display.width, display.height))
    draw = ImageDraw.Draw(image)

    draw.text((10, 25), message, font=font, fill=255)

    display.image(image)
    display.show()

# ✅ Function to capture an image
def capture_image():
    img_path = "/home/ansh/Desktop/project/captured_image.jpg"
    
    if os.path.exists(img_path):
        os.remove(img_path)

    try:
        subprocess.run([
            "libcamera-still", "-o", img_path, "--width", "1920", "--height", "1080",
            "--quality", "95", "--shutter", "10000", "--gain", "1.5", "-t", "2000"
        ], check=True)
        
        time.sleep(1)  # Allow time for the image to be saved
        
        if not os.path.exists(img_path):
            raise FileNotFoundError("❌ Image capture failed: File not created.")

        print(f"✅ Image captured and saved as {img_path}")
        return img_path

    except Exception as e:
        print(f"❌ Error capturing image: {e}")
        return None

# ✅ Function to detect if a leaf is present
def detect_leaf(img_path):
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green color range (for leaves)
    lower_green = np.array([35, 40, 20])  
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    leaf_pixels = cv2.countNonZero(mask)

    if leaf_pixels > 5000:  # Threshold for detecting a leaf
        print("🍃 Leaf detected!")
        return True
    else:
        print("🚫 No leaf detected. Skipping prediction.")
        return False

# ✅ Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # Enhance sharpness & contrast
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.5)

    img = img.resize((224, 224))  # Resize for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ✅ Function to predict the class of the image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100  
    predicted_label = class_labels[predicted_class]

    return predicted_label, confidence

# ✅ Function to display prediction on OLED
def display_on_oled(prediction, confidence):
    display.fill(0)
    font = ImageFont.load_default()
    image = Image.new('1', (display.width, display.height))
    draw = ImageDraw.Draw(image)

    draw.text((5, 15), f"Detected:", font=font, fill=255)
    draw.text((5, 30), prediction, font=font, fill=255)
    draw.text((5, 45), f"{confidence:.1f}%", font=font, fill=255)

    display.image(image)
    display.show()
    time.sleep(2)

# ✅ Function to check button press
def wait_for_button_press():
    show_message_on_oled("Press Button")  # Show "Press Button" when idle
    
    while True:
        button_state = GPIO.input(BUTTON_PIN)

        if button_state == GPIO.LOW:  
            time.sleep(0.1)  # Debounce
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Confirm button is still pressed
                return True  
        time.sleep(0.05)  

# ✅ Main loop
print("🔵 System Ready: Press the button to capture and predict...")

try:
    while True:
        wait_for_button_press()  

        print("📸 Button Pressed! Capturing Image...")
        show_message_on_oled("Capturing...")  # Show capturing message
        img_path = capture_image()

        if img_path:
            print("🔍 Checking for leaf...")
            show_message_on_oled("Checking Leaf...")  # Show checking leaf message
            
            if detect_leaf(img_path):  
                print("🔍 Processing Image...")
                show_message_on_oled("Predicting...")  # Show predicting message
                
                predicted_label, confidence = predict_image(img_path)
                print(f'✅ Predicted: {predicted_label} ({confidence:.1f}%)')

                display_on_oled(predicted_label, confidence)
            else:
                print("❌ No leaf detected. Skipping classification.")
                show_message_on_oled("No Leaf")  # Show "No Leaf" message

        else:
            print("❌ Image capture failed.")
            show_message_on_oled("Capture Failed")  # Show failure message

        time.sleep(2)  # Debounce delay

except KeyboardInterrupt:
    print("\n🔴 Exiting...")
    GPIO.cleanup()
