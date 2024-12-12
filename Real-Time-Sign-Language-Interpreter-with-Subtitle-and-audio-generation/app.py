# from flask import Flask, render_template, Response
# from scripts.inference_classifier import GestureClassifier
# import cv2

# app = Flask(__name__)
# gesture_classifier = GestureClassifier()
# camera = cv2.VideoCapture(0)


# @app.route("/")
# def index():
#     return render_template("index.html")


# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Perform gesture classification using your GestureClassifier
#         predicted_character, frame = gesture_classifier.predict(frame)

#         # Convert the frame to JPEG format
#         ret, jpeg = cv2.imencode(".jpg", frame)
#         frame_bytes = jpeg.tobytes()

#         # Yield the frame for streaming
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
#         )


# @app.route("/video_feed")
# def video_feed():
#     return Response(
#         generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     )


# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, Response
from scripts.inference_classifier import GestureClassifier
import cv2
import pyttsx3
import threading
import logging

app = Flask(__name__)
gesture_classifier = GestureClassifier()
camera = cv2.VideoCapture(0)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize text-to-speech engine
engine = None
engine_lock = threading.Lock()

def initialize_engine():
    global engine
    with engine_lock:
        if not engine:
            engine = pyttsx3.init()

def speak(text):
    initialize_engine()
    try:
        with engine_lock:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        logging.error(f"Error occurred during speech synthesis: {e}")

# Wrapper to run the speak function in a new thread
def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        try:
            # Perform gesture classification using your GestureClassifier
            predicted_character, frame = gesture_classifier.predict(frame)

            # If a gesture is recognized, convert it to speech
            if predicted_character:
                speak_async(predicted_character)

            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()

            # Yield the frame for streaming
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
            )
        except Exception as e:
            logging.error(f"Error in frame generation or speech synthesis: {e}")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
