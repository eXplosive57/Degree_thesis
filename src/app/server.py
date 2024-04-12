import os
from flask import Flask, request, send_file, jsonify, send_from_directory
from omegaconf import OmegaConf
from pathlib import Path
from ultralytics import YOLO
import cv2
import tempfile
from ultralytics.utils.plotting import Annotator
from collections import Counter
from flask_cors import CORS
import uuid
import base64

app = Flask(__name__)
CORS(app)

analyzed_photos_dir = 'analyzed_photos'


@app.route('/photo_list')
def get_photo_list():
    # Ottieni l'elenco dei nomi delle immagini nella directory analyzed_photos,
    # escludendo il file .DS_Store se presente
    photo_data_list = []

    for filename in os.listdir(analyzed_photos_dir):
        if filename != ".DS_Store":
            with open(os.path.join(analyzed_photos_dir, filename), "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                photo_data_list.append(f"data:image/jpeg;base64,{image_data}")

    return jsonify(photo_data_list)


# Load Hydra configuration file
with open("yolo/config/config.yaml") as f:
    cfg = OmegaConf.load(f)

# Load YOLO model
model = YOLO(Path(cfg.model_path), task="detect")


def predicted_classes(boxes, class_names):
    """
    Extracts classes and corresponding frequencies.
    """
    # Extract predicted classes and corresponding frequencies
    classes_ids = [int(id.cls) for id in boxes]
    class_freq = Counter(classes_ids)

    # Convert class IDs to names
    pred_class_names = {}
    for c in class_freq:
        pred_class_names[class_names[c]] = class_freq[c]

    return pred_class_names


@app.route('/analyze_photo', methods=['POST'])
def analyze_photo():
    # Check if the file was uploaded
    if 'photo' not in request.files:
        return 'No file uploaded', 400

    # Get the image file from the form
    photo = request.files['photo']

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        photo.save(temp_file.name)

    # Read the image from the temporary file
    image = cv2.imread(temp_file.name)

    # Analyze the image using the YOLO model
    results = model.predict(image)

    # Delete the temporary file
    os.unlink(temp_file.name)

    # Annotate the image with identifying rectangles
    annotator = Annotator(image)
    for res in results:
        for box in res.boxes:
            box_xy = box.xyxy[0]  # Get the bounding box coordinates
            cls = box.cls  # Class index
            annotator.box_label(box_xy, model.names[int(cls)])

    # Get the annotated image
    annotated_image = annotator.result()

    # Get unique id for every image to save
    unique_id = uuid.uuid4().hex

    # Save the analyzed image to a folder with a unique name
    analyzed_photos_dir = 'analyzed_photos'
    os.makedirs(analyzed_photos_dir, exist_ok=True)
    analyzed_photo_path = os.path.join(
        analyzed_photos_dir, f'analyzed_photo_{unique_id}.jpg')
    cv2.imwrite(analyzed_photo_path, annotated_image)

    # Return the analyzed image as a response to the client
    return send_file(analyzed_photo_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
