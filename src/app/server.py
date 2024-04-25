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
from flask_socketio import SocketIO, emit
import glob
import time
# get the file nam from uplaoded file
from werkzeug.utils import secure_filename
import argparse
from PIL import Image
from resnet50nodown import resnet50nodown


app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le rotte

socketio = SocketIO(app, cors_allowed_origins="*")


analyzed_photos_dir = 'analyzed_photos'


@socketio.on('new_photo_analyzed')
def handle_new_photo_analyzed():
    # Invia un segnale ai client quando viene analizzata una nuova foto
    socketio.emit('photo_analyzed_notification', namespace='/')


@app.route('/photo_list')
def get_photo_list():
    photo_data_list = []

    for filename in os.listdir(analyzed_photos_dir):
        if filename != ".DS_Store":
            with open(os.path.join(analyzed_photos_dir, filename), "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                photo_data_list.append({
                    "name": filename,  # Aggiungi il nome del file all'oggetto JSON
                    "imageUrl": f"data:image/jpeg;base64,{image_data}"
                })

    return jsonify(photo_data_list)


# # Load Hydra configuration file
# with open("yolo/config/config.yaml") as f:
#     cfg = OmegaConf.load(f)

# Load YOLO model
model = YOLO('model/yolov8n.pt')


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


@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if the file was uploaded
    if 'file' not in request.files:
        return 'No file uploadeddddd', 400

    # Get the file and selected model from the form
    uploaded_file = request.files['file']
    selected_model = request.form.get('model')

    # Get the file extension
    file_extension = os.path.splitext(uploaded_file.filename)[1]

    if selected_model == 'object':
        if file_extension.lower() in ('.jpg', '.jpeg', '.png', '.gif'):
            print('ok')
            return analyze_photo(uploaded_file)
        else:
            # aggiungi check formati video
            return analyze_video(uploaded_file)
    elif selected_model == 'fake':
        return analyze_fake(uploaded_file)
    else:
        return 'Invalid model selected', 400


def analyze_photo(photo):
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

    handle_new_photo_analyzed()

    # Return the analyzed image as a response to the client
    return send_file(analyzed_photo_path, mimetype='image/jpeg')


def analyze_video(video):
    # Salvare temporaneamente il video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video.read())
        temp_file_path = temp_file.name

    # Apri il video
    video_capture = cv2.VideoCapture(temp_file_path)

    # Controlla se l'apertura del video è avvenuta correttamente
    if not video_capture.isOpened():
        print("Errore nell'apertura del video")
        return None

    # Percorso completo del video di output nella cartella analyzed_video
    analyzed_videos_dir = 'analyzed_video'
    os.makedirs(analyzed_videos_dir, exist_ok=True)
    output_filename = f"analyzed_{video.filename}"
    output_path = os.path.join(analyzed_videos_dir, output_filename)

    # Imposta i codec e il frame rate del video di output
    codec = cv2.VideoWriter_fourcc(*'avc1')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Controlla se il frame rate è valido
    if fps <= 0:
        print("Frame rate non valido")
        return None

    # Crea il video di output
    out = cv2.VideoWriter(output_path, codec, fps, frame_size)

    # Controlla se il video di output è stato creato correttamente
    if not out.isOpened():
        print("Errore nella creazione del video di output")
        return None

    # Ciclo sui frame del video di input
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Analizza il frame con il modello YOLO
        results_list = model(frame)  # Ogni elemento è un oggetto Results

        # Ciclo su ogni elemento della lista
        for results in results_list:
            # Annotate the image with identifying rectangles and class names
            for box in results.boxes:
                box_xy = box.xyxy[0]  # Get the bounding box coordinates
                cls = box.cls  # Class index
                class_name = model.names[int(cls)]  # Get the class name
                cv2.rectangle(frame, (int(box_xy[0]), int(box_xy[1])), (int(
                    box_xy[2]), int(box_xy[3])), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(box_xy[0]), int(
                    box_xy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Salva il frame nel video di output
        out.write(frame)

    # Rilascia le risorse
    video_capture.release()
    out.release()
    handle_new_photo_analyzed()

    return 'Video analyzed successfully', 200


def analyze_fake(photo):
    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        photo.save(temp_file.name)

    image = cv2.imread(temp_file.name)

    print('prova')
    # parser = argparse.ArgumentParser(description="This script tests the network on an image folder and collects the results in a CSV file.",
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--weights_path', '-m', type=str,
    #                     default='./weights/gandetection_resnet50nodown_progan.pth', help='weights path of the network')
    # parser.add_argument('--input_folder', '-i', type=str,
    #                     default='./misto', help='input folder with PNG and JPEG images')
    # parser.add_argument('--output_csv', '-o', type=str,
    #                     default=None, help='output CSV file')

    weights_path = ('./weights/gandetection_resnet50nodown_progan.pth')
    # output_csv = os.path.join(
    #     'output', f'out_{os.path.basename(temp_file.name)}.csv')

    # config = parser.parse_args()
    # weights_path = config.weights_path
    # input_folder = config.input_folder
    # output_csv = config.output_csv

    from torch.cuda import is_available as is_available_cuda
    device = 'cuda:0' if is_available_cuda() else 'cpu'
    net = resnet50nodown(device, weights_path)

    # Analyze the received image
    tic = time.time()
    img = Image.open(temp_file.name).convert('RGB')
    img.load()
    logit = net.apply(img)
    toc = time.time()

    # Determine if the image is fake or real
    state = 'real' if logit < 0 else 'fake'

    # # Write the results to the CSV file
    # with open(output_csv, 'w') as fid:
    #     fid.write('filename,logit,time,predict\n')
    #     fid.write('%s,%f,%f,%s\n' % (temp_file.name, logit, toc-tic, state))
    print('\nDONE')
    print(state)
    # print('OUTPUT: %s' % output_csv)
    handle_new_photo_analyzed()
    return 'Daje', 200


if __name__ == '__main__':
    socketio.run(app, debug=True)
