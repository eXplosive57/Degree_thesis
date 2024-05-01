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
import base64
from flask_socketio import SocketIO, emit
import time
import numpy as np
from PIL import Image
from resnet50nodown import resnet50nodown
from typing import List, Dict


app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le rotte

socketio = SocketIO(app, cors_allowed_origins="*")


analyzed_photos_dir = 'analyzed_photos'

analyzed_photos_dic = {}  # Dizionario per mantenere i dati delle immagini analizzate

result_dic = {}

fake_dic = {}
foto = False
videocheck = False


@socketio.on('new_photo_analyzed')
def handle_new_photo_analyzed():
    # Invia un segnale ai client quando viene analizzata una nuova foto
    socketio.emit('photo_analyzed_notification', namespace='/')


@app.route('/photo_list')
def get_results():

        combined_results = {**analyzed_photos_dic, **result_dic, **fake_dic}
        
        return jsonify(list(combined_results.values()))



     
    

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
    file_name = request.form['nome']

    # Get the file extension
    file_extension = os.path.splitext(uploaded_file.filename)[1]

    if selected_model == 'object':
        if file_extension.lower() in ('.jpg', '.jpeg', '.png', '.gif'):
            print('ok')
            return analyze_photo(uploaded_file, file_name)
        else:
            # aggiungi check formati video
            return analyze_video(uploaded_file, file_name)
    elif selected_model == 'fake':
        return analyze_fake(uploaded_file, file_name)
    else:
        return 'Invalid model selected', 400




def analyze_photo(photo, nome_file):
    
    global foto

    # Read the image
    image = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Analyze the image using the YOLO model
    results = model.predict(image)

    # Annotate the image with identifying rectangles
    annotator = Annotator(image)
    for res in results:
        for box in res.boxes:
            box_xy = box.xyxy[0]  # Get the bounding box coordinates
            cls = box.cls  # Class index
            annotator.box_label(box_xy, model.names[int(cls)])

    # Get the annotated image
    annotated_image = annotator.result()

    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Add image data to analyzed_photos dictionary
    analyzed_photos_dic[nome_file] = {
        "file_name": nome_file,
        "anteprima": f"data:image/jpeg;base64,{encoded_image}"
    }
    foto = True
    handle_new_photo_analyzed()
    # Return success message
    return 'Image analyzed successfully', 200







def analyze_video(video, nome_file):

    global videocheck
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

    frame_output_path = os.path.join(analyzed_videos_dir, f"frame_{nome_file}.jpg")

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

            # Chiudi il video di input
    video_capture.release()

    # Chiudi il video di output
    out.release()

    # Apri il video analizzato
    video_capture_analyzed = cv2.VideoCapture(output_path)

    # Imposta la posizione dei frame nel video analizzato
    video_capture_analyzed.set(cv2.CAP_PROP_POS_FRAMES, 2)

    # Estrai il frame desiderato dal video analizzato
    ret, frame_to_send = video_capture_analyzed.read()
    if not ret:
        print("Errore nell'estrazione del frame")
        return None, None

    # Codifica il frame come immagine JPEG per l'invio al client
    _, frame_encoded = cv2.imencode('.jpg', frame_to_send)
    frame_base64 = base64.b64encode(frame_encoded).decode('utf-8')

    cv2.imwrite(frame_output_path, frame_to_send)
    # Rilascia le risorse
    video_capture.release()
    out.release()
    
    # PROVA A NON CODIFICARE E INVIA IL VIDEO 
    # Codifica il video analizzato come base64 per l'invio al client
    with open(output_path, 'rb') as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

    # Combine video base64 and frame base64 into a dictionary
    result_dic[nome_file] = {
        "video": video_base64,
        "anteprima": f"data:image/jpeg;base64,{frame_base64}",
        "file_name" : nome_file
    }
    videocheck = True
    handle_new_photo_analyzed()

    return 'video analyzed successfully', 200


def analyze_fake(photo, nome_file):
    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        photo.save(temp_file.name)

    weights_path = ('./weights/gandetection_resnet50nodown_progan.pth')

    from torch.cuda import is_available as is_available_cuda
    device = 'cuda:0' if is_available_cuda() else 'cpu'
    net = resnet50nodown(device, weights_path)

    # Analyze the received image
    img = Image.open(temp_file.name).convert('RGB')
    img.load()
    logit = net.apply(img)

    # Determine if the image is fake or real
    state = 'Real' if logit < 0 else 'Fake'

     # Convert the image to base64
    with open(temp_file.name, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    fake_dic[nome_file] = {
        "state": state,
        "file_name": nome_file,
        "anteprima": f"data:image/jpeg;base64,{img_base64}"
        
    }
    handle_new_photo_analyzed()
    return 'Daje', 200


if __name__ == '__main__':
    socketio.run(app, debug=True)
