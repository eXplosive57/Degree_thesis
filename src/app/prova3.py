# from tkinter import Y
# from unittest import result
# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# # Analizza un video
# video_path = 'video/cars.mp4'  # Specifica il percorso del tuo video
# results = model(source=video_path, show=True, conf=0.4, save=True)

import cv2
from ultralytics import YOLO

# Carica il modello YOLO
model = YOLO('yolov8n.pt')

# Percorso del video di input
video_path = 'video/cars.mp4'  # Specifica il percorso del tuo video

# Percorso completo del video di output
output_path = 'video/cars_analyzed.mp4'

# Apri il video
video = cv2.VideoCapture(video_path)

# Imposta i codec e il frame rate del video di output
codec = cv2.VideoWriter_fourcc(*'avc1')
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Crea il video di output
out = cv2.VideoWriter(output_path, codec, fps, frame_size)

# Ciclo sui frame del video di input
while True:
    ret, frame = video.read()
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

    # # Mostra il frame con gli oggetti rilevati (opzionale)
    # cv2.imshow('Analyzed Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Rilascia le risorse
video.release()
out.release()
# cv2.destroyAllWindows()
