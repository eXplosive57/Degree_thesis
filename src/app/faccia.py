import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definisci la struttura del modello
input_shape = (28, 28)
num_classes = 10

# Definisci la classe del modello
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Crea un'istanza del modello
model = MyModel()

# Cartella contenente le foto da analizzare
local_photos_dir = 'foto'

# Cartella dei log
log_dir = os.path.join(local_photos_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Funzione per analizzare un'immagine
def analyze_image(image, filename):
    # Ridimensiona l'immagine
    image = cv2.resize(image, input_shape)
    image = image.astype(np.float32) / 255.0  # Normalizza i valori dei pixel

    # Effettua la predizione
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        predictions = model(image_tensor).numpy()
    
    # Salva il log
    log_path = os.path.join(log_dir, f"log_{filename.split('.')[0]}.txt")
    with open(log_path, 'w') as f:
        f.write(f"Predictions: {predictions.tolist()}")

    return predictions.tolist()

# Carica i pesi dal file di checkpoint
checkpoint_path = 'faces/40.ckpt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Loop attraverso le immagini nella cartella locale
for filename in os.listdir(local_photos_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Assicurati di leggere solo file immagine
        image_path = os.path.join(local_photos_dir, filename)
        # Carica l'immagine
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Analizza l'immagine
            predictions = analyze_image(image, filename)
            print(f"Predictions for {filename}: {predictions}")
        else:
            print(f"Unable to read {filename}.")
