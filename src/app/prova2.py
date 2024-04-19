import os
import sys
import torch
from PIL import Image
from torchvision import transforms

try:
    # Add the project root directory to the system path
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

    from modelli import *
except:
    from modelli import *

# Definisci il percorso del file di checkpoint
checkpoint_path = "ckpt_folder/ViT_timm_EA_50epochs.ckpt"

# Carica l'immagine di esempio
image_path = "static/2.png"
image = Image.open(image_path)

# Definisci le trasformazioni per preparare l'immagine
image_transforms = transforms.Compose([
    # Assicurati che le dimensioni dell'immagine siano compatibili con il modello
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Applica le trasformazioni all'immagine
input_image = image_transforms(image).unsqueeze(
    0)  # Aggiungi una dimensione batch

# Crea un'istanza del modello ViT_timm_EA
model = ViT_timm_EA()

# Carica i pesi del modello dal file di checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Imposta il modello in modalità valutazione (non addestramento)
model.eval()

# Esegui l'inoltro (forward) dell'immagine attraverso il modello
with torch.no_grad():
    logits, encoding, att_map_cls, att_map_rest = model(input_image)

# Calcola le previsioni di classe
predicted_class = logits.argmax(dim=1)

# Stampa i risultati
print("Classe predetta:", predicted_class.item())
