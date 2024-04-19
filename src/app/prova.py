import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

try:
     # Add the project root directory to the system path
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
     from modelli import * 
except:
     from modelli import * 

# model = ViT_timm_EA()
# checkpoint = torch.load('ckpt_folder/ViT_timm_EA_50epochs.ckpt', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)

# image_path = os.path.join('static', '1.png')
# image = Image.open(image_path)

# # Define transformations to prepare the image for the model
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize the image to the size expected by the model
#     transforms.ToTensor()            # Convert the image to a tensor
# ])

# # Apply transformations to the image
# input_image = transform(image)

# # Add a batch dimension to the image
# input_image = input_image.unsqueeze(0)

# # Pass the input image to the model
# output = model(input_image)
def test_ViTEA():
    # define test input
    # x = T.rand((32, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)

    img_1 = trans_input_base()(Image.open("/static/1.png"))
    img_2 = trans_input_base()(Image.open("/static/test_image_attention2.png"))
    img_2 = img_2[:-1]
    
    print(img_1.shape)
    print(img_2.shape)
    

    x = T.stack([img_1, img_2], dim=0).cuda()
    
    showImage(img_1)
    showImage(img_2)
    
    print(x.shape)
    
    tests = [0, 1, 0]  # vit_EA, encoder, vit_EA + encoder
    
    if tests[0]:
        vit = ViT_timm_EA(resize_att_map=True, prog_model=3).to(device = device)
        vit.getSummary()
        print(vit.getAttributes())
        
    
        out = vit.forward(x)
        logits          = out[0]
        encoding        = out[1]
        attention       = out[2]
        attention_full  = out[3]
        
        input("press something to continue")
        
        print("logits shape     -> ", logits.shape)
        print("encoding shape   -> ", encoding.shape)
        print("attention cls shape  -> ", attention.shape)
        print("attention full shape  -> ", attention_full.shape)

        print(T.sum(attention[0]))
        print(T.sum(attention[1]))
        
        vit.getSummary()
        print(vit.transform)
        print(attention[0].shape)
        
        att_map = attention[0]
        
        showImage(x[0])
        showImage(attention[0], has_color= False)
        showImage(attention_full[1], has_color= False)
        
        blend, att = include_attention(x[0], attention[0])
        
        print(blend.shape)
        print(att.shape)
        
        showImage(blend)
        showImage(att)
        

    elif tests[1]:
        # ae = AutoEncoder()
        ae = VAE()
        ae.getSummary()
        
        batch_example = T.rand(size=(32,1,INPUT_HEIGHT,INPUT_WIDTH)).to(device)
        ae.forward(batch_example)
        input("press something to continue")
        
    elif tests[2]:
        vit = ViT_timm_EA().to(device = device)
        ae  = AutoEncoder().to(device=device)
        vit.getSummary()
        print(vit.getAttributes())
        logits, encoding, attention = vit.forward(x)
        
        out = vit.forward(x)
        logits      = out[0]
        encoding    = out[1]
        attention   = out[2]
        
        print(out[0])
        print(attention.shape)
        
        rec = ae(attention)
        print(rec.shape)
        
        showImage(attention[0], has_color= False)
        showImage(rec[0], has_color= False)
    
    input("press enter to exit ")


if __name__ == "__main__":

    # setUp test
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    input_example = T.rand(size=(INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))
    batch_example = T.rand(size=(32,INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))