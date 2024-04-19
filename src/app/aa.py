from    torchvision                         import transforms
from    torchvision.transforms.functional   import InterpolationMode
from    PIL                                 import Image


spatial_dim = 224

def get_inputConfig():
    # choose between images of spatial dimensions 224p of 112p, always squared resolution
    return {
        "width"     : spatial_dim,
        "height"    : spatial_dim,
        "channels"  : 3
    }

concat_dict = lambda x,y: {**x, **y}

def print_dict(x):
    for idx,(k,v) in enumerate(x.items()):
        print("{:>3}) {:<15} -> {:<15}".format(str(idx),str(k),str(v)))

def print_list(x):
    for idx, elem in enumerate(x): 
        print("{:>3}){}".format(str(idx), str(elem)))


def trans_input_base(isTensor = False):
    """ function that returns transformation operations 

    Returns:
        compose pytorch object
    """
    
    config = get_inputConfig()
    w = config['width']
    h = config['height']
    
    
    if isTensor:
        transform_ops = transforms.Compose([
                        transforms.Resize((h, w), interpolation= InterpolationMode.BILINEAR, antialias= True),
                        lambda x: T.clamp(x, 0, 1)
                                                  ])
    else: 
        transform_ops = transforms.Compose([
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
            transforms.Resize((h, w), interpolation= InterpolationMode.BILINEAR, antialias= True),
        ])
    
    return transform_ops