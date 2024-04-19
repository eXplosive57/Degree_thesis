import  torch                           as T
import  torch.nn.functional             as F
import  numpy                           as np
import  math
import  torch.nn                        as nn
from    utilities                       import print_dict, print_list, expand_encoding, convTranspose2d_shapes, get_inputConfig, \
                                            showImage, trans_input_base, alpha_blend_pytorch, include_attention
                                            
from    torchsummary                    import summary
import  timm
from    torchvision                     import transforms
from    PIL                             import Image
from    torchvision                     import models
from    torchvision.models              import ResNet50_Weights, ViT_B_16_Weights

# input settigs:
config = get_inputConfig()

INPUT_WIDTH     = config['width']
INPUT_HEIGHT    = config['height']
INPUT_CHANNELS  = config['channels']


class Project_DFD_model(nn.Module):
    
    def __init__(self, c,h,w, n_classes):
        super(Project_DFD_model,self).__init__()
        print("Initializing {} ...".format(self.__class__.__name__))
        
        # initialize the model to None        
        self.input_dim  = (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
        
        # features, classes and spatial dimensions
        self.n_classes      = n_classes
        self.n_channels     = c
        self.width          = h
        self.height         = w
        self.input_shape    = (self.n_channels, self.height, self.width)   # input_shape doesn't consider the batch

    
    def getSummary(self, input_shape = None, verbose = True):  #shape: color,width,height
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
            
        try:
            model_stats = summary(self, input_shape, verbose = int(verbose))
            return str(model_stats)
        except Exception as e:
            # print(e)
            summ = ""
            n_params = 0
            for k,v in self.getLayers().items():
                summ += "{:<50} -> {:<50}".format(k,str(tuple(v.shape))) + "\n"
                n_params += T.numel(v)
            summ += "Total number of parameters: {}\n".format(n_params)
            if verbose: print(summ)
            return summ
    
    def getLayers(self):
        return dict(self.named_parameters())
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    def getDevice(self):
        return next(self.parameters()).device
    
    def isCuda(self):
        return next(self.parameters()).is_cuda
    
    def to_device(self, device):   # alias for to(device) function of nn.Module
        self.to(device)
        
    def _init_weights_kaimingNormal(self):
        # Initialize the weights  using He initialization (good for conv net)
        print("Weights initialization using kaiming Normal")
               
        for param in self.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def _init_weights_normal(self):
        print(f"Weights initialization using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
    
    def _init_weights_kaimingNormal_module(self, model = None):
        # Initialize the weights  using He initialization
        print("Weights initialization using kaiming Normal")
        
        if model is None: model = self
        
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def _init_weights_normal_module(self, model = None):
        # Initialize the weights with Gaussian distribution
        print(f"Weights initialization using Gaussian distribution")
        
        if model is None: model = self
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
    
    def getAttributes(self):
        att = self.__dict__
        
        def valid(pair):
            # unzip pair k,v
            _, x = pair 
            type_x = type(x)
            condition =  (type_x is int) or (type_x is str) or (type_x is tuple)
            return condition
        
        filterd_att = dict(filter(valid, att.items()))
        return filterd_att
    
    def forward(self):
        raise NotImplementedError


class VAE(Project_DFD_model):
    def __init__(self, input_channels=1, image_size=224, latent_dim=512):
        super(VAE, self).__init__(c = 1, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * (image_size // 32) * (image_size // 32), latent_dim * 2)  # Two times latent_dim for mean and logvar
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * (image_size // 32) * (image_size // 32)),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, image_size // 32, image_size // 32)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # self._init_weights_normal()

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = T.chunk(x, 2, dim=1)  # Split into mean and logvar
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = T.randn_like(var).to(self.device)      
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def loss_function(self,recon_x, x, mu, logvar, rec_loss:str = "bce", kld_loss:str = "sum"):
        """ 
            rec_loss (str, optional): select the reconstruction loss between: "bce","mse" and "mae". Defaults to "bce".
            kld_loss (str, optional): select the KL divergence loss aggregation modality between: "sum" and "mean". Defaults to "sum"
        """
        # use_bce = True, use_mse = False, KLD_mean = False
        
        # reconstruction loss
        if rec_loss == "bce":
            rec_loss =  nn.BCELoss(reduction='sum')(recon_x, x)
        elif rec_loss == "mse":
            rec_loss = nn.functional.mse_loss(recon_x, x)
        elif rec_loss == "mae":
            rec_loss = nn.functional.l1_loss(recon_x, x)
        
        # KL divergece loss   
        if kld_loss == "mean":
            KLD = -0.5 * T.mean(1 + logvar - mu.pow(2) - logvar.exp())
        elif kld_loss == "sum":
            KLD = -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
        return rec_loss + KLD
    
    def forward(self, x, train = False):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        
        if train:
            return x_hat,  mean, logvar
        else:
            return x_hat



class ViT_timm_EA(Project_DFD_model):
    def __init__(self, n_classes = 10, dropout = 0.1, prog_model = 3, encoding_type = "mean", resize_att_map = True,\
                 interpolation_mode = "bilinear", only_transfNorm = False, image_size = 224):   # bilinear
        """_summary_

        Args:
            num_classes (int, optional): _description_. Defaults to 10.
            dropout (int, optional): dropout rate used in attention and MLP layers. Defaults to 0.
            prog_model (int, optional): progressive id to select the model (use getModels for the complete list)
            resize_att_map(boolean, optinal): select if output attention map should have same dimension of input images, 
            if false patch_size is used when attention map of cls token is True, or lenght of tokens-1,for the attention map of 
            the remaining tokens. Defaults to True.
            interpolation_mode(boolean, optinal): choose the algorithm used to compute interpolation among: [bilinear, bicubic, area]. Defaults to "bilinear"
        """
        super(ViT_timm_EA, self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        self.models_avaiable = [
            'vit_base_patch16_224',
            'vit_base_patch16_224.augreg_in21k',
            'deit_small_distilled_patch16_224',
            'deit_tiny_distilled_patch16_224',
            'deit_base_distilled_patch16_224'
            ]
        self.name = self.models_avaiable[prog_model]
        print(f"Model architecture select in ViT_EA: {self.name}")
        
        self.model_vit = timm.create_model(model_name=self.name, pretrained=True, num_classes=n_classes, drop_rate=dropout)
        """
            Other ViT timm model parameters:
            num_classes: Mumber of classes for classification head. (int)
            global_pool: Type of global pooling for final sequence (String), default is 'token'. Choose btw: 'avg', 'token', 'map'
            class_token: Use class token (boolean), defaults is True
            drop_rate: Head dropout rate. (float), defaults is 0.
            pos_drop_rate: Position embedding dropout rate.(float), defaults is 0.
            attn_drop_rate: Attention dropout rate. (float), defaults is 0.
        """
        
        
        # data trasnformation
        try:
            data_config = timm.data.resolve_model_data_config(self.model_vit.pretrained_cfg)
            print("Transformation ops from pre-trained model:")
            print_dict(data_config)
            transform_pretrained = timm.data.create_transform(**data_config)
            if only_transfNorm:
                for t in transform_pretrained.transforms:
                    if isinstance(t, transforms.Normalize):
                        self.transform = t
            else:
                self.transform = transforms.Compose([t for t in transform_pretrained.transforms if not isinstance(t, transforms.ToTensor)])
        except:
            print("Not found transformation for the input use by pre-trained model")
            self.transform = None
        
        print("Transformation ops selected: ", self.transform)

        self.prog_model             = prog_model
        self.resize_att_map         = resize_att_map
        self.interpolation_mode     = interpolation_mode
        self.image_size             = image_size
        
        if prog_model in [0,1]:
            self.emb_size               = 768 
            self.mlp_dim                = 3072
            self.patch_size             = 16
            self.n_heads                = 12
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
        elif prog_model == 2:
            self.emb_size               = 384 
            self.mlp_dim                = "empty"
            self.patch_size             = 16
            self.n_heads                = 6
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
            
        elif prog_model == 3: 
            self.emb_size               = 192 
            self.mlp_dim                = "empty"
            self.patch_size             = 16
            self.n_heads                = 3
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
        elif prog_model == 4:
            self.emb_size               = 768 
            self.mlp_dim                = "empty"
            self.patch_size             = 16
            self.n_heads                = 12
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
        else:
            self.emb_size               = "empty"
            self.mlp_dim                = "empty"
            self.patch_size             = "empty"
            self.n_heads                = "empty"
            self.n_layers               = "empty"
            self.dim_head               = "empty"
            
        
        self.dropout                = dropout
        self.encoding_type          = encoding_type
        
        # get model parts
        self.embedding  = self.model_vit.patch_embed
        self.encoder    = self.model_vit.blocks
        self.head       = self.model_vit.head    

        # wrapper for attention extraction 
        self.wrapper_prog = 0
        
        if prog_model in [2,3,4]:   # Deit architecture
            self.model_vit.blocks[-1].attn.forward = self.forward_wrapper_1(self.model_vit.blocks[-1].attn)
        else:                       # classic ViT architecture
            if self.wrapper_prog == 0:
                self.model_vit.blocks[-1].attn.forward = self.forward_wrapper(self.model_vit.blocks[-1].attn)
            elif self.wrapper_prog == 1:
                self.model_vit.blocks[-1].attn.forward = self.forward_wrapper_1(self.model_vit.blocks[-1].attn)
        
    def getModels(self):
        print_list(self.models_avaiable)
    
    # reference: https://github.com/lcultrera/WildCapture/
    
    def forward_wrapper(self, attn_obj):
        def forward_wrap(x):
            # get batch, number of elements in the sequence (patches + cls token) and latent representation dims 
            B, N, C = x.shape    # on last layer the input channels has dim self.emb_size (768)
            
            # compute the embedding size for each head
            head_emb_size = C // attn_obj.num_heads

            # get the 3 different features vector for q, k and v (3 in 3rd dimension stands for this) for each head
            # and change dimensions order to: qkv dim, batch, head_dim, sequence_dim, head_embedding
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, head_emb_size).permute(2, 0, 3, 1, 4)
            
            # remove qkv dimension returning the 3 different vectors
            q, k, v = qkv.unbind(dim = 0)

            # print(q.shape)
            # compute the matrix calculus for attention
            
            # first transposition of key vector between sequence_dim, head_embedding, to make matrix multiplication feasible
            k_T = k.transpose(-2, -1)
            
            # matmul + scaling
            attn = (q @ k_T) * attn_obj.scale
            
            # apply softmax over last dimension
            attn = attn.softmax(dim=-1)
            
            # apply dropout
            attn = attn_obj.attn_drop(attn)
            
            # print("att full", attn.shape)
            
            # save the full attention map (used if self.use_attnmap_cls is False)
            attn_obj.attn_map = attn
            
            # get attention map for [cls] token and save, dropping first element in last dimension since is relative to token and not to patches (used if self.use_attnmap_cls is True)
            # attn_obj.cls_attn_map = attn[:, :, 0, 2:]
            attn_obj.cls_attn_map = attn[:, :, 0, 1:]
            
            # compute the remaining operations for the attention forward

            # matmul + exchange of dim between head_dim and sequence_dim
            x = (attn @ v).transpose(1, 2)
            
            # collapse head dim, and head_embedding in a single dimension
            x = x.reshape(B, N, C)
            
            # apply ap and dropout
            x = attn_obj.proj(x)                        # linear activation fuction
            x = attn_obj.proj_drop(x)                 
            
            return x
        return forward_wrap
    
    def forward_wrapper_1(self, attn_obj):
        def forward_wrap(x):
            # get batch, number of elements in the sequence (patches + cls token) and latent representation dims 
            B, N, C = x.shape    # on last layer the input channels has dim self.emb_size (768)
            
            # compute the embedding size for each head
            head_emb_size = C // attn_obj.num_heads

            # get the 3 different features vector for q, k and v (3 in 3rd dimension stands for this) for each head
            # and change dimensions order to: qkv dim, batch, head_dim, sequence_dim, head_embedding
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, head_emb_size).permute(2, 0, 3, 1, 4)
            
            # remove qkv dimension returning the 3 different vectors
            q, k, v = qkv.unbind(dim = 0)

            # print(q.shape)
            # compute the matrix calculus for attention
            
            # first transposition of key vector between sequence_dim, head_embedding, to make matrix multiplication feasible
            k_T = k.transpose(-2, -1)
            
            # matmul + scaling
            attn = (q @ k_T) * attn_obj.scale
            
            # apply softmax over last dimension
            attn = attn.softmax(dim=-1)
            
            # apply dropout
            attn = attn_obj.attn_drop(attn)
            
            # print("att full", attn.shape)
            
            # save the full attention map (used if self.use_attnmap_cls is False)
            attn_obj.attn_map = attn
            
            # get attention map for [cls] token and save, dropping first and second element in last dimension since are relative to cls and distill and not to patches.
            attn_obj.cls_attn_map = attn[:, :, 0, 2:]
            
            # compute the remaining operations for the attention forward

            # matmul + exchange of dim between head_dim and sequence_dim
            x = (attn @ v).transpose(1, 2)
            
            # collapse head dim, and head_embedding in a single dimension
            x = x.reshape(B, N, C)
            
            # apply ap and dropout
            x = attn_obj.proj(x)                        # linear activation fuction
            x = attn_obj.proj_drop(x)                 
            
            return x
        return forward_wrap
    
    def forward(self, x, verbose = False):
        
        # transform whether available
        if not(self.transform is None):
            x = self.transform(x)

        # print(x.shape)
        features    = self.model_vit.forward_features(x)   # output with the following semantic [cls token, distillation token, patches tokens]
        if verbose: print("features shape: ", features.shape)
        
        #                                       1) get encoding
        encoding = features.mean(dim = 1) if self.encoding_type == 'mean' else features[:, 0]  
        if verbose: print("encoding shape: ",encoding.shape)

        #                                       2) get logits
        logits      = self.model_vit.forward_head(features)         # using [cls] token embedding 
        if verbose: print("logits shape: ",logits.shape)

        #                                       3) get attention map
    
        # extract [cls] token attention map
        att_map_cls     = self.model_vit.blocks[-1].attn.cls_attn_map.mean(dim=1) # mean over heads results
        
        # if use 2nd wrapper, include a value to have a batch of vectors of size: patch_size**2
        if self.wrapper_prog == 1:
            extension_value = T.empty(att_map_cls.shape[0], 1)    # value to be added in the bathes of attention map
            extension_value[:, 0] = att_map_cls[:, 165]
            extension_value = extension_value.cuda()
            att_map_rest = T.cat((att_map_cls, extension_value), dim=1)
        
        att_map_cls     = att_map_cls.view(-1, 14, 14).detach()   # transform in images of dim: patch_size x patch_size
        att_map_cls     = att_map_cls.unsqueeze(dim = 1)          # add channel (grayscale) dim
        
        
        # extract patches tokens attention map
        att_map_rest     = self.model_vit.blocks[-1].attn.attn_map.mean(dim=1).detach()
        # print(att_map_rest.shape)
        
        if self.prog_model in [2,3,4]:
            att_map_rest     = att_map_rest[:, 2:, 2:].view(-1,1,196,196)       # exclude cls and distillation
            # att_map_rest     = att_map_rest[:, 1:, 1:].view(-1,1,197,197)     # exclude cls
            # att_map_rest     = att_map_rest.view(-1,1,198,198)
        else:
            att_map_rest     = att_map_rest[:, 1:, 1:].view(-1,1,196,196)       # exclude cls
        
    
        if self.resize_att_map:
            att_map_cls     = F.interpolate(att_map_cls,  (self.image_size, self.image_size),   mode = self.interpolation_mode)
            att_map_rest    = F.interpolate(att_map_rest, (self.image_size, self.image_size),   mode = self.interpolation_mode)
            
        return logits, encoding, att_map_cls, att_map_rest


class AutoEncoder(Project_DFD_model):
    def __init__(self):
        super(AutoEncoder, self).__init__(c = 1, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)
        self.flc = 32
        self.zdim = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        # self.sigmoid = nn.Sigmoid()

        self.encoder.add_module("final_convs", nn.Sequential(
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*4, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.zdim, kernel_size=8, stride=1, padding=0)
        ))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.zdim, self.flc, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*4, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*2, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.flc, 1, kernel_size=4, stride=2, padding=1)
            # nn.Sigmoid()
        )
        
        self._init_weights_kaimingNormal()

    def forward(self, x):
        x1 = self.encoder(x)
        # print(x1.shape)
        x2 = self.decoder(x1)
        # print(x2.shape)
        return x2


#                                       ResNet 50 ImageNet
class ResNet_ImageNet(Project_DFD_model):   # not nn.Module subclass, but still implement forward method calling the one of the model
    """ 
    This is a wrap class for pretraiend Resnet use the getModel function to get the nn.module implementation.
    The model expects color images in RGB standard, of size 244x244
    """
    
    
    def __init__(self, n_classes = 10):
        super(ResNet_ImageNet,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)

        self.weight_name =  ResNet50_Weights.IMAGENET1K_V2  # weights with accuracy 80.858% on ImageNet 
        self.model = self._create_net()
        
    def _create_net(self):
        model = models.resnet50(weights= self.weight_name)
        
        # edit first layer to accept grayscale images
        if self.n_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # edit fully connected layer for the output
        model.fc = nn.Linear(model.fc.in_features, self.n_classes)
        return model
        
    def getModel(self):
        return self.model
    
    def getSummary(self, input_shape = None, verbose = True):  #shape: color,width,height
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
        
        model_stats = summary(self.model, input_shape, verbose = int(verbose))
        return str(model_stats)
     
    def to(self, device):
        self.model.to(device)
    
    def getLayers(self):
        return dict(self.model.named_parameters())
    
    def freeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    def isCuda(self):
        return next(self.model.parameters()).is_cuda
         
    def forward(self, x):
        x = self.model(x)
        return x