from torchvision import models
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
import numpy as np

# Apply the transformations needed
import torchvision.transforms as T

# Define the helper function
def decode_segmap(image, source, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
    
  rgb = np.stack([r, g, b], axis=2)
  
  # Resize the input image to match the RGB output map
  source = source.resize((r.shape[1],r.shape[0]))
  
  # Apply a Gaussian blur Filter on the same input image and save a copy
  blurredImage = source.filter(ImageFilter.GaussianBlur(15))
  
  # Load the input image into the array  for looping through it
  pixels = source.load() 

  # Load the blurred image into the array for looping through it
  bgpixels = blurredImage.load()

  # Create an output image which will be have the required foreground and background pixel data
  out = Image.new('RGB', (rgb.shape[1], rgb.shape[0]))
  
  # Load the RGB output map in pixels array for looping
  pix = Image.fromarray(rgb)

  # We need width and height before we can loop through any array
  width, height = rgb.shape[1], rgb.shape[0]
  

  for x in range(width):
      for y in range(height):
          # Get separate RGB bands and compare them to see if its a foreground pixel
          r,g,b = pix.getpixel((x,y))          
          if r > 0 and g > 0 and b > 0:
              # If its foreground get the corresponding pixel at x,y location in the input image
              out.putpixel((x,y), pixels[x,y])
          else:
              # If its background get the corresponding pixel at x,y location in the blurred image
              out.putpixel((x,y), bgpixels[x,y])

  return out

def segment(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(450), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  
  rgb = decode_segmap(om, img)
    
  plt.imshow(rgb); plt.axis('off'); plt.show()
  

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

segment(dlab, './images/blur/girl.png', show_orig=False)
segment(dlab, './images/blur/boy.png', show_orig=False)