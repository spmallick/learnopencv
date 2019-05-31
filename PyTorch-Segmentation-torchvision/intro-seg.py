# <markdowncell>
### What is Segmentation?

# <markdowncell>
### Applications of Segmentation

# <markdowncell>
### What is mIoU?

# <markdowncell>
### Using torchvision for Semantic Segmentation

# <markdowncell>
#### Load FCN - Fully Convolutional Neural Networks.
# <codecell>
from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=1)

# <markdowncell>
#### Get an image
# <codecell>
from PIL import Image
!wget [link] -o r.png
img = Image.open('./r.png')

# Apply the transforms
import torchvision.transforms as T
trf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

# Pass the input through the net
out = net(inp)['out']
om = torch.argmax(out.squeeze(), dim=0)

# <markdowncell>
#### Helper function to turn segmentation maps to RGB images.
# <codecell>
# Define the helper function
def decode_segmap(image, nc=21):
  
  label_colours = np.array([(0, 0, 0),  # 0=background
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
    r[image == l] = label_colours[l, 0]
    g[image == l] = label_colours[l, 1]
    b[image == l] = label_colours[l, 2]

  rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
  rgb[:, :, 0] = r
  rgb[:, :, 1] = g
  rgb[:, :, 2] = b
  return rgb
