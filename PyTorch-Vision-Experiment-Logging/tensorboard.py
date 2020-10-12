# ### OUR CODE ###
import os.path as osp
import tensorboardX

# create a new class inheriting from tensorboardX.SummaryWriter
class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        super().__init__(log_dir, comment, **kwargs)
    # create a new function that will take dictionary as input and uses built-in add_scalar() function
    # that function combines all plots into one subgroup by a tag
    def add_scalar_dict(self, dictionary, global_step, tag=None):
        for name, val in dictionary.items():
            if tag is not None:
                name = osp.join(tag, name)
            self.add_scalar(name, val.item(), global_step)

# COCO labels mapping
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# define variables as globals to have an access everywhere
args = None  # will be replaced to argparse dictionary
total_epochs = 0  # init total number of epochs
global_iter = 0  # init total number of iterations

name = "exp-000"  # init experiment name
log_dir = "experiments"  # init path where tensorboard logs will be stored
# (if log_dir is not specified writer object will automatically generate filename)
# Log files will be saved in 'experiments/exp-000'
# create our custom logger
logger = SummaryWriter(log_dir=osp.join(log_dir, name))

# ### END OF OUR CODE ###
