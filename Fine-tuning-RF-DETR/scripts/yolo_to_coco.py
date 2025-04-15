import json
import os
from PIL import Image

train_dir_images = "/home/opencv/bhomik_its_AI/RF-DETR/underwater_YOLO_dataset/aquarium_pretrain/valid/images"
train_dir_labels = "/home/opencv/bhomik_its_AI/RF-DETR/underwater_YOLO_dataset/aquarium_pretrain/valid/labels"
output_dir = "/home/opencv/bhomik_its_AI/RF-DETR/underwater_COCO_dataset"

categories = [{"id": 0, "name": 'fish', "supercategory": "animal"}, {"id": 1, "name": 'jellyfish', "supercategory": "animal"}, {"id": 2, "name": "penguin", "supercategory": "animal"}, {"id": 3, "name": "puffer_fish", "supercategory": "animal"}, {"id": 4, "name": "shark", "supercategory": "animal"}, {"id": 5, "name": "stingray", "supercategory": "animal"}, {"id": 6, "name": "starfish","supercategory": "animal"}]

coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": [],
}

annotation_id = 0
image_id_counter = 0
for image_fol in os.listdir(train_dir_images):
    # print(image_fol)
    image_path = os.path.join(train_dir_images, image_fol)
    image = Image.open(image_path)
    width, height = image.size

    image_id = image_id_counter
    image_dict = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": image_fol,
    }

    # print(image_dict['id'])

    coco_dataset["images"].append(image_dict)

    with open(os.path.join(train_dir_labels, f"{image_fol.split('.jpg')[0]}.txt")) as f:
        # print(f)
        annotations = f.readlines()

        for ann in annotations:
            category_id = int(ann[0])
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w/2)*width), int((y - h/2)*height)
            x_max, y_max = int((x + w/2)*width), int((y + h/2)*height)

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": bbox_height * bbox_width,
                "iscrowd": 0,
            }

            coco_dataset["annotations"].append(ann_dict)
            annotation_id += 1
        image_id_counter += 1

with open(os.path.join(output_dir, '_annotations.coco.json'), 'w') as f:
    json.dump(coco_dataset, f)