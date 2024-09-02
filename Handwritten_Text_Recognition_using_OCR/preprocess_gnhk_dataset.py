import os
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm

def create_directories():
    dirs = [
        'input/gnhk_dataset/train_processed/images',
        'input/gnhk_dataset/test_processed/images',
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def polygon_to_bbox(polygon):
    points = np.array([(polygon[f'x{i}'], polygon[f'y{i}']) for i in range(4)], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    return x, y, w, h

def process_dataset(input_folder, output_folder, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_filename', 'text'])
        
        for filename in tqdm(os.listdir(input_folder), desc=f"Processing {os.path.basename(input_folder)}"):
            if filename.endswith('.json'):
                json_path = os.path.join(input_folder, filename)
                img_path = os.path.join(input_folder, filename.replace('.json', '.jpg'))
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                img = cv2.imread(img_path)
                
                for idx, item in enumerate(data):
                    text = item['text']
                    if text.startswith('%') and text.endswith('%'):
                        text = 'SPECIAL_CHARACTER'
                    
                    x, y, w, h = polygon_to_bbox(item['polygon'])
                    
                    cropped_img = img[y:y+h, x:x+w]
                    
                    output_filename = f"{filename.replace('.json', '')}_{idx}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_img)
                    
                    csv_writer.writerow([output_filename, text])
                    
def main():
    create_directories()
    
    process_dataset(
        'input/gnhk_dataset/train_data/train', 
        'input/gnhk_dataset/train_processed/images',
        'input/gnhk_dataset/train_processed.csv'
    )
    process_dataset(
        'input/gnhk_dataset/test_data/test', 
        'input/gnhk_dataset/test_processed/images',
        'input/gnhk_dataset/test_processed.csv'
    )

if __name__ == '__main__':
    main()