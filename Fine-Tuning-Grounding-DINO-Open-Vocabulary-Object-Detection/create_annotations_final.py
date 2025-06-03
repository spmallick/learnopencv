import os
import random
import shutil
import xml.etree.ElementTree as ET
import csv

# === CONFIG ===
ORIG_IMG_DIR        = 'images'
ORIG_ANN_DIR        = 'annotations'
TRAIN_DIR           = 'train'
TEST_DIR            = 'test'
NUM_TRAIN_IMAGES    = 600
CSV_OUTPUT_FILENAME = 'annotations_final.csv'   # will be placed under TRAIN_DIR

# === UTILITIES ===

def xmls_to_csv(annotations_dir: str, output_csv: str):
    """Parse all PascalVOC‐style XMLs in `annotations_dir` → single CSV."""
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'label_name',
            'bbox_x1','bbox_y1','bbox_x2','bbox_y2',
            'image_name',
            'image_width','image_height'
        ])

        for fname in os.listdir(annotations_dir):
            if not fname.lower().endswith('.xml'):
                continue
            xml_path = os.path.join(annotations_dir, fname)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_name = root.findtext('filename')
            size = root.find('size')
            width  = size.findtext('width')
            height = size.findtext('height')

            for obj in root.findall('object'):
                label = obj.findtext('name')
                b = obj.find('bndbox')
                xmin = b.findtext('xmin')
                ymin = b.findtext('ymin')
                xmax = b.findtext('xmax')
                ymax = b.findtext('ymax')

                writer.writerow([
                    label,
                    xmin, ymin, xmax, ymax,
                    image_name,
                    width, height
                ])
    print(f"[+] Wrote CSV annotations to {output_csv}")


def mkdirs_safe(path):
    os.makedirs(path, exist_ok=True)


# === MAIN ===

def main():
    # 1) Create train/ and test/ sub-dirs
    for split in (TRAIN_DIR, TEST_DIR):
        mkdirs_safe(os.path.join(split, 'images'))
        mkdirs_safe(os.path.join(split, 'labels'))

    # 2) Collect all image filenames (we assume .png here)
    all_images = [f for f in os.listdir(ORIG_IMG_DIR)
                  if f.lower().endswith(('.png','.jpg','.jpeg'))]
    random.shuffle(all_images)

    # 3) Split into train/test
    train_imgs = set(all_images[:NUM_TRAIN_IMAGES])
    test_imgs  = set(all_images[NUM_TRAIN_IMAGES:])

    # 4) Copy files
    for img_set, dest_split in [(train_imgs, TRAIN_DIR), (test_imgs, TEST_DIR)]:
        for img_name in img_set:
            # copy image
            src_img = os.path.join(ORIG_IMG_DIR, img_name)
            dst_img = os.path.join(dest_split, 'images', img_name)
            shutil.copy2(src_img, dst_img)

            # copy corresponding XML
            xml_name = os.path.splitext(img_name)[0] + '.xml'
            src_xml  = os.path.join(ORIG_ANN_DIR, xml_name)
            dst_xml  = os.path.join(dest_split, 'labels', xml_name)
            if os.path.exists(src_xml):
                shutil.copy2(src_xml, dst_xml)
            else:
                print(f"[!] Warning: annotation not found for {img_name}")

    print(f"[+] Copied {len(train_imgs)} images → {TRAIN_DIR}/images")
    print(f"[+] Copied {len(test_imgs)} images → {TEST_DIR}/images")

    # 5) Generate CSV from train/labels
    train_labels_dir = os.path.join(TRAIN_DIR, 'labels')
    csv_outpath      = os.path.join(TRAIN_DIR, CSV_OUTPUT_FILENAME)
    xmls_to_csv(train_labels_dir, csv_outpath)


if __name__ == '__main__':
    main()
