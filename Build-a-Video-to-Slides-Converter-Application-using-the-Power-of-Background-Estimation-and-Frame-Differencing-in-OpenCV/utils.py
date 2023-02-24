import os
import cv2
import shutil
import img2pdf
import glob

# PIL can also be used to convert the image set into PDFs.
# However, using PIL requires opening each of the images in the set.
# Hence img2pdf package was used, which is able to convert the entire image set into a PDF
# without opening at once.

def resize_image_frame(frame, resize_width):

    ht, wd, _ = frame.shape
    new_height = resize_width * ht / wd
    frame = cv2.resize(frame, (resize_width, int(new_height)), interpolation=cv2.INTER_AREA)

    return frame


def create_output_directory(video_path, output_path, type_bgsub):
    
    vid_file_name = video_path.rsplit('/')[-1].split('.')[0]
    output_dir_path = os.path.join(output_path, vid_file_name, type_bgsub)

    # Remove the output directory if there is already one.
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)

    # Create output directory.
    os.makedirs(output_dir_path, exist_ok=True)
    print('Output directory created...')
    print('Path:', output_dir_path)
    print('***'*10,'\n')
    
    return output_dir_path


def convert_slides_to_pdf(video_path, output_path):

    pdf_file_name = video_path.rsplit('/')[-1].split('.')[0]+'.pdf'
    output_pdf_path = os.path.join(output_path, pdf_file_name)

    print('Output PDF Path:', output_pdf_path)
    print('Converting captured slide images to PDF...')

    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_path}/*.png"))))
 
    print('PDF Created!')
    print('***'*10,'\n')