import imagehash
from PIL import Image
import os


def find_similar_images(base_dir, hash_size=8):

    snapshots_files = sorted(os.listdir(base_dir))

    hash_dict = {}
    duplicates = []
    num_duplicates = 0

    print('---'*5,"Finding similar files",'---'*5)

    for file in snapshots_files:
        read_file = Image.open(os.path.join(base_dir, file))
        comp_hash = str(imagehash.dhash(read_file, hash_size=hash_size))

        if comp_hash not in hash_dict:
            hash_dict[comp_hash] = file
        else:
            print('Duplicate file: ', file)
            duplicates.append(file)
            num_duplicates+=1
    
    print('\nTotal duplicate files:', num_duplicates)
    print("-----"*10)
    return hash_dict, duplicates


def remove_duplicates(base_dir):

    _, duplicates = find_similar_images(base_dir, hash_size=10)

    if not len(duplicates):
        print('No duplicates found!')

    else:
        print("Removing duplicates...")

        for dup_file in duplicates:
            file_path = os.path.join(base_dir, dup_file)

            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print('Filepath: ', file_path, 'does not exists.')
        

        print('All duplicates removed!')
    
    print('***'*10,'\n')

if __name__ == "__main__":
    remove_duplicates('sample_1')