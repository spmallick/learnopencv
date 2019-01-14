from PIL import Image
import requests
import csv
import subprocess

def get_file(url, local_filename):
    attempts = 0
    while attempts < 3:
        try:
            r = requests.get(url)
            with open(local_filename, 'wb') as f:  
                    f.write(r.content)
            f.close()
            img = Image.open(local_filename)

            break
        except:
            attempts += 1
            print("Attempt#" + str(attempts) + " - Failed to read from "+url)

    try:
        img = Image.open(local_filename)
        img.verify()
    except Exception as e:
        print(e)
        print(local_filename + " corrupted..deleting it..")
        subprocess.run(['rm', local_filename])

    return

with open('snowmanDataLinks.csv', 'r') as csvfile:
#with open('snowmantestLinks.csv', 'r') as csvfile:
    urlreader = csv.reader(csvfile, delimiter=',')
    for row in urlreader:
        get_file(row[0], 'JPEGImages/' + row[1])
        print(row[0] + '-->' + row[1])
