import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

cpu_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Download Class specific images from OpenImagesV4')
parser.add_argument("--mode", help="Dataset category - train, validation or test", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count*2)
parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)

args = parser.parse_args()

runMode = args.mode

threads = args.nthreads

classes = []
for className in args.classes.split(','):
    classes.append(className)

with open('./class-descriptions-boxable.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]:rows[0] for rows in reader}

subprocess.run(['rm', '-rf', runMode])
subprocess.run([ 'mkdir', runMode])

pool = ThreadPool(threads)
commands = []
cnt = 0

for ind in range(0, len(classes)):
    
    className = classes[ind]
    print("Class "+str(ind) + " : " + className)
    
    subprocess.run([ 'mkdir', runMode+'/'+className])

    commandStr = "grep "+dict_list[className] + " ./" + runMode + "-annotations-bbox.csv"
    class_annotations = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    class_annotations = class_annotations.splitlines()

    for line in class_annotations:

        lineParts = line.split(',')
        
        #IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
        if (args.occluded==0 and int(lineParts[8])>0):
            print("Skipped %s",lineParts[0])
            continue
        if (args.truncated==0 and int(lineParts[9])>0):
            print("Skipped %s",lineParts[0])
            continue
        if (args.groupOf==0 and int(lineParts[10])>0):
            print("Skipped %s",lineParts[0])
            continue
        if (args.depiction==0 and int(lineParts[11])>0):
            print("Skipped %s",lineParts[0])
            continue
        if (args.inside==0 and int(lineParts[12])>0):
            print("Skipped %s",lineParts[0])
            continue

        cnt = cnt + 1

        command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'+runMode+'/'+lineParts[0]+'.jpg '+ runMode+'/'+className+'/'+lineParts[0]+'.jpg'
        commands.append(command)
        
        with open('%s/%s/%s.txt'%(runMode,className,lineParts[0]),'a') as f:
            f.write(','.join([className, lineParts[4], lineParts[5], lineParts[6], lineParts[7]])+'\n')

print("Annotation Count : "+str(cnt))
commands = list(set(commands))
print("Number of images to be downloaded : "+str(len(commands)))

#list(tqdm(pool.imap(os.system, commands), total = len(commands) ))

pool.close()
pool.join()



