import os
import random
import numpy as np

list1=[]
for i in range(65, 65+26): # List of Characters
      list1.append(chr(i))

list2=[]
for j in range(48,48+10):  # List of digits
		list2.append(chr(j))

list3=list1+list2

def get_distort_arg():  # Function to create Distortion
  amount = 5
  hundred_minus_amount = 100 - amount
  return '\'0,0 ' + str(np.random.randint(0,amount)) + ',' + str(np.random.randint(0,amount)) + ' 100,0 '  + str(np.random.randint(hundred_minus_amount,100)) + ',' + str(np.random.randint(0,amount)) + ' 0,100 '  + str(np.random.randint(0,amount)) + ',' + str(np.random.randint(hundred_minus_amount,100)) + ' 100,100 '  + str(np.random.randint(hundred_minus_amount,100)) + ',' + str(np.random.randint(hundred_minus_amount,100)) + '\''

distort_arg = get_distort_arg()


blur_list = ['0x1','0x1','0x2','0x3','0x2'] # List of blur values
blur_e = random.choice(blur_list)
GN = random.randint(0,1) # Select figure randomly to generate noise

gravity = ['south','north','east'] # Assign gravity to place text
color_light = ['white','lime','gray','yellow','silver','aqua'] # List of light font colors
color_dark = ['black','green','maroon','blue','purple','red']  #  List of light dark colors

path1 = 'path to light backgrounds '
path2 = 'path to dark backgrounds'

# Command Required [convert image.jpg -resize 600x400\> image.jpg]

list_files_light=(os.listdir('path1'))  # To make list of light backgrounds
list_files_light = ['path1' + x for x in list_files_light]


list_files_fontt = (os.listdir('path of font file'))
list_files_fontt = ['path of font file'+ y for y in list_files_fontt] # To make list of fonts
  


list_files_dark = (os.listdir('path2'))  # To make list of dark backgrounds
list_files_dark = ['path2' + x for x in list_files_dark]


final_list = [list_files_dark, list_files_light] # Make list of paths of light and dark backgrounds


for k in range(0,len(list_files_light)):
		p1 = random.randint(0,300)  # Randomly select first co-ordinate of square for cropping image
		p2 = random.randint(0,300)
		command = "magick convert "+ str(list_files_light[k])+ " -crop 32x32"+"+"+str(p1)+"+"+str(p2)+" " + str(list_files_light[k])
		print(command)
		os.system(str(command))

for m in range(0,len(list_files_dark)):
		p1 = random.randint(0,200) # Randomly select first co-ordinate of square for cropping image
		p2 = random.randint(0,200)
		command = "magick convert "+ str(list_files_dark[m])+ " -crop 32x32"+"+"+str(p1)+"+"+str(p2)+" " + str(list_files_dark[m])
		#print(command)
		os.system(str(command))



# Sample Command-----  magick convert image.jpg -fill Black -font Courier-Oblique -weight 50 -pointsize 12 -gravity center -blur 0x8 -evaluate Gaussian-noise 1.2  -annotate 0+0 "Some text" output_image

for i in range(0,len(list3)):
	
	directory = "path to save images of each label"
	char = list3[i]
	directory = directory + str(char) + "/"
	
	if not os.path.exists(directory):
		os.makedirs(directory)
		print("Directory made")

	for j in range(0,1000): # To generate 1000 images for each character.
		gv = random.choice(gravity)
		path = random.choice(final_list)
		list_filernd = random.choice(path)
		list_rfo = random.choice(list_files_fontt)
		if( path == final_list[0]):
			color = random.choice(color_light)
			command =  "magick convert " + str(list_filernd) + " -fill "+str(color)+" -font "+ \
            str(list_rfo) + " -weight 200 -pointsize 24 -distort Perspective "+str(distort_arg)+" "+"-gravity "+str(gv) + " -blur " + str(blur_e) \
+ " -evaluate Gaussian-noise " + str(GN) +  " " + " -annotate +0+0 "+  str(list3[i]) + " " + directory + "output_file"+str(i)+str(j)+".jpg"
			print(command)
			os.system(str(command))
			

		elif(path == final_list[1]):
			color = random.choice(color_dark)
			command =  "magick convert " + str(list_filernd) + " -fill "+str(color)+" -font "+ \
            str(list_rfo) + " -weight 200 -pointsize 24 -distort Perspective "+str(distort_arg)+" "+"-gravity "+str(gv) + " -blur " + str(blur_e) \
+ " -evaluate Gaussian-noise " + str(GN) +  " " + " -annotate +0+0 "+  str(list3[i]) + " " + directory + "output_file"+str(i)+str(j)+".jpg"
			
			os.system(str(command))
			
