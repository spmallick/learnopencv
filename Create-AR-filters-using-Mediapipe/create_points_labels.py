
with open("points_labels.txt", 'w') as file:
    for x in range(68):
        file.writelines(str(x)+"\n")
