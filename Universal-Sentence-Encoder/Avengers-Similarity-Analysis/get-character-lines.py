chars = []
file_handles = {}
script = open("script-processed",'r')
for line in script.readlines():
    line.strip()
    if line=="":
        continue
    else:
        try:
            if ":" not in line:
                continue
            character = line.split(":")[0]
            newline = " ".join(line.split(":")[1:])
            if len(character.split())>2:
                print(line,character)
                input()
            #print(line,character)
            #input()
            if character=="":
                continue
            elif character in chars:
                file_handles[character].write("{}\n".format(newline.strip()))
            else:
                chars.append(character)
                file_handles[character] = open(character+".txt",'w')
                file_handles[character].write("{}\n".format(newline.strip()))
        except:
            print("Unknown error. Skipping line...")
#print("Files written for:\n{}".format("\n".join(chars)))
