import re
script_processed = open("script-processed",'w')
with open("script-raw",'r') as script_raw:
    for line in script_raw.readlines():
        x = re.sub("[\(\[].*?[\)\]]", "", line)
        x = x.strip()
        script_processed.write("{}\n".format(x))
script_processed.close()
