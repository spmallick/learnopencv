def modifyScript(batchScript):
	f_write = open("finalScript.bat",'w')
	with open(batchScript,'r') as f:
		count = 0
		for line in f.readlines():
			if line.strip()=="::DO_NOT_CHANGE::":
				count+=1
			if count%2==0:
				stringToWrite=line.replace('\\','/').strip()+"\n"
			else:
				stringToWrite=line.strip()+"\n"
			f_write.write(stringToWrite)
	f_write.close()

if __name__ == "__main__":
	modifyScript('runScript.bat')
