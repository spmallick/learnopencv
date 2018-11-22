def modifyScript(batchScript):
	f_write = open("finalScript.bat",'w')
	with open(batchScript,'r') as f:
		for line in f.readlines():
			stringToWrite=line.replace('\\','/').strip()+"\n"
			f_write.write(stringToWrite)
	f_write.close()

if __name__ == "__main__":
	modifyScript('runScript.bat')