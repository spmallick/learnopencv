def main():
	print("Welcome to OpenCV Installation by LearnOpenCV")

def writeBatchFile(batchFile):
	newBatchFileName = batchFile[:-4]+"_modified.bat"
	f_write = open(newBatchFileName,'w')
	f_write.write("@echo off\n")
	f_write.write("setlocal enabledelayedexpansion\n")
	
	writeFlag=0
	with open(batchFile,'r') as f:
		for line in f.readlines():
			if line[:3]=='::x':
				writeFlag+=1
			writeFlag=writeFlag%2
			if writeFlag==0:
				stringToWrite=line.rstrip()+"\n"
				f_write.write(stringToWrite)
	f_write.write("::====================================::\n")
	
	f_write.write("echo @echo off>>runScript.bat\n")
	f_write.write("echo setlocal enabledelayedexpansion >> runScript.bat\n")
	writeFlag=0
	with open(batchFile,'r') as f:
		for line in f.readlines():
			if line[:3] == '::/':
				writeFlag+=1
			writeFlag=writeFlag%2
			if writeFlag==0:
				stringToWrite="echo "+line.rstrip()+" >> runScript.bat\n"
				f_write.write(stringToWrite)
	f_write.close()

if __name__=="__main__":
	cvVersionChoice=main()
	writeBatchFile("installOpenCV.bat")
