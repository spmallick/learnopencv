def main():
	print("Welcome to OpenCV Installation by LearnOpenCV")
	print("What OpenCV version would you like to install?")
	cvVersionChoice=input("1 for 3.4.1 (default) and 2 for Master branch: ")
	if cvVersionChoice != '2':
		cvVersionChoice='1'
	print(cvVersionChoice)
	return cvVersionChoice

def writeBatchFile(batchFile,cvVersionChoice):
	newBatchFileName = batchFile[:-4]+"_modified.bat"
	f_write = open(newBatchFileName,'w')
	f_write.write("@echo off\n")
	f_write.write("setlocal enabledelayedexpansion\n")
	cvVersionString = "set cvVersionChoice="+cvVersionChoice+"\n"
	f_write.write(cvVersionString)
	
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
	f_write.write("echo setlocal enabledelayedexpansion >> runScript.bat")
	cvVersionString = "echo set cvVersionChoice="+cvVersionChoice+" >> runScript.bat\n"
	f_write.write(cvVersionString)
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
	writeBatchFile("installOpenCV.bat",cvVersionChoice)