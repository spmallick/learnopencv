.SUFFIXES: .class .java
FILES = libsvm/svm.class libsvm/svm_model.class libsvm/svm_node.class \
		libsvm/svm_parameter.class libsvm/svm_problem.class \
		libsvm/svm_print_interface.class \
		svm_train.class svm_predict.class svm_toy.class svm_scale.class

#JAVAC = jikes
JAVAC_FLAGS = -target 1.7 -source 1.7
JAVAC = javac
# JAVAC_FLAGS =
export CLASSPATH := .:$(CLASSPATH)

all: $(FILES)
	jar cvf libsvm.jar *.class libsvm/*.class

.java.class:
	$(JAVAC) $(JAVAC_FLAGS) $<

libsvm/svm.java: libsvm/svm.m4
	m4 libsvm/svm.m4 > libsvm/svm.java

clean:
	rm -f libsvm/*.class *.class *.jar libsvm/*~ *~ libsvm/svm.java

dist: clean all
	rm *.class libsvm/*.class
