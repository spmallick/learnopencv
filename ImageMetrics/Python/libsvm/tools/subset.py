#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
	xrange = range

def exit_with_help(argv):
	print("""\
Usage: {0} [options] dataset subset_size [output1] [output2]

This script randomly selects a subset of the dataset.

options:
-s method : method of selection (default 0)
     0 -- stratified selection (classification only)
     1 -- random selection

output1 : the subset (optional)
output2 : rest of the data (optional)
If output1 is omitted, the subset will be printed on the screen.""".format(argv[0]))
	exit(1)

def process_options(argv):
	argc = len(argv)
	if argc < 3:
		exit_with_help(argv)

	# default method is stratified selection
	method = 0  
	subset_file = sys.stdout
	rest_file = None

	i = 1
	while i < argc:
		if argv[i][0] != "-":
			break
		if argv[i] == "-s":
			i = i + 1
			method = int(argv[i])
			if method not in [0,1]:
				print("Unknown selection method {0}".format(method))
				exit_with_help(argv)
		i = i + 1

	dataset = argv[i]
	subset_size = int(argv[i+1])
	if i+2 < argc:
		subset_file = open(argv[i+2],'w')
	if i+3 < argc:
		rest_file = open(argv[i+3],'w')

	return dataset, subset_size, method, subset_file, rest_file

def random_selection(dataset, subset_size):
	l = sum(1 for line in open(dataset,'r'))
	return sorted(random.sample(xrange(l), subset_size))

def stratified_selection(dataset, subset_size):
	labels = [line.split(None,1)[0] for line in open(dataset)]
	label_linenums = defaultdict(list)
	for i, label in enumerate(labels):
		label_linenums[label] += [i]

	l = len(labels)
	remaining = subset_size
	ret = []

	# classes with fewer data are sampled first; otherwise
	# some rare classes may not be selected
	for label in sorted(label_linenums, key=lambda x: len(label_linenums[x])):
		linenums = label_linenums[label]
		label_size = len(linenums) 
		# at least one instance per class
		s = int(min(remaining, max(1, math.ceil(label_size*(float(subset_size)/l)))))
		if s == 0:
			sys.stderr.write('''\
Error: failed to have at least one instance per class
    1. You may have regression data.
    2. Your classification data is unbalanced or too small.
Please use -s 1.
''')
			sys.exit(-1)
		remaining -= s
		ret += [linenums[i] for i in random.sample(xrange(label_size), s)]
	return sorted(ret)

def main(argv=sys.argv):
	dataset, subset_size, method, subset_file, rest_file = process_options(argv)
	#uncomment the following line to fix the random seed 
	#random.seed(0)
	selected_lines = []

	if method == 0:
		selected_lines = stratified_selection(dataset, subset_size)
	elif method == 1:
		selected_lines = random_selection(dataset, subset_size)

	#select instances based on selected_lines
	dataset = open(dataset,'r')
	prev_selected_linenum = -1
	for i in xrange(len(selected_lines)):
		for cnt in xrange(selected_lines[i]-prev_selected_linenum-1):
			line = dataset.readline()
			if rest_file: 
				rest_file.write(line)
		subset_file.write(dataset.readline())
		prev_selected_linenum = selected_lines[i]
	subset_file.close()

	if rest_file:
		for line in dataset: 
			rest_file.write(line)
		rest_file.close()
	dataset.close()

if __name__ == '__main__':
	main(sys.argv)

