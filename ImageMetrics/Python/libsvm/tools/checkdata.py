#!/usr/bin/env python

#
# A format checker for LIBSVM
#

#
# Copyright (c) 2007, Rong-En Fan
#
# All rights reserved.
#
# This program is distributed under the same license of the LIBSVM package.
# 

from sys import argv, exit
import os.path

def err(line_no, msg):
	print("line {0}: {1}".format(line_no, msg))

# works like float() but does not accept nan and inf
def my_float(x):
	if x.lower().find("nan") != -1 or x.lower().find("inf") != -1:
		raise ValueError

	return float(x)

def main():
	if len(argv) != 2:
		print("Usage: {0} dataset".format(argv[0]))
		exit(1)

	dataset = argv[1]

	if not os.path.exists(dataset):
		print("dataset {0} not found".format(dataset))
		exit(1)

	line_no = 1
	error_line_count = 0
	for line in open(dataset, 'r'):
		line_error = False

		# each line must end with a newline character
		if line[-1] != '\n':
			err(line_no, "missing a newline character in the end")
			line_error = True

		nodes = line.split()

		# check label
		try:
			label = nodes.pop(0)
			
			if label.find(',') != -1:
				# multi-label format
				try:
					for l in label.split(','):
						l = my_float(l)
				except:
					err(line_no, "label {0} is not a valid multi-label form".format(label))
					line_error = True
			else:
				try:
					label = my_float(label)
				except:
					err(line_no, "label {0} is not a number".format(label))
					line_error = True
		except:
			err(line_no, "missing label, perhaps an empty line?")
			line_error = True

		# check features
		prev_index = -1
		for i in range(len(nodes)):
			try:
				(index, value) =  nodes[i].split(':')

				index = int(index)
				value = my_float(value)

				# precomputed kernel's index starts from 0 and LIBSVM
				# checks it. Hence, don't treat index 0 as an error.
				if index < 0:
					err(line_no, "feature index must be positive; wrong feature {0}".format(nodes[i]))
					line_error = True
				elif index <= prev_index:
					err(line_no, "feature indices must be in an ascending order, previous/current features {0} {1}".format(nodes[i-1], nodes[i]))
					line_error = True
				prev_index = index
			except:
				err(line_no, "feature '{0}' not an <index>:<value> pair, <index> integer, <value> real number ".format(nodes[i]))
				line_error = True

		line_no += 1

		if line_error:
			error_line_count += 1
	
	if error_line_count > 0:
		print("Found {0} lines with error.".format(error_line_count))
		return 1
	else:
		print("No error.")
		return 0

if __name__ == "__main__":
	exit(main())
