#!/usr/local/bin/python
"""
	tokenizer in python (albeit a (very) long one 
	outputs similar to shell script
"""
import os
import re
import subprocess
import sys
import tempfile

if len(sys.argv) < 2:
	print("Error! No argument given. Need the filename for the command.")
	sys.exit(1)

lines = open(sys.argv[1], 'r').readlines()
lines_str = '\n'.join(lines).lower()
lines_str = re.sub("[^a-zA-Z]", '\n', lines_str)
lines = lines_str.split('\n')

count_dict = {}
for line in lines:
	line = line.strip()
	if line not in count_dict:
		count_dict[line] = 0
	count_dict[line] += 1

count_dict.pop('', None)

output_str = ""
for w in sorted(count_dict, key=count_dict.get, reverse=True):
	output_str += str(count_dict[w]) + " " + w + "\n"

temp_f = tempfile.NamedTemporaryFile()
temp_f.write(output_str)
# print(temp_f.name)
subprocess.Popen(['less', temp_f.name], stdin=subprocess.PIPE).communicate()
