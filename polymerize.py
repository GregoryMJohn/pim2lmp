#!/usr/bin/env python3

import os
import numpy as np
import math
import argparse
import re
import random
from math import sin, cos

parser = argparse.ArgumentParser(description='Extend MPIM homopolymer from monomer or oligomer',add_help=True)
parser.add_argument("-xyz",help=".xyz file containing monomer or oligomer atomic coordinates")
parser.add_argument("-n",help='Chain length')
parser.add_argument("-o",help='Name of .xyz file to write structure to')

args = parser.parse_args()
nrepeats,outfile = args.n,args.o

repeat_unit_xyz = args.xyz

head = int(input("Head atom index: "))
tail = int(input("Tail atom index: "))
head_cap = int(input("Head cap atom index: "))
tail_cap = int(input("Tail cap atom index: "))

repeat_unit_lines = [line.split() for line in open(repeat_unit_xyz,'r') if line.strip()]

repeat_unit_atoms = [line for line in repeat_unit_lines[2:]]

backbone_vector = np.array(list(map(float,repeat_unit_atoms[tail_cap-1][1:]))) - np.array(list(map(float,repeat_unit_atoms[head-1][1:])))
backbone_vector_len = np.linalg.norm(backbone_vector)
backbone_vector = ((backbone_vector_len + 0.3) / backbone_vector_len) * backbone_vector

with open(outfile,'w') as outfh:
	# write new atoms to output file
	outfh.write(str(int(nrepeats)*(len(repeat_unit_atoms)-1) - int(nrepeats) + 2) + "\n\n")

	for n,atom in enumerate(repeat_unit_atoms,start=1):
		if n != tail_cap:
			outfh.write(" ".join(atom)+"\n")

	for i in range(1, int(nrepeats)):
		for n,atom in enumerate(repeat_unit_atoms,start=1):
			element = atom[0]
			coords = np.array(list(map(float,atom[1:])))
			coords = [sum(x) for x in zip(coords,i*backbone_vector)]

			if not n in [head_cap, tail_cap]:
				outfh.write(element+" "+" ".join(list(map(str,coords)))+"\n")
			elif all([n==tail_cap, i == int(nrepeats) - 1]):
				outfh.write(element+" "+" ".join(list(map(str,coords)))+"\n")
		
