#!/usr/bin/env python3

import os
import numpy as np
import math
import argparse
import re
from scipy import sparse
import timeit
from alive_progress import alive_bar
from alive_progress import config_handler
from textwrap import wrap

config_handler.set_global(bar='classic2')

startTime = timeit.default_timer()


parser = argparse.ArgumentParser(description='Build LAMMPS input file from .pdb',add_help=True)
parser.add_argument("pdb",help='A .pdb file to provide coordinates and atoms types for the LAMMPS input file')
parser.add_argument("output",help='the root name of the LAMMPS data and input files to be generated')
parser.add_argument("-ff",nargs='?',default="opls-metallocene",help='Which force field to use for the simulation')
parser.add_argument("-v",default='1',help='Verbosity level: set to 0 to run silently, 1 for low, 2 for medium, and 3 for high output and debugging')
parser.add_argument("-input",help='If the -input flag is provided, an LAMMPS input file will be written in addition to the data file',action="store_true")
args = parser.parse_args()
pdb,infile,datafile,frc,verbosity = args.pdb,args.output + ".in",args.output + ".data", os.environ["PIM2LMPHOME"]+"/" + args.ff + ".ff",int(args.v) 
write_infile = args.input

def rot_x(angle,vector):
	
	theta = math.radians(angle-90)

	Rx = np.array([[1,0,0],
		           [0,math.cos(theta),-1*math.sin(theta)],
		           [0,(math.sin(theta)),math.cos(theta)]])

	return np.dot(Rx,vector)

def rot_y(angle,vector):
	
	theta = math.radians(angle-90)

	Ry = np.array([[math.cos(theta),0,(math.sin(theta))],
	       [0,1,0],
	       [-1*math.sin(theta),0,math.cos(theta)]])

	return np.dot(Ry,vector)

def rot_z(angle,vector):
	
	theta = math.radians(angle-90)

	Rz = np.array([[math.cos(theta),-1*math.sin(theta),0],
		           [(math.sin(theta)),math.cos(theta),0],
		           [0,0,1]])

	return np.dot(Rz,vector)

def get_lattice_vectors(abc, alpha, beta, gamma):

	a = np.array([abc[0],0.0])
	b = np.array([0,abc[1],0])
	b = rot_z(gamma,b)
	c = np.array([0,0,abc[2]])
	c = rot_x(beta,c)
	c = rot_y(alpha,c)

	return np.array([a,b,c],dtype=object)

def get_fractional(xyz,F_matrix):
	return np.dot(F_matrix,xyz)

def get_tilt(a,b,c,alpha,beta,gamma):

	xy = b*math.cos(gamma)
	xz = c*math.cos(beta)
	yz = (b*math.cos(alpha) - xy*xz)/math.sqrt(b**2 - xy**2)

	return np.array([xy,xz,yz])


def getMinXYZ(atomList):
	xValues = [float(atomList[n].xyz[0]) for n in atomList]
	yValues = [float(atomList[n].xyz[1]) for n in atomList]
	zValues = [float(atomList[n].xyz[2]) for n in atomList]

	return [min(val) for val in [xValues,yValues,zValues]]


def getMaxXYZ(atomList):
	xValues = [float(atomList[n].xyz[0]) for n in atomList]
	yValues = [float(atomList[n].xyz[1]) for n in atomList]
	zValues = [float(atomList[n].xyz[2]) for n in atomList]

	return [max(val) for val in [xValues,yValues,zValues]]


atoms,bonds,angles,dihedrals,impropers,atom_types,bond_types,angle_types,dihedral_types,improper_types,residues,atom_types_list = 0,0,0,0,0,0,0,0,0,0,0,[]

triclinic,abc,alpha,beta,gamma,a,b,c,xy,xz,yz,atomList,atomTypeDict,atomType,bondsDict,bondTypes,printedBonds,termPositions,anglesDict = False,np.empty(3,dtype=float),0,0,0,[1,0,0],[0,1,0],[0,0,1],0,0,0,{},{},0,{},{},{},set(),{}


class Atom:
	def __init__(self,index,xyz,element,atom_type=None,charge=0.0,bonds=None,mass=None,molecule=1):
		self.index = index
		self.xyz = xyz
		self.element = element
		self.atom_type = atom_type
		self.charge = charge
		self.bonds = bonds
		self.mass = mass
		self.molecule = molecule


def getAtomTypesInBond(Atom1,Atom2):
	i = str(Atom1.atom_type[1])
	j = str(Atom2.atom_type[1])

	return [i,j] 

def updateBondList(bondsDict,terminators):
	updatedBonds = {}

	if verbosity > 0: print(".....\n.....\n***** Updating bonds *****\n**************************")

	for b in bondsDict:
		num_terms = 0
		for term in terminators:
			if int(b) >= term: num_terms += 1

		#if isinstance(bondsDict[b],str):
			#if int(bonds_dict[b]) not in terminators:
			#updatedBonds[str(int(b)-num_terms)] = [str(int(bondsDict[b]) - num_terms)]

		#elif isinstance(bondsDict[b],list):
		updatedBonds[str(int(b)-num_terms)] = [str(int(x) - num_terms) for x in bondsDict[b]]

	for bond in updatedBonds.copy():
		for b in updatedBonds.copy():
			if int(b) < int(bond) and bond in updatedBonds[b]:
				updatedBonds[bond].remove(b)

		if len(updatedBonds[bond]) == 0:
			del updatedBonds[bond]

	return updatedBonds


def getParagraph(ff_lines,flag,delimiter="!"):
	reading = False

	paragraph = []
	for line in ff_lines:
		if flag in line:
			reading = True
		elif "$end" in line:
			reading = False
		elif reading == True and delimiter not in line:
			paragraph.append(line)
	return paragraph

class AtomType:
	def __init__(self,line):
		self.type = line[0]
		self.name = line[1]
		self.mass = line[2]
		self.charge = line[3]
		self.element = line[4]
		self.connections = {}
		
		connex_def = line[5].split(",")
	
		for entry in connex_def: 
			 connex = re.findall('\d+|\D+',entry)
			 self.connections[connex[1]] = int(connex[0])

class LJParams:
	def __init__(self,line):
		self.atom = line[0]
		self.sigma = line[1]
		self.epsilon = line[2]


class BondsParams:
	def __init__(self,line):
		self.i = line[0]
		self.j = line[1]
		self.r0 = line[2]
		self.K = line[3]


class AnglesParams: 
	def __init__(self,line):
		self.i = line[0]
		self.j = line[1]
		self.k = line[2]
		self.theta0 = line[3]
		self.K = line[4]


class TorsionsParams:
	def __init__(self,line):
		self.i = line[0]
		self.j = line[1]
		self.k = line[2]
		self.l = line[3]
		self.K1 = line[4]
		self.K2 = line[5]
		self.K3 = line[6]
		self.K4 = line[7]
		self.atoms = [self.i,self.j,self.k,self.l]
		self.constants = [self.K1,self.K2,self.K3,self.K4]

class ImpropersParams:
	def __init__(self,line):
		self.i = line[0]
		self.j = line[1]
		self.k = line[2]
		self.l = line[3]
		self.theta = line[4]
		self.K = line[5]

class ForceField:
	def __init__(self,frc):
		force_field = [line.split() for line in open(frc,'r') if line.strip() and not "!" in line]
		self.atoms = [AtomType(line) for line in getParagraph(force_field,"$atoms")]
		self.lj = [LJParams(line) for line in getParagraph(force_field,"$lj")]
		self.bonds = [BondsParams(line) for line in getParagraph(force_field,"$bonds")]
		self.angles = [AnglesParams(line) for line in getParagraph(force_field,"$angles")]
		self.torsions = [TorsionsParams(line) for line in  getParagraph(force_field,"$torsions")]
		self.impropers = [ImpropersParams(line) for line in getParagraph(force_field,"$impropers")]



def getAdjacencyMatrix(pdb):
	start = timeit.default_timer()
	atoms = 0

	with open(pdb, 'r') as fh:
		for line in fh:
			#Get the number of atoms, which will be the dimension of the square adjacency matrix, A
			if "ATOM" in line or "HETATM" in line:
				atoms+=1
	
			#Read the bonds and create adjacency matrix
			elif "CONECT" in line:
				current_line = list(map(int,re.findall('.....',line[6:])))
				
				#If the line begins the bonds section of the PDB, create an ajacency matrix, A, full of zeroes 
				if current_line[0] == 1:
					A = np.zeros(shape=(atoms,atoms), dtype=np.uint8)
	
				#Populate the adjacency matrix with ones for bonds between atoms in the PDB 
				i = current_line[0] - 1
				J = current_line[1:]
				J = [j-1 for j in J]
	
				for j in J:
					A[i,j] = 1
					A[j,i] = 1

	stop = timeit.default_timer()
	print(".....\n..... Got adjacency matrix in",'%.3g' % (stop-start)+"s")
	return A 


def getBondedAtoms(atom_index,atomList):
	atom = atomList[atom_index]
	if atom.bonds != None:
		return set(atom.bonds)
	else:	
		return None

def getMasses(atomList,force_field):
	start = timeit.default_timer()
	if verbosity > 0: print(".....\n.....\n***** Getting masses *****\n**************************")

	with alive_bar(len(force_field.atoms)*len(atomList)) as bar:
		for atype in force_field.atoms:
			for n in atomList:
				atom = atomList[n]
				if atom.atom_type[0] == atype.type:
					atom.mass = atype.mass
				bar()
	stop = timeit.default_timer()
	print(".....\n..... Got masses in",'%.3g' % (stop-start) + "s")

def getDegreeOfConnectivity(A):
	D = [row.sum() for row in A]
	return D 

def getBondedByElem(atom_index,atomList,element):
	bondedAtoms = getBondedAtoms(atom_index,atomList)
	return [atom for atom in bondedAtoms if atomList[str(atom)].element == element]

def numberOfBondedX_byElem(atom_index,atomList,element):
	bondedX = getBondedByElem(atom_index,atomList,element)
	return len(bondedX)

def getAtomTypes(A,atomList,force_field):
	start = timeit.default_timer()
	if verbosity > 0: print(".....\n.....\n***** Getting atom types *****\n******************************")
	A_sparse = sparse.csr_matrix(A)
	A2_sparse = A_sparse.dot(A_sparse)
	A3_sparse = A2_sparse.dot(A_sparse).toarray()
	
	#Calculate degree of connectivity for each atom
	D = getDegreeOfConnectivity(A)

	#Start loading bar loop
	with alive_bar(len(atomList)) as bar:
		#Loop over atoms
		for n in atomList:
			atom = atomList[n]
			connections = D[int(atom.index)-1]
			#Loop over atom types in force field
			for atype in force_field.atoms:
				#Ignore H atoms, since they will be assigned types according to the atom types they are bonded to
				if all([atom.element == atype.element,atom.element != 'H',connections == sum([atype.connections[elem] for elem in atype.connections]) ]):
					#Ignore CA and CA5 types, as they share the same connection data
					if not atype.name in ['CA','CA5','Fe','CZ']:
						if all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
							atom.atom_type = [atype.type,atype.name]

							if verbosity > 2: print(atom.index,atom.atom_type)
							if atom.atom_type[1] == 'CT':
								for bonded_atom in atom.bonds:
									if atomList[str(bonded_atom)].element == 'H':
										atomList[str(bonded_atom)].atom_type = ['140','HC']
										
										if verbosity > 2: print(atomList[str(bonded_atom)].index,atomList[str(bonded_atom)].atom_type)
					elif atype.name == 'CZ':
						if numberOfBondedX_byElem(atom.index,atomList,'C') == 2 and connections == 2:
							if any([ atomList[str(bonded_atom2)].element == 'H' for bonded_atom in atom.bonds for bonded_atom2 in atomList[str(bonded_atom)].bonds ]):
								atom.atom_type = ['929','CZ']
							elif all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
								atom.atom_type = [atype.type, atype.name]

						elif all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
							atom.atom_type = [atype.type, atype.name]
							atom.charge = atype.charge
							for bonded_atom in atom.bonds:
								if atomList[str(bonded_atom)].element == 'H':
									atomList[str(bonded_atom)].atom_type = ['926','HC']

					elif atype.name in ['CA','CA5']:
						if connections == 3 and (numberOfBondedX_byElem(atom.index,atomList,'C') == 3 or (numberOfBondedX_byElem(atom.index,atomList,'C') == 2 and numberOfBondedX_byElem(atom.index,atomList,'H') == 1)) or all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
							if atype.name == 'CA5' and any([A3_sparse[bonded_atom1-1][bonded_atom2-1] == 1 for bonded_atom1 in atom.bonds for bonded_atom2 in atom.bonds if bonded_atom1 != bonded_atom2 ]):
								atom.atom_type = ['998','CA5']
								if verbosity > 2: print(atom.index,atom.atom_type)
								for bonded_atom in atom.bonds:
									if atomList[str(bonded_atom)].element == 'H':
										atomList[str(bonded_atom)].atom_type = ['999','HA5']
										if verbosity > 2: print(atomList[str(bonded_atom)].index,atomList[str(bonded_atom)].atom_type)
								
							elif atype.name == 'CA' and atom.atom_type==None:
								isBondedToCZ = False
								
								for bonded_atom in atom.bonds:
									if atomList[str(bonded_atom)].element == 'C' and len(atomList[str(bonded_atom)].bonds) == 2:
										isBondedToCZ = True
										break

								if isBondedToCZ == True and all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
									if atype.type in ['260','1001']:
										inPyridine = False
										for bonded_atom in atom.bonds:
											for other_atom in atomList[str(bonded_atom)].bonds:
												if atomList[str(other_atom)].element == 'N':
													inPyridine = True
													break
										if inPyridine:
											atom.atom_type = ['1001','CA']
										else:
											atom.atom_type = ['260','CA']
									else:
										atom.atom_type = [atype.type,atype.name]

								elif isBondedToCZ == False and all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] or elem == '*' for elem in atype.connections]):
									if numberOfBondedX_byElem(atom.index,atomList,'N')==1 and atype.type in ['521','1000']:
										atom.atom_type = [atype.type,atype.name]
									else:
										hasNTwoBondsAway = False
										hasNThreeBondsAway = False

										for bonded_atom in atom.bonds:
											for other_atom in atomList[str(bonded_atom)].bonds:
												if atomList[str(other_atom)].element == 'N':
													hasNTwoBondsAway = True
													break
												else:
													for third_atom in atomList[str(other_atom)].bonds:
														if atomList[str(third_atom)].element == 'N':
															hasNThreeBondsAway = True
															break

										if hasNTwoBondsAway:
											atom.atom_type = ['522','CA']
										
										elif hasNThreeBondsAway:
											atom.atom_type = ['523','CA']

										else:
											atom.atom_type = [atype.type,atype.name]
								
								if verbosity > 2: print(atom.atom_type)
								for bonded_atom in atom.bonds:
									if atomList[str(bonded_atom)].element == 'H':
										atomList[str(bonded_atom)].atom_type = ['146','HA']
										if verbosity > 2: print(atomList[str(bonded_atom)].index,atomList[str(bonded_atom)].atom_type)

					elif atype.name == 'Fe':
						atom.atom_type = [atype.type,atype.name]
						if verbosity > 2: print(atom.index,atom.atom_type)
	
				elif all([atom.element == atype.element,atom.element == 'H',not atype.name in ['HC','HA']]) and all([numberOfBondedX_byElem(atom.index,atomList,elem) == atype.connections[elem] for elem in atype.connections]):
					if atom.atom_type == None:
						if any([len(atomList[str(bonded_atom)].bonds) == 2 for bonded_atom in atom.bonds]):
							atom.atom_type = ['926','HC']
						else:
							atom.atom_type = [atype.type,atype.name]

					if verbosity > 2: print(atom.index,atom.atom_type)

				elif all([atom.element == atype.element, atom.element == 'C', connections == 4, numberOfBondedX_byElem(atom.index,atomList,'PS') == 1]):
					if atom.atom_type == None:
						atom.atom_type = ['998','CA5']
						if verbosity > 2: print(atom.index,atom.atom_type)
						for bonded_atom in atom.bonds:
							if atomList[str(bonded_atom)].element == 'H':
								atomList[str(bonded_atom)].atom_type = ['999','HA5']
								if verbosity > 2: print(atomList[str(bonded_atom)].index,atomList[str(bonded_atom)].atom_type)

			bar()

	stop = timeit.default_timer()
	print(".....\n..... Got atom types in",'%.3g' % (stop-start)+"s")


def getBondedByType(atom_index,atomList,atom_type):
	bondedAtoms = getBondedAtoms(atom_index,atomList)
	return [atom for atom in bondedAtoms if atomList[atom].atom_type == atom_type]


def numberOfBondedX_byType(atom_index,atomList,atom_type):
	bondedX = getBondedByType(atom_index,atomList,atom_type)
	return len(bondedX)

def getCharges(atom,force_field):
	for atype in force_field.atoms:
		if str(atom.atom_type[0]) == atype.type:
			if atom.charge != None:
				atom.charge = atype.charge
			else:
				continue

		if atom.atom_type[1] == 'CA5' and numberOfBondedX_byElem(atom.index,atomList,'C') == 3:
			atom.charge = '-0.010'  

def getBonds(A,atomList):
	start = timeit.default_timer()
	A_sparse = sparse.csr_matrix(A)

	bonds = [[i+1,j+1] for i,j in zip(*A_sparse.nonzero())]
	for bond in bonds:
		if atomList[str(bond[0])].bonds:
			atomList[str(bond[0])].bonds.append(bond[1])

		else:
			atomList[str(bond[0])].bonds = [bond[1]] 

	stop = timeit.default_timer()
	print(".....\n..... Got bonds in",'%.3g' % (stop-start)+"s")

def getBondTypes(atomList):
	if verbosity > 0: print(".....\n.....\n***** Getting bond types *****\n******************************")
	start = timeit.default_timer()

	bond_types = 0
	bondTypes = {}

	with alive_bar(len(atomList)) as bar:
		for n in atomList:
			atom = atomList[n]
			if atom.bonds != None:
				for other_atom in atom.bonds:	
					atoms = getAtomTypesInBond(atom,atomList[str(other_atom)])
					if set(atoms) in bondTypes.values():
						continue
					else:
						bond_types+=1				
						bondTypes[bond_types] = set(atoms)
			bar()

	stop = timeit.default_timer()
	print(".....\n..... Got bond types in ",'%.3g'%(stop-start)+"s")
	return bondTypes


class Angle:
	def __init__(self,atoms,index=None):
		self.atoms = atoms
		self.index = index
		self.type = None
		self.atom_types = [atom.atom_type[1] for atom in atoms]
		self.vertex = atoms[1]


def getBondAngles(atomList):
	start = timeit.default_timer()

	angles = 0
	angleDict = {}

	with alive_bar(len(atomList)) as bar:
		#Iterate over all atoms
		for n in atomList:
			atom = atomList[n]
			#Check if current atom has any bonds
			if atom.bonds != None:
				#If it has more than one bond (is not a terminal atom),
				#then create list of angles that have the current atom as the vertex
				if len(atom.bonds) > 1:
					angles_list = [Angle([ atomList[str(bonded_atom)],atom,atomList[str(other_atom)] ]) for bonded_atom in atom.bonds for other_atom in atom.bonds if bonded_atom!=other_atom]

					#if atom.element == 'PS':
					#	iron_atom = str(getBondedByElem(atom.index,atomList,'Fe')[0])
					#	carbon_1 = str(getBondedByElem(atom.index,atomList,'C')[0])
					#	ring_carbons = []
					#	ring_carbons.append(carbon_1)
					#	for x in atomList[carbon_1].bonds:
					#		if atomList[str(x)].element == 'C':
					#			ring_carbons.append(str(x))
					#			for y in atomList[str(x)].bonds:
					#				if atomList[str(y)].element == 'C' and not str(y) in ring_carbons:
					#					ring_carbons.append(str(y))

					#	for carbon_atom in ring_carbons:
					#		angles_list.append(Angle([ atomList[carbon_atom], atom, atomList[iron_atom] ]))

					for angle in angles_list:
						for other_angle in angles_list:
							if set(angle.atoms) == set(other_angle.atoms) and angle!=other_angle:
								angles_list.remove(other_angle)

						if angle.atom_types == ['CA5','PS','CA5']:
							if verbosity > 2: print(angle.atom_types)
							angles_list.remove(angle)

						elif not angle in angleDict.values():
							angles+=1
							angle.index = angles
							angleDict[angles] = angle
			bar()
	
				#for bonded_atom in atom.bonds:
				#	for other_atom in atom.bonds:
				#		if bonded_atom != other_atom:
				#			angle = Angle([ atomList[str(bonded_atom)],atom,atomList[str(other_atom)] ])
				#			if angle.atoms not in angleDict.values() and angle.atoms[::-1] not in angleDict.values():
				#				angles+=1
				#				angleDict[angles] = angle

	return angleDict



def getAngleTypes(anglesDict,atomList):
	start = timeit.default_timer()

	if verbosity > 0: print(".....\n.....\n***** Getting angle types *****\n*******************************")

	angleTypeDict = {}
	angleType = 0

	with alive_bar(len(anglesDict)) as bar:
		for n in anglesDict:
			angle = anglesDict[n]
			#if angleType == 0: 
			#	angleType+=1
			#	angle.type = angleType
			#	angleTypeDict[angle.type] = angle.atom_types
			#else:
			type_exists = False
			for index in angleTypeDict:
				if angleTypeDict[index] == angle.atom_types or angleTypeDict[index] == angle.atom_types[::-1]:
					type_exists = True
					angle.type = index
					break
			
			if not type_exists:
				angleType+=1 
				angle.type = angleType
				angleTypeDict[angle.type] = angle.atom_types 
			bar()
	
	stop = timeit.default_timer()
	print(".....\n..... Got angle types in",'%.3g' % (stop-start)+"s")
	
	if verbosity > 2: 
		for n in angleTypeDict: 
			print(n,angleTypeDict[n])

	return angleTypeDict

class Torsion:
	def __init__(self,index,atoms,type=None,style=None):
		self.index = index
		self.type = type
		self.atoms = atoms
		self.style = style


class Improper:
	def __init__(self,index,atoms,type=None,style=None,coeffs=None):
		self.index = index
		self.type = type
		self.atoms = atoms
		self.coeffs = coeffs 


def getImpropers(angleDict,atomList):
	start = timeit.default_timer()

	if verbosity > 0: print(".....\n.....\n***** Getting impropers *****\n*****************************")

	impropers = 0
	impropersList = []
	with alive_bar(len(angleDict)) as bar:
		for n in angleDict:
			angle = angleDict[n]
			if not False in [bool(atom.atom_type[1] in ['CA','CA5']) for atom in angle.atoms]:
				for bonded_atom in angle.vertex.bonds:
					if atomList[str(bonded_atom)].atom_type[1] in ['HA','HA5','PS','CZ']:
						improper = [str(bonded_atom)]
						for atom in angle.atoms:
							improper.append(str(atom.index))
						impropers+=1
						impropersList.append(Improper(impropers,improper))
					#if atomList[str(bonded_atom)].atom_type[1] == 'PS':
					#	improper = [str(bonded_atom)]
					#	for atom in angle.atoms:
					#		improper.append(str(atom.index))
					#	impropers+=1
					#	impropersList.append(Improper(impropers,improper))
			bar()

	stop = timeit.default_timer()
	print(".....\n..... Got impropers in",'%.3g' % (stop-start)+"s")
	

	return impropersList

def getBondCoeffs(bondTypes,force_field):
	start = timeit.default_timer()

	if verbosity > 0: print(".....\n.....\n***** Getting bond coefficients *****\n*************************************")

	bondCoeffs = {}
	with alive_bar(len(bondTypes)) as bar:
		for n in bondTypes:
			atom_types = bondTypes[n]
			for btype in force_field.bonds:
				if btype.i in atom_types and btype.j in atom_types:
					bondCoeffs[n] = [btype.K, btype.r0]
			bar()

	stop = timeit.default_timer()
	print(".....\n..... Got bond coefficients in",'%.3g' % (stop-start)+"s")

	return bondCoeffs


def getAngleCoeffs(angleTypes,force_field):
	start = timeit.default_timer()

	if verbosity > 0: print(".....\n.....\n***** Getting angle coefficients *****\n**************************************")
	#angc = dict(zip(angleTypes,[ [angtype.K,angtype.theta0] for n in angleTypes for angtype in force_field.angles if set((angtype.i,angtype.j,angtype.k)) == set(angleTypes[n]) ]))

	angc = {}

	for n in angleTypes:
		for angtype in force_field.angles:
			if [angtype.i,angtype.j,angtype.k] in [angleTypes[n], angleTypes[n][::-1]]:
				angc[n] = [angtype.K,angtype.theta0]


	stop = timeit.default_timer()
	print(".....\n..... Got angle coefficients in",'%.3g' % (stop-start)+"s")
	for n in angc: print(n,angc[n])

	return angc 

	#angleCoeffs = {}
	#for n in angleTypes:
	#	atom_types = angleTypes[n]
	#	for angtype in force_field.angles:
	#		if all([if atom1 == atom2 for atom2 in (angtype.i,angtype.j,angtype.k) for atom1 in atom_types])
#
	#for n in angleTypes:
	#	atom_types = angleTypes[n]
	#	for angtype in force_field.angles:
	#		
	#		if set([angtype.i,angtype.j,angtype.k]) == set(atom_types):
	#			if verbosity > 2: print("Found angle coefficients")
	#			angleCoeffs[n] = [angtype.K, angtype.theta0]
#
	#return angleCoeffs 


def getLJ(atomTypeDict,force_field):
	start = timeit.default_timer()
	if verbosity > 0: print(".....\n.....\n***** Getting Lennard-Jones parameters *****\n********************************************")

	#ljp = [[atom.atom,atom.sigma,atom.epsilon] for atom in force_field.lj for atype in atomTypeDict if str(atype) == atom.atom]
	
	

	#return dict(zip(list(atomTypeDict.values()),ljp))
	ljpDict = {}
	for atom in force_field.lj:
		for atype in atomTypeDict:
			if str(atype) == atom.atom:
				ljpDict[str(atype)] = [atom.sigma,atom.epsilon]

	stop = timeit.default_timer()
	print(".....\n..... Got Lennard-Jones parameters in",'%.3g' % (stop-start)+"s")
	return ljpDict 



def updateMolecules(atomList,termPositions):
	if verbosity > 0: print(".....\n.....\n***** Updating molecules *****\n******************************")
	for n in atomList:
		atom = atomList[n]
		terminators = 0
		for term in termPositions:
			if int(atom.index) >= term:
				terminators+=1

		if terminators > 0:
			atom.molecule = str(terminators)

	return atomList 


def getLaplacian(A):
	#Calculate degree of connectivity for each atom
	D = getDegreeOfConnectivity(A)
	#Generate the Laplacian matrix
	L = -1*A 
	for i in range(L.shape[0]):
		L[i,i] = D[i]

	return L


def countDihedrals(A):
	#Generate the Laplacian matrix
	L = getLaplacian(A)

	#Enumerate dihedrals
	num_dihedrals = 0
	for i,row in enumerate(L):
		degree_a = L[i,i]
		for j, element in enumerate(row):
			#Only count upper triangular submatrix to avoid duplicate entries
			if j>i and element == -1:
				degree_b = L[j,j]
				num_dihedrals+= (degree_a * degree_b) - (degree_a + degree_b) + 1

	return num_dihedrals


def getTorsions(Laplacian,atomList):
	if verbosity > 0: print(".....\n.....\n***** Getting dihedrals *****\n*****************************")
	start = timeit.default_timer()

	torsions = 0
	torsion_list = []
	mlist = []

	with alive_bar(len(atomList)) as bar:
		for n in atomList:
			atom = atomList[n]
			if atom.bonds != None and len(atom.bonds) > 1:
				tlist = [ [int(i),int(atom.index),int(k),int(l)] for k in atom.bonds for i in atom.bonds if i!=k and len(atomList[str(k)].bonds) > 1 for l in atomList[str(k)].bonds if l!=int(atom.index) ]
				for t in tlist:
					if any([a == b for a in ['HC','PS','Fe'] for b in (atomList[str(t[0])].atom_type[1], atomList[str(t[3])].atom_type[1])]):
						continue

					elif any([atomList[str(t_atom)].atom_type[1] == 'PS' for t_atom in t]):
						continue

					if not t in mlist and not t[::-1] in mlist and not any([atomList[str(t_atom)].atom_type[1] == 'CZ' for t_atom in t[1:3]]): 
						mlist.append(t)
			bar()

	

	#molecules = int(max([atomList[n].molecule for n in atomList]))
	#print(molecules)
	#for molecule in range(1,molecules + 1):
	#	for a in atomList:
	#		atom1 = atomList[a]
	#		for b in atomList:
	#			atom2 = atomList[b]
	#			if all([a!=b, atom1.atom_type[1] == 'PS', atom2.atom_type[1] == 'PS', int(atom1.molecule) == molecule, int(atom2.molecule) == molecule]):
	#				t =[int(getBondedByElem(atom1.index,atomList,'C')[0]),int(atom1.index),int(atom2.index),int(getBondedByElem(atom2.index,atomList,'C')[0])]
	#				if not t in mlist and not t[::-1] in mlist:
	#					mlist.append(t)
	#					print(t)

	for a in atomList:
		if atomList[a].element == "Fe":
			iron = atomList[a]
			ps1 = iron.bonds[0]
			ps2 = iron.bonds[1]
			c1 = map(int, getBondedByElem(str(ps1),atomList,'C'))
			c2 = map(int, getBondedByElem(str(ps2),atomList,'C'))
			t_list = [[a, ps1, ps2, b] for a in c1 for b in c2]
			for t in t_list:
				if not t in mlist and not t[::-1] in mlist:
					mlist.append(t)
					if verbosity > 2: print(t)

	for tor in mlist:			
		torsions+=1
		if any([ atomList[str(atom)].atom_type[1] == 'PS' for atom in tor ]):
			torsion_list.append(Torsion(torsions,list(map(str,tor)),style="fourier"))

		else:
			torsion_list.append(Torsion(torsions,list(map(str,tor)),style="opls"))	
		

	#for i,row in enumerate(Laplacian):
	#	for j,elem in enumerate(row):
	#		if Laplacian[i,i] > 1 and Laplacian[j,j] > 1 and j>i and elem!=0:
	#			for k,atom1 in enumerate(row):
	#				if k!=i and k!=j and atom1!=0 and atomList[str(k+1)].atom_type[1] !='NZ':
	#					for l,atom2 in enumerate(Laplacian[j]):
	#						if l!=i and l!=j and atom2!=0 and atomList[str(l+1)].atom_type[1] != 'NZ':
	#							torsions+=1
	#							torsion = [k+1,i+1,j+1,l+1]
	#							atom_types = []
	#							for atom in torsion:
	#								atom_types.append(atomList[str(atom)].atom_type[1])
	#							torsion_list.append(Torsion(torsions,list(map(str,torsion))))

	stop = timeit.default_timer()
	print(".....\n..... Got dihedrals in",'%.3g' % (stop-start)+"s")
	
	return torsion_list


def getTorsionTypes(dihedralList,atomList,force_field):
	if verbosity > 0: print(".....\n.....\n***** Getting dihedral types *****\n**********************************")
	start = timeit.default_timer()

	torsion_types = 0
	torsionTypes = {}


	for n,torsion in enumerate(dihedralList,start=1):
		atom_types = [atomList[str(atom)].atom_type[1] for atom in torsion.atoms]
		if verbosity > 2: print(n,atom_types)
		for entry in force_field.torsions:
			if entry.atoms in [atom_types,atom_types[::-1]]:
				#if not entry.atoms in torsionTypes.values() and not entry.atoms[::-1] in torsionTypes.values():
				if not any([ tor_type[0] in [entry.atoms, entry.atoms[::-1]] for tor_type in torsionTypes.values() ]):
					torsion_types+=1
					torsionTypes[torsion_types] = [entry.atoms,torsion.style]
					torsion.type = torsion_types
					if verbosity > 2: print(entry.atoms)
				else:
					for ttype in torsionTypes:
						if torsionTypes[ttype][0] in [entry.atoms,entry.atoms[::-1]]:
							#print(entry.atoms)
							torsion.type = ttype

	stop = timeit.default_timer()
	print(".....\n..... Got dihedral types in",'%.3g' % (stop-start)+"s")
	return torsionTypes


def getTorsionCoeffs(torsionTypeDict,force_field):
	start = timeit.default_timer()

	torsionCoeffs = {}

	for torsion in force_field.torsions:
		for torsion_type in torsionTypeDict:	
			if torsionTypeDict[torsion_type][0] == torsion.atoms:
				torsionCoeffs[torsion_type] = torsion.constants

	#with open(frc,'r') as ffh:
	#	reading_torsions = False
	#	for line in ffh:
	#		if "$torsions" in line: reading_torsions = True
	#		if "$end" in line: reading_torsions = False
	#		if reading_torsions == True:
	#			if "!" in line: continue
	#			current_line = line.split()
	#			for torsion_type in torsionTypeDict:
	#				torsion = torsionTypeDict[torsion_type]
	#				if torsion == current_line[0:4]:
	#					torsionCoeffs[torsion_type] = current_line[4:]

	stop = timeit.default_timer()
	print(".....\n..... Got dihedral coefficients in",'%.3g' % (stop-start)+"s")

	return torsionCoeffs


def getImproperTypesAndCoeffs(impropersList,force_field,atomList):
	improper_types = 0
	improper_type_dict = {}
	type_coeff_dict = {}
	for improper in impropersList:
		for imp_type in force_field.impropers:
			if set([atomList[atom].atom_type[1] for atom in improper.atoms]) == set([imp_type.i,imp_type.j,imp_type.k,imp_type.l]):
				if not [imp_type.i,imp_type.j,imp_type.k,imp_type.l] in improper_type_dict.values():
					improper_types+=1
					improper.type = improper_types
					improper_type_dict[improper_types] = [imp_type.i,imp_type.j,imp_type.k,imp_type.l]
					improper.coeffs = [imp_type.K,imp_type.theta]

				else:
					for key in improper_type_dict:
						if improper_type_dict[key] == [imp_type.i,imp_type.j,imp_type.k,imp_type.l]:
							improper.type = key
							improper.coeffs = [imp_type.K,imp_type.theta]

				type_coeff_dict[improper.type] = improper.coeffs

	return type_coeff_dict



boxDefined = False

if verbosity > 0: print("Writing data file:")
with open(datafile,'w') as datafh:
	
	datafh.write("LAMMPS data file generated from "+pdb+" using "+args.ff+" parameters\n\n")

	with open(pdb,'r') as reader:
		for line in reader:
			if "REMARK" in line:
				continue
			elif "CRYST1" in line:
				crystal_parameters = np.array(line.split())
				lattice_parameters = crystal_parameters[1:7].astype(np.float)
				abc = lattice_parameters[:3]
				alpha,beta,gamma = lattice_parameters[3],lattice_parameters[4],lattice_parameters[5]

				latticeMatrix = get_lattice_vectors(abc,alpha,beta,gamma)
				a = latticeMatrix[0]
				b = latticeMatrix[1]
				c = latticeMatrix[2]

				if crystal_parameters[7] == "P1":
					triclinic = True
					if list(map(float,crystal_parameters[4:7])) != [90.00,90.00,90.00]:
						tilt_factors = get_tilt(abc[0],abc[1],abc[2],alpha,beta,gamma)
						xy = tilt_factors[0]
						xz = tilt_factors[1]
						yz = tilt_factors[2]

				boxDefined = True

			else:
				current_line = line.split()
				
				#if "ATOM" in line or "HETATM" in line:
					#atoms+=1
					#if boxDefined == True:
						#current_atom_type = "".join(re.split("[^a-zA-Z]*", current_line[11]))
						#atomList[str(atoms)] = Atom(str(atoms),[current_line[6],current_line[7],current_line[8]],current_atom_type)

					#else:
						#current_atom_type = "".join(re.split("[^a-zA-Z]*", current_line[-1]))
						#atomList[str(atoms)] = Atom(str(atoms),current_line[-6:-3],current_atom_type,molecule=str(current_line[-7]))

				if "HETATM" in line:
					atomList[str(int(line[6:11]))] = Atom(line[6:11].strip(), [coord.strip() for coord in re.findall('........',line[30:54])], line[76:78].strip(), molecule=line[22:26].strip())
					atoms+=1
                         
				if current_line[0] == "TER":
					residues+=1
					termPositions.add(int(current_line[1]))

				elif current_line[0] == "END":
					
					if boxDefined == False:
						xyz_lo = [min(getMinXYZ(atomList))-5.00] * 3

						#for i in range(len(xyz_lo)):
						#	xyz_lo[i] -= 5.0

						xyz_hi = [max(getMaxXYZ(atomList))+5.00] * 3

						#for i in range(len(xyz_hi)):
						#	xyz_hi[i] += 5.0

					else:
						xyz_lo = [0,0,0]
						xyz_hi = abc

					#Get Adjacency Matrix
					adjacency_matrix = getAdjacencyMatrix(pdb)
					A_sparse = sparse.csr_matrix(adjacency_matrix)
					#Get Bonds
					getBonds(adjacency_matrix,atomList)

					#Get Force Field
					force_field = ForceField(frc)

					#Atom types
					getAtomTypes(adjacency_matrix,atomList,force_field)
					for n in atomList:
						if atomList[n].atom_type == None:
							print(atomList[n].index)

					#Get Masses
					getMasses(atomList,force_field)

					#Get Charges and append atom types
					if verbosity > 0: print(".....\n.....\n***** Getting charges *****\n**************************")
					start = timeit.default_timer()
					with alive_bar(len(atomList)) as bar:
						for atom in atomList:
							getCharges(atomList[atom],force_field)
							atom_types_list.append(atomList[atom].atom_type[0])
							bar()

					stop = timeit.default_timer()
					print(".....\n..... Got charges in",'%.3g' % (stop-start)+"s")

					#for atom in atomList:
					#	atom_types_list.append(atomList[atom].atom_type[0])

					atom_types_set = set(atom_types_list)
					atom_types = len(atom_types_set)

					#Bond Types
					#bonds = sum([sum(row[i+1:]) for i,row in enumerate(adjacency_matrix)])
					bonds = int(A_sparse.sum() / 2)
					bondTypes = getBondTypes(atomList)
					bond_types = len(bondTypes)

					#Angles
					start = timeit.default_timer()
					anglesDict = getBondAngles(atomList)
					stop = timeit.default_timer()
					print(".....\n..... Got bond angles in ",'%.3g' % (stop-start)+"s")
					for key in anglesDict:
						angles+=1

					#Angles types
					angleTypes = getAngleTypes(anglesDict,atomList)
					for key in angleTypes:
						angle_types+=1

					#Dihedrals
					L = getLaplacian(adjacency_matrix)
					dihedralList = getTorsions(L,atomList)

					dihedrals = len(dihedralList)

					#dihedrals = countDihedrals(adjacency_matrix)
					#if dihedrals != len(dihedralList):
					#	print(".....\n.....\n***** ERROR: Incorrect number of dihedrals! *****")

					#Dihedral types
					torsionTypeDict = getTorsionTypes(dihedralList,atomList,force_field)					
					dihedral_types = len(torsionTypeDict)

					#Impropers
					improperList = getImpropers(anglesDict,atomList)
					improper_coeffs = getImproperTypesAndCoeffs(improperList,force_field,atomList)

					impropers = len(improperList)
					improper_types = len(improper_coeffs)

					datafh.writelines([str(atoms)," atoms\n",
									   str(bonds)," bonds\n",
									   str(angles)," angles\n",
									   str(dihedrals)," dihedrals\n",
									   str(impropers)," impropers\n"])

					datafh.write(str(atom_types)+" atom types\n")
					if bonds != 0: datafh.write(str(bond_types)+" bond types\n")
					if angles != 0: datafh.write(str(angle_types)+" angle types\n")
					if dihedrals != 0: datafh.write(str(dihedral_types)+" dihedral types\n")
					if impropers != 0: datafh.write(str(improper_types)+" improper types\n")

					break

	datafh.writelines(["\n",
					   str(xyz_lo[0])+"\t",str(xyz_hi[0])," xlo xhi\n",
					   str(xyz_lo[1])+"\t",str(xyz_hi[1])," ylo yhi\n",
					   str(xyz_lo[2])+"\t",str(xyz_hi[2])," zlo zhi\n"])
	
	if [xy,xz,yz] != [0,0,0]:
		datafh.writelines([str(xy)," ",str(xz)," ",str(yz)," xy xz yz\n\n"])
	else:
		datafh.write("\n\n")

	datafh.write("Masses\n\n")
	atom_types_printed = []
	mass_index = 0
	for n in atomList:
		atom = atomList[n]
		if atom.atom_type[0] in atom_types_printed:
			continue
		else:
			mass_index+=1
			datafh.write("\t".join([str(mass_index),atom.mass,"#"+"/".join(list(map(str,atom.atom_type))),"\n"]))
			atom_types_printed.append(atom.atom_type[0])
			atomTypeDict[atom.atom_type[0]] = str(mass_index) 

	datafh.write("\n")

	datafh.write("\n")
	datafh.write("Pair Coeffs\n\n")
	ljParams = getLJ(atomTypeDict,force_field)
	if verbosity > 2:
		for params in ljParams:
			print(params,ljParams[params])

	for i,atom in enumerate(atomTypeDict,start=1):
		atom_type = str(atom)
		datafh.write(str(i)+"\t"+"\t".join(ljParams[atom_type][::-1])+"\t#"+atom_type+"\n")

	datafh.write("\n")

	datafh.write("Atoms\n\n")
	for atom in atomList:
		datafh.writelines([str(atomList[atom].index)+"\t",str(atomList[atom].molecule)+"\t",atomTypeDict[atomList[atom].atom_type[0]]+"\t",str("{:6.4f}".format(float(atomList[atom].charge)))+"\t"," \t".join(atomList[atom].xyz)+"\n"])
	datafh.write("\n")

	datafh.write("Bonds\n\n")
	start = timeit.default_timer()
	bonds = 0
	n = 0
	if verbosity > 0: print(".....\n.....\n***** Printing bonds *****\n**************************")
	A_sparse = A_sparse.tocoo()

	#def printBond(bond,btype):
	#	bonds+=1
	#	bond_type = btype
	#	datafh.write(str(bonds)+"\t"+str(bond_type)+"\t"+str(bond[0]+1)+"\t"+str(bond[1]+1)+"\n")
#
	#[ printBond([i,j],btype) for i,j in zip(*A_sparse.nonzero()) if j>i for btype in bondTypes if bondTypes[btype] == set([atomList[str(i+1)].atom_type[1],atomList[str(j+1)].atom_type[1]])]

	#A = adjacency_matrix
	#for i,row in enumerate(A):
	#	for j,elem in enumerate(row):
	#		if j>i and elem == 1:
	#			bonds+=1
	#			atom_types = set([atomList[str(i+1)].atom_type[1],atomList[str(j+1)].atom_type[1]])
	#			for btype in bondTypes:
	#				if bondTypes[btype] == atom_types:
	#					bond_type = btype
	#			datafh.write(str(bonds)+"\t"+str(bond_type)+"\t"+str(i+1)+"\t"+str(j+1)+"\n")
	#datafh.write("\n")

	for i,j,v in zip(A_sparse.row, A_sparse.col, A_sparse.data):
		if j>i and v == 1:
			bonds+=1
			atom_types = set([atomList[str(i+1)].atom_type[1],atomList[str(j+1)].atom_type[1]])

			for btype in bondTypes:
				if bondTypes[btype] == atom_types:
					bond_type = btype

			datafh.write(str(bonds)+"\t"+str(bond_type)+"\t"+str(i+1)+"\t"+str(j+1)+"\n")

	datafh.write("\n")
	stop = timeit.default_timer()
	print("..... Printed bonds in ",'%.3g' % (stop-start)+"s")
	
	datafh.write("Angles\n\n")
	for i in range(len(anglesDict)):
		angle = anglesDict[i+1]
		datafh.write(str(angle.index)+"\t"+str(angle.type)+"\t"+"\t".join(list(map(str,[atom.index for atom in angle.atoms])))+"\n")
	datafh.write("\n")

	datafh.write("Dihedrals\n\n")
	for i,torsion in enumerate(dihedralList):
		if torsion.index == i+1:
			datafh.write(str(torsion.index)+"\t"+str(torsion.type)+"\t"+"\t".join(list(map(str,torsion.atoms)))+"\n")
	datafh.write("\n")

	datafh.write("Impropers\n\n")
	for improper in improperList:
		datafh.write(str(improper.index)+"\t"+str(improper.type)+"\t"+" ".join(improper.atoms)+"\n")
	datafh.write("\n")

	bondCoeffs = getBondCoeffs(bondTypes,force_field)
	datafh.write("Bond Coeffs #harmonic\n\n")
	for i in bondCoeffs:
		datafh.write(str(i)+"\t"+"  ".join(bondCoeffs[i])+"\n")
	datafh.write("\n")

	datafh.write("Angle Coeffs #harmonic\n\n")
	angleCoeffs = getAngleCoeffs(angleTypes,force_field)
	for n in angleTypes:
		datafh.write(str(n)+"\t"+"\t".join(angleCoeffs[n])+"\t#"+"-".join(angleTypes[n])+"\n")
	datafh.write("\n")

	datafh.write("Dihedral Coeffs #OPLS-AA\n\n")
	dihedralCoeffs = getTorsionCoeffs(torsionTypeDict,force_field)
	for i in range(len(dihedralCoeffs)):
		torsion_type = i+1
		datafh.write(str(torsion_type)+"\t"+torsionTypeDict[torsion_type][1]+"\t"+"  ".join(dihedralCoeffs[torsion_type])+"\t#"+"-".join(torsionTypeDict[torsion_type][0])+"\n")
	datafh.write("\n")

	datafh.write("Improper Coeffs #harmonic\n\n")
	for entry in improper_coeffs:
		datafh.write(str(entry)+"\t"+"\t".join(improper_coeffs[entry])+"\n")
	datafh.write("\n")
	#datafh.write("1\t1.1\t180")
	#datafh.write("\n")

	if verbosity > 0: print("\nFinished writing data file!\n")

if write_infile:
	if verbosity > 0: print("Writing input file:")

	with open(infile,'w') as infh:
	
		infh.write("dimension 3\n"+"units real\n"+"boundary p p p\n"+"atom_style full\n")
		infh.writelines(["pair_style lj/cut/coul/cut 15.0\n",
						 "bond_style harmonic\n",
						 "angle_style harmonic\n",
						 "dihedral_style opls\n",
						 "improper_style harmonic\n\n"])
	
		infh.write("read_data "+datafile+"\n")
		infh.write("\n"+"neighbor 2.0 bin\n\n\n")
		#infh.write("# ------------------------ CVFF ------------------------------\n\n")
		#infh.write("pair_style buck6d/coul/gauss/dsf     0.9000    12.0000\n\n")
		infh.write("# ------------------------ Run -------------------------------\n\n")
		infh.write('''timestep 1.0
	
		minimize 0.0 1.0e-8 1000 100000
	
		fix 1  all nvt temp 600.0 600.0 100.0
		run 50000
		unfix 1
	
		fix 2 all nvt temp 300.0 300.0 100.0
		run 50000
		unfix 2
	
		fix 3 all npt tri 987.0 987.0 1000.0 temp 300.0 300.0 100.0
		run 50000
		unfix 3
	
		fix 4 all nvt temp 600.0 600.0 100.0
		run 50000
		unfix 4
	
		fix 5 all nvt temp 600.0 600.0 100.0
		run 100000
		unfix 5
	
		fix 6 all npt tri 29608.0 29608.0 1000.0 temp 300.0 300.0 100.0
		run 50000
		unfix 6
	
		fix 7 all nvt temp 600.0 600.0 100.0 
		run 50000
		unfix 7
	
		fix 8 all nvt temp 300.0 300.0 100.0
		run 100000
		unfix 8
	
		fix 9 all npt tri 49346.2 49346.2 1000.0 temp 300.0 300.0 100.0
		run 50000
		unfix 9 
	
		fix 10 all nvt temp 600.0 600.0 100.0
		run 50000
		unfix 10 
	
		fix 11 all nvt temp 300.0 300.0 100.0
		run 100000
		unfix 11
	
		fix 12 all npt tri 24673.0 24673.0 1000.0 temp 300.0 300.0 100.0
		run 5000
		unfix 12 
	
		fix 13 all nvt temp 600.0 600.0 100.0 
		run 5000
		unfix 13
	
		fix 14 all nvt temp 300.0 300.0 100.0 
		run 10000
		unfix 14
	
		fix 15 all npt tri 4934.6 4934.6 1000.0 temp 300.0 300.0 100.0
		run 5000
		unfix 15
	
		fix 16 all nvt temp 600.0 600.0 100.0
		run 5000
		unfix 16
	
		fix 17 all nvt temp 300.0 300.0 100.0 
		run 10000
		unfix 17
	
		fix 18 all npt tri 493.5 493.5 1000.0 temp 300.0 300.0 100.0
		run 5000 
		unfix 18
	
		fix 19 all nvt temp 600.0 600.0 100.0
		run 5000
		unfix 19
	
		fix 20 all nvt temp 300.0 300.0 100.0
		run 10000
		unfix 20
		
		fix 21 all npt tri 0.987 0.987 1000.0 temp 300.0 300.0 100.0
		run 800000
		''')
	
		if verbosity > 0: print("Finished writing input file!\n")
	
	stopTime = timeit.default_timer()
	print('Total build time: ', '%.3g' % (stopTime-startTime)+"s")
