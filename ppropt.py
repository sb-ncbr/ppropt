import argparse
from dataclasses import dataclass
import json
from os import system, path

from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from Bio import SeqUtils
from scipy.spatial.distance import cdist

import numba
import numpy as np


def load_arguments():
    print("\nParsing arguments... ", end="")
    parser = argparse.ArgumentParser()
    parser.add_argument('--PDB_file', type=str, required=True,
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory for saving results.')
    args = parser.parse_args()
    if not path.isfile(args.PDB_file):
        print(f"\nERROR! File {args.PDB_file} does not exist!\n")
        exit()
    print("ok")
    return args


class SelectIndexedResidues(Select):
    def accept_residue(self, residue):
        if residue.id[1] in self.indices:
            return 1
        else:
            return 0


@dataclass
class Residue:
    index: int
    constrained_atom_symbols: set
    non_constrained_atom_symbols: set
    constrained_atoms: list


@numba.jit(cache=True, nopython=True, fastmath=True, boundscheck=False, nogil=True)
def numba_dist(optimized_residue, residue):
     distances = np.empty(len(residue))
     mins = np.empty(len(optimized_residue))
     for i,a in enumerate(optimized_residue):
         for j,b in enumerate(residue):
             distances[j] = ((a[0]-b[0])**2 + (a[1]-b[1])**2 +(a[2]-b[2])**2 )**(1/2)
         mins[i] = distances.min()
     return mins, mins.min()


class Substructure:
    def __init__(self,
                 optimized_residue,
                 PRO):
        # print(f"Optimization of {optimized_residue.id[1]}. residue...", end="\r")

        self.optimized_residue = optimized_residue
        self.PRO = PRO
        self.substructure_data_dir = f"{PRO.data_dir}/sub_{self.optimized_residue.id[1]}"
        self.PDBParser = PDBParser(QUIET=True)
        self.residues = []
        self.constrained_atoms_indices = []
        # near_residues = sorted(PRO.kdtree.search(optimized_residue.center_of_mass(geometric=True), 12, level="R"))
        near_atoms = sorted(PRO.kdtree.search(optimized_residue.center_of_mass(geometric=True), 15, level="A"))
        counter_atoms = 1  # start from 1 because of xtb countering
        counter_residues = 0
        # from time import time
        # s = time()

        for residue in enumerate(near_residues, start=1):
            if residue == self.optimized_residue:
                self.substructure_optimized_residue_index = counter_residues

            mins, total_min = numba_dist(np.array([atom.coord for atom in residue.get_atoms()]), np.array([atom.coord for atom in optimized_residue.get_atoms()]))

            if total_min < 5:
                counter_residues += 1
                constrained_atoms = []
                non_constrained_atoms_symbols = set()
                for atom_distance, atom in zip(mins, residue.get_atoms()):
                    if atom.name == "CA" or atom_distance > 3:
                        constrained_atoms.append(atom)
                        self.constrained_atoms_indices.append(str(counter_atoms))
                    else:
                        non_constrained_atoms_symbols.add(atom.name)
                    counter_atoms += 1
                self.residues.append(Residue(index=residue.id[1],
                                             constrained_atom_symbols={atom.name for atom in constrained_atoms},
                                             non_constrained_atom_symbols=non_constrained_atoms_symbols,
                                             constrained_atoms=constrained_atoms))
        self.num_of_atoms = counter_atoms - 1 # smazat!!
        print(self.num_of_atoms)
        self.residues_indices = {res.index for res in self.residues}
        # print(time()-s)
        system(f"mkdir {self.substructure_data_dir}")
        selector = SelectIndexedResidues()
        selector.indices = set([residue.index for residue in self.residues])
        self.PRO.io.save(f"{self.substructure_data_dir}/substructure.pdb", selector)


    def optimize(self):
        xtb_settings_template = """$constrain
   atoms: xxx
   force constant=1.0
$end
$opt
    engine=rf
$end
"""
        substructure_settings = xtb_settings_template.replace("xxx", ", ".join(self.constrained_atoms_indices))
        with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
            xtb_settings_file.write(substructure_settings)
        from time import time
        ss = time()
        system(f"cd {self.substructure_data_dir} ;"
               f"ulimit -s unlimited ;"
               f"export OMP_NUM_THREADS=1,1 ;"
               f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
               f"export MKL_NUM_THREADS=1 ;"
               f"xtb substructure.pdb --gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        if not path.isfile(f"{self.substructure_data_dir}/xtbopt.pdb"): # second try by L-ANCOPT
            substructure_settings = xtb_settings_template.replace("xxx", ", ".join(self.constrained_atoms_indices)).replace("rf", "lbfgs")
            with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                xtb_settings_file.write(substructure_settings)
            system(f"cd {self.substructure_data_dir} ;"
                   f"ulimit -s unlimited ;"
                   f"export OMP_NUM_THREADS=1,1 ;"
                   f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
                   f"export MKL_NUM_THREADS=1 ;"
                   f"xtb substructure.pdb --gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        print(time()-ss)

        if not path.isfile(f"{self.substructure_data_dir}/xtbopt.pdb"):
            print(f"\n\nWarning! {self.optimized_residue.id[1]}. residue skipped due to convergence issues. \n\n")
            return False
        return True

    def update_PDB(self):

        optimized_substructure = self.PDBParser.get_structure("substructure", f"{self.substructure_data_dir}/xtbopt.pdb")[0]
        optimized_substructure_residues = list(list(optimized_substructure.get_chains())[0].get_residues())

        constrained_atoms = []
        for optimized_residue, residue in zip(optimized_substructure_residues, self.residues):
            for atom in optimized_residue.get_atoms():
                if atom.name in residue.constrained_atom_symbols:
                    constrained_atoms.append(atom)

        # superimpose
        sup = Superimposer()
        sup.set_atoms([atom for residue in self.residues for atom in residue.constrained_atoms], constrained_atoms)
        sup.apply(optimized_substructure.get_atoms())
        for optimized_residue, residue in zip(optimized_substructure_residues, self.residues):
            for atom_symbol in residue.non_constrained_atom_symbols:
                self.PRO.structure[int(residue.index)][atom_symbol].set_coord(optimized_residue[atom_symbol].coord)


class PRO:
    def __init__(self,
                 data_dir: str,
                 PDB_file: str):
        self.data_dir = data_dir
        self.PDB_file = PDB_file

    def optimize(self):
        self._prepare_directory()
        self._load_molecule()
        self._calculate_depth_of_residues()
        # optimize residues from the most embedded residues
        logs = []
        c = 0
        for res_i in sorted(range(len(self.residues_depth)), key=self.residues_depth.__getitem__, reverse=True):
            from time import time
            s = time()
            residue = self.residues[res_i]
            substructure = Substructure(residue,
                                        self)
            optimized = substructure.optimize()
            if optimized:
                substructure.update_PDB()
            logs.append({"residue index": res_i+1,
                         "residue name": SeqUtils.IUPACData.protein_letters_3to1[residue.resname.capitalize()],
                         "optimized": optimized})
            print(time()-s)
            print("\n\n")
            c += 1
            if c == 10:
                exit()
        with open(f"{self.data_dir}/residues.logs", "w") as residues_logs:
            logs = json.dumps(sorted(logs, key=lambda x: x['residue index']), indent=2)
            residues_logs.write(logs)
        self.io.save(f"{self.data_dir}/optimized_PDB/{path.basename(self.PDB_file[:-4])}_optimized.pdb")
        print(f"Structure succesfully optimized.\n")

    def _prepare_directory(self):
        print("\nPreparing a data directory... ", end="")
        if path.exists(self.data_dir):
            exit(f"\n\nError! Directory with name {self.data_dir} exists. "
                 f"Remove existed directory or change --data_dir argument.")
        system(f"mkdir {self.data_dir};"
               f"mkdir {self.data_dir}/inputed_PDB;"
               f"mkdir {self.data_dir}/optimized_PDB;"
               f"cp {self.PDB_file} {self.data_dir}/inputed_PDB")
        print("ok\n")

    def _load_molecule(self):
        print(f"Loading of structure from {self.PDB_file}... ", end="")
        try:
            structure = PDBParser(QUIET=True).get_structure("structure", self.PDB_file)
            io = PDBIO()
            io.set_structure(structure)
            self.io = io
            self.structure = io.structure[0]["A"]

        except KeyError:
            print(f"\nERROR! PDB file {self.PDB_file} does not contain any structure.\n")
            exit()
        self.residues = list(self.structure.get_residues())
        print("ok\n")

    def _calculate_depth_of_residues(self):
        self.kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.residues_depth = [len(self.kdtree.search(residue.center_of_mass(geometric=True), 15, level="A"))
                               for residue in self.residues]


if __name__ == '__main__':
    args = load_arguments()
    PRO(args.data_dir, args.PDB_file).optimize()
