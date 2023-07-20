import argparse
import math

from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from os import system, path
from scipy.spatial.distance import cdist # přepsat a zkontrolovat
from dataclasses import dataclass


xtb_settings_template = """$constrain
   atoms: xxx
   force constant=1.0
$end
"""


def load_arguments():
    print("\nParsing arguments... ", end="")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str, required=True,
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument('--data_dir', type=str, default="data_dir_prostrop",
                        help='Directory for saving results.')
    args = parser.parse_args()
    print("ok")
    return args


def prepare_directory(args):
    print("\nPreparing a data directory... ", end="")
    if path.exists(args.data_dir):
        exit(f"\n\nError! Directory with name {args.data_dir} exists. "
             f"Remove existed directory or change --data_dir argument.")
    system(f"mkdir {args.data_dir};"
           f"mkdir {args.data_dir}/inputed_PDB;"
           f"mkdir {args.data_dir}/optimized_PDB;"
           f"cp {args.pdb_file} {args.data_dir}/inputed_PDB;"
           f"cp {args.pdb_file} {args.data_dir}/optimized_PDB")
    print("ok\n")
    return f"{args.data_dir}/inputed_PDB/{path.basename(args.pdb_file)}", f"{args.data_dir}/optimized_PDB/{path.basename(args.pdb_file)}"


class SelectIndexedResidues(Select):
    def accept_residue(self, residue):
        if residue.id[1] in self.indices:
            return 1
        else:
            return 0

@dataclass
class Residuum:
    index: int
    constrained_atom_symbols: list # předělat na set
    non_constrained_atom_symbols: list # předělat na set
    constrained_atoms: list


class Substructure:
    def __init__(self,
                 optimized_residuum,
                 data_dir):
        optimized_residuum_index = optimized_residuum.id[1]
        self.substructure_data_dir = f"{data_dir}/sub_{optimized_residuum_index}"
        self.residues = []
        print(f"Optimization of {optimized_residuum_index}. residuum...", end="\r")
        kdtree = NeighborSearch(list(structure.get_atoms()))
        near_residues = sorted(kdtree.search(optimized_residuum.center_of_mass(geometric=True), 15, level="R"))
        self.flexible_residues = [] # smazat!!!!!
        for residuum_i in near_residues:
            distances = cdist([atom.coord for atom in residuum_i.get_atoms()],
                              [atom.coord for atom in optimized_residuum.get_atoms()])
            if distances.min() < 6:
                constrained_atoms = []
                non_constrained_atoms = []
                if residuum_i.id[1] == optimized_residuum_index:
                    non_constrained_atoms = optimized_residuum.get_atoms()
                    self.flexible_residues.append(optimized_residuum_index)
                else:
                    for atom_distances, atom in zip(distances, residuum_i.get_atoms()):
                        if atom_distances.min() > 4:
                            constrained_atoms.append(atom)
                        else:
                            non_constrained_atoms.append(atom)
                            self.flexible_residues.append(residuum_i.id[1])
                self.residues.append(Residuum(index=residuum_i.id[1],
                                              constrained_atom_symbols=[atom.name for atom in constrained_atoms],
                                              non_constrained_atom_symbols=[atom.name for atom in non_constrained_atoms],
                                              constrained_atoms=constrained_atoms))







        io.set_structure(structure)
        selector = SelectIndexedResidues()
        selector.indices = [residuum.index for residuum in self.residues]
        system(f"mkdir {self.substructure_data_dir}")
        io.save(f"{self.substructure_data_dir}/substructure.pdb", selector)

    def find_cutted_residues(self):
        self.cutted_residues = []
        all_indices = [res.index for res in self.residues]
        for index in all_indices:
            if index == 1:
                if 2 not in all_indices:
                    self.cutted_residues.append(index)
            elif index == len(list(structure.get_residues())):
                if len(list(structure.get_residues())) -1 not in all_indices:
                    self.cutted_residues.append(index)
            else:
                if index +1 not in all_indices or index -1 not in all_indices:
                    self.cutted_residues.append(index)

    def optimize(self):
        self.find_cutted_residues()
        system(f"cd {self.substructure_data_dir} ;"
               f"/home/dargen3/miniconda3/envs/babel_env/bin/obabel -h -ipdb -opdb substructure.pdb > reprotonated_substructure.pdb 2>&1")
        with open(f"{self.substructure_data_dir}/substructure.pdb") as substructure_file:
            substructure_lines = [line for line in substructure_file.readlines() if line[:4] == "ATOM"]
            num_of_atoms = len(substructure_lines)
        with open(f"{self.substructure_data_dir}/reprotonated_substructure.pdb") as reprotonated_substructure_file:
            added_atoms = [line for line in reprotonated_substructure_file.readlines() if line[:4] == "ATOM"][num_of_atoms:]
        with open(f"{self.substructure_data_dir}/repaired_substructure.pdb", "w") as repaired_substructure_file:
            repaired_substructure_file.write("".join(substructure_lines))
            ac = 0
            for line in added_atoms:
                res_i = int(line.split()[5])
                if res_i in self.cutted_residues:
                    if math.dist([float(x) for x in line.split()[6:9]], structure[res_i]["C"].coord) < 1.1:
                        repaired_substructure_file.write(line)
                        ac += 1
                    elif math.dist([float(x) for x in line.split()[6:9]], structure[res_i]["N"].coord) < 1.1:
                        repaired_substructure_file.write(line)
                        ac += 1


        repaired_substructure = pdb_parser.get_structure("repaired_substructure", f"{self.substructure_data_dir}/substructure.pdb")[0]["A"]
        self.constrained_atoms_indices = []
        c = 1
        for repaired_residuum, residuum in zip(repaired_substructure.get_residues(), self.residues):
            for atom in repaired_residuum.get_atoms():
                if atom.name not in residuum.non_constrained_atom_symbols:
                    self.constrained_atoms_indices.append(c)
                c += 1

        for x in range(c, ac+c): # constrain added hydrogens
            self.constrained_atoms_indices.append(x)



        substructure_settings = xtb_settings_template.replace("xxx", ", ".join([str(x) for x in self.constrained_atoms_indices]))
        with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
            xtb_settings_file.write(substructure_settings)
        system(f"cd {self.substructure_data_dir} ;"
               f"xtb repaired_substructure.pdb "
               f"--gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1")


    def update_PDB(self):
        substructure = pdb_parser.get_structure("substructure", f"{self.substructure_data_dir}/xtbopt.pdb")[0]
        substructure_residues = list(list(substructure.get_chains())[0].get_residues())
        non_constrained_atoms = []
        constrained_atoms = []
        for optimized_residuum, residuum in zip(substructure_residues, self.residues):
            for atom in optimized_residuum.get_atoms():
                if atom.name in residuum.non_constrained_atom_symbols:
                    non_constrained_atoms.append(atom)
                elif atom.name in residuum.constrained_atom_symbols:
                    constrained_atoms.append(atom)
        sup = Superimposer()
        sup.set_atoms([atom for residuum in self.residues for atom in residuum.constrained_atoms], constrained_atoms)
        sup.apply(non_constrained_atoms)


        for optimized_residuum, residuum in zip(substructure_residues, self.residues):
            for atom_symbol in residuum.non_constrained_atom_symbols:
                structure[int(residuum.index)][atom_symbol].set_coord(optimized_residuum[atom_symbol].coord)





def load_molecule(pdb_file):
    print(f"Loading of molecule from {pdb_file}... ", end="")
    structure = pdb_parser.get_structure("structure", pdb_file)[0]["A"]
    residues = list(structure.get_residues())
    print("ok\n")
    return structure, residues


if __name__ == '__main__':
    args = load_arguments()
    pdb_parser = PDBParser(QUIET=True)
    original_pdb, optimized_pdb = prepare_directory(args)
    structure, residues = load_molecule(args.pdb_file)
    io = PDBIO()

    for residuum in structure.get_residues():
        substructure = Substructure(residuum,
                                    args.data_dir)
        substructure.optimize()
        substructure.update_PDB()



    io.set_structure(structure)
    io.save(optimized_pdb)
    print(f"Structure succesfully optimized.\n")





# -476.929954898424

# -91.009391050480




# zkontrolovat optimalizaci kodu
# zkontrolovat čitelnost kodu
# error
# předělat na set





