import argparse
from dataclasses import dataclass
import json
from os import system, path

from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from scipy.spatial.distance import cdist


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


class Substructure:
    def __init__(self,
                 optimized_residue,
                 PRO):
        print(f"Optimization of {optimized_residue.id[1]}. residue...", end="\r")
        self.optimized_residue = optimized_residue
        self.PRO = PRO
        self.substructure_data_dir = f"{PRO.data_dir}/sub_{self.optimized_residue.id[1]}"
        self.PDBParser = PDBParser(QUIET=True)
        self.residues = []
        self.constrained_atoms_indices = []
        near_residues = sorted(PRO.kdtree.search(optimized_residue.center_of_mass(geometric=True), 15, level="R"))
        counter_atoms = 1  # start from 1 because of xtb countering
        counter_residues = 0
        for res_i, residue in enumerate(near_residues, start=1):
            if residue == self.optimized_residue:
                self.substructure_optimized_residue_index = counter_residues
            distances = cdist([atom.coord for atom in residue.get_atoms()],
                              [atom.coord for atom in optimized_residue.get_atoms()])
            if distances.min() < 6:
                counter_residues += 1
                constrained_atoms = []
                non_constrained_atoms = []
                for atom_distances, atom in zip(distances, residue.get_atoms()):
                    if atom.name == "CA" or atom_distances.min() > 4:
                        constrained_atoms.append(atom)
                        self.constrained_atoms_indices.append(str(counter_atoms))
                    else:
                        non_constrained_atoms.append(atom)
                    counter_atoms += 1
                self.residues.append(Residue(index=residue.id[1],
                                             constrained_atom_symbols={atom.name for atom in constrained_atoms},
                                             non_constrained_atom_symbols={atom.name for atom in non_constrained_atoms},
                                             constrained_atoms=constrained_atoms))
        self.num_of_atoms = counter_atoms - 1
        self.residues_indices = {res.index for res in self.residues}
        io = PDBIO()
        io.set_structure(self.PRO.structure)
        selector = SelectIndexedResidues()
        selector.indices = [residue.index for residue in self.residues]
        system(f"mkdir {self.substructure_data_dir}")
        io.save(f"{self.substructure_data_dir}/substructure.pdb", selector)
        self._find_cutted_residues()
        self._add_hydrogens()

    def _find_cutted_residues(self):
        self.cutted_residues = []
        for index in self.residues_indices:
            if index == 1:
                if 2 not in self.residues_indices:
                    self.cutted_residues.append(index)
            elif index == len(self.PRO.residues):
                if len(self.PRO.residues) - 1 not in self.residues_indices:
                    self.cutted_residues.append(index)
            else:
                if index + 1 not in self.residues_indices or index - 1 not in self.residues_indices:
                    self.cutted_residues.append(index)

    def _add_hydrogens(self):
        system(f"cd {self.substructure_data_dir} ;"
               f"obabel -h -iPDB -oPDB substructure.pdb > reprotonated_substructure.pdb 2>/dev/null")
        with open(f"{self.substructure_data_dir}/reprotonated_substructure.pdb") as reprotonated_substructure_file:
            atom_lines = [line for line in reprotonated_substructure_file.readlines() if line[:4] == "ATOM"]
            original_atoms = atom_lines[:self.num_of_atoms]
            added_atoms = atom_lines[self.num_of_atoms:]
        with open(f"{self.substructure_data_dir}/repaired_substructure.pdb", "w") as repaired_substructure_file:
            repaired_substructure_file.write("".join(original_atoms))
            added_hydrogens_counter = 0
            for line in added_atoms:
                res_i = int(line.split()[5])
                if res_i in self.cutted_residues:
                    if any((cdist(([float(x) for x in [line[30:38], line[38:46], line[46:54]]],), (self.PRO.structure[res_i]["C"].coord,)) < 1.1,
                            cdist(([float(x) for x in [line[30:38], line[38:46], line[46:54]]],), (self.PRO.structure[res_i]["N"].coord,)) < 1.1)):
                        repaired_substructure_file.write(line)
                        added_hydrogens_counter += 1
        for at_i in range(self.num_of_atoms + 1, self.num_of_atoms + 1 + added_hydrogens_counter):  # constrain added hydrogens
            self.constrained_atoms_indices.append(str(at_i))

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
        system(f"cd {self.substructure_data_dir} ;"
               f"xtb repaired_substructure.pdb "
               f"--gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        if not path.isfile(f"{self.substructure_data_dir}/xtbopt.pdb"): # second try by L-ANCOPT
            substructure_settings = xtb_settings_template.replace("xxx", ", ".join(self.constrained_atoms_indices)).replace("rf", "lbfgs")
            with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                xtb_settings_file.write(substructure_settings)
            system(f"cd {self.substructure_data_dir} ;"
                   f"xtb repaired_substructure.pdb "
                   f"--gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        if not path.isfile(f"{self.substructure_data_dir}/xtbopt.pdb"):
            print(f"\n\nWarning! {self.optimized_residue.id[1]}. residue skipped due to convergence issues. \n\n")
            return False
        return True

    def update_PDB(self):
        optimized_substructure = self.PDBParser.get_structure("substructure", f"{self.substructure_data_dir}/xtbopt.pdb")[0]
        optimized_substructure_residues = list(list(optimized_substructure.get_chains())[0].get_residues())
        non_constrained_atoms = []
        constrained_atoms = []
        for optimized_residue, residue in zip(optimized_substructure_residues, self.residues):
            for atom in optimized_residue.get_atoms():
                if atom.name in residue.non_constrained_atom_symbols:
                    non_constrained_atoms.append(atom)
                elif atom.name in residue.constrained_atom_symbols:
                    constrained_atoms.append(atom)
                # added hydrogens are ignored

        # superimpose
        sup = Superimposer()
        sup.set_atoms([atom for residue in self.residues for atom in residue.constrained_atoms], constrained_atoms)
        sup.apply(optimized_substructure.get_atoms())
        optimized_residue_coordinates = [atom.coord for atom in optimized_substructure_residues[self.substructure_optimized_residue_index]]
        for optimized_residue, residue in zip(optimized_substructure_residues, self.residues):
            # update position of non-constrained atoms
            for atom_symbol in residue.non_constrained_atom_symbols:
                self.PRO.structure[int(residue.index)][atom_symbol].set_coord(optimized_residue[atom_symbol].coord)
            # update position of constrained atoms
            for atom_symbol in residue.constrained_atom_symbols:
                ox, oy, oz = self.PRO.structure[int(residue.index)][atom_symbol].coord
                nx, ny, nz = optimized_residue[atom_symbol].coord  # o means original
                distance = cdist([optimized_residue[atom_symbol].coord],  # n means new
                                      optimized_residue_coordinates).min()
                mx = 6  # maximum
                if distance < mx:
                    opt_x = ox - (ox-nx) * (1-distance/mx)
                    opt_y = oy - (oy-ny) * (1-distance/mx)
                    opt_z = oz - (oz-nz) * (1-distance/mx)
                    self.PRO.structure[int(residue.index)][atom_symbol].set_coord((opt_x, opt_y, opt_z))


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
        for res_i in sorted(range(len(self.residues_depth)), key=self.residues_depth.__getitem__, reverse=True):
            residue = self.residues[res_i]
            substructure = Substructure(residue,
                                        self)
            optimized = substructure.optimize()
            if optimized:
                substructure.update_PDB()
            logs.append({"residue index": res_i+1,
                         "residue name": residue.resname,
                         "optimized": optimized})
        with open(f"{self.data_dir}/residues.logs", "w") as residues_logs:
            logs = json.dumps(sorted(logs, key=lambda x: x['residue index']), indent=2)
            residues_logs.write(logs)
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(f"{self.data_dir}/optimized_PDB/{path.basename(self.PDB_file[:-4])}_optimized.pdb")
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
            self.structure = PDBParser(QUIET=True).get_structure("structure", self.PDB_file)[0]["A"]
        except KeyError:
            print(f"\nERROR! PDB file {self.PDB_file} does not contain any structure.\n")
            exit()
        self.residues = list(self.structure.get_residues())
        print("ok\n")

    def _calculate_depth_of_residues(self):
        self.kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.residues_depth = [len(self.kdtree.search(residue.center_of_mass(geometric=True), 15, level="R"))
                               for residue in self.residues]


if __name__ == '__main__':
    args = load_arguments()
    PRO(args.data_dir, args.PDB_file).optimize()



# od Tomáše
# upravit readme ať to instaluje z requirements
# pathlib


# zkusit jak se to chová s convergence errorem
# dopsat ať jde poznat, které reziduum bylo a nebylo optimalizované

# do článku
# limitations
# openbabel, version

# do readme
# limitations, requirements, instalatin, usage do readme

# udělat requirements.txt soubor
# do requirements přidat babel!!! přepsat verzi v článku