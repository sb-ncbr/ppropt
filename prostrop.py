import argparse
import Bio.PDB

from os import system, path
from time import time
from sklearn.neighbors import KDTree as kdtreen
from collections import defaultdict
from termcolor import colored
from dataclasses import dataclass
from scipy.spatial.distance import cdist

amk_radius = {'ALA': 2.4801,
              'ARG': 4.8618,
              'ASN': 3.2237,
              'ASP': 2.8036,
              'CYS': 2.5439,
              'GLN': 3.8456,
              'GLU': 3.3963,
              'GLY': 2.1455,
              'HIS': 3.8376,
              'ILE': 3.4050,
              'LEU': 3.5357,
              'LYS': 4.4521,
              'MET': 4.1821,
              'PHE': 4.1170,
              'PRO': 2.8418,
              'SER': 2.4997,
              'THR': 2.7487,
              'TRP': 4.6836,
              'TYR': 4.5148,
              'VAL': 2.9515}

xtb_settings = """$constrain
   atoms: xxx
   force constant=1.0
$end
"""


# xtb_settings = """$constrain
#   elements: 6,7,8,16
#   atoms: xxx
#   force constant=1.0
# $end
# """

def load_arguments():
    print("\nParsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str, required=True,
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument('--data_dir', type=str, default="data_dir_prostrop",
                        help='Directory for saving results.')
    parser.add_argument("--rewrite", help="Rewrite data directory if exists.",
                        action="store_true")
    parser.add_argument("--sp_calculation", help="Run single point calculation before and after optimization "
                                                 "to compare energy. For debugging only.",
                        action="store_true")
    args = parser.parse_args()
    print(colored("ok", "green"))
    return args


def prepare_directory(args):
    print("\nPreparing a data directory...")
    if path.exists(args.data_dir):
        if args.rewrite:
            print(f"   Removing old data directory {args.data_dir}")
            system(f"rm -r {args.data_dir}")
        else:
            exit(f"Error! Directory with name {args.data_dir} exists. "
                 f"Remove existed directory or change --data_dir argument.")
    system(f"mkdir {args.data_dir};"
           f"mkdir {args.data_dir}/inputed_PDB;"
           f"mkdir {args.data_dir}/optimized_PDB;"
           f"cp {args.pdb_file} {args.data_dir}/inputed_PDB;"
           f"cp {args.pdb_file} {args.data_dir}/optimized_PDB")
    print(colored("ok\n", "green"))
    return f"{args.data_dir}/inputed_PDB/{args.pdb_file}", f"{args.data_dir}/optimized_PDB/{args.pdb_file}"


def sp_calculation(pdb_file, time):
    path = "/".join(pdb_file.split("/")[:-1])
    system(f"xtb --gfnff {pdb_file} --alpb water --verbose 2>&1 | grep 'TOTAL ENERGY' | awk '{{ print $4 }}'"
           f" > {path}/xtb_sp_energy_{time}.txt;"
           f"rm gfnff_adjacency gfnff_topo gfnff_charges")
    sp_energy = float(open(f"{path}/xtb_sp_energy_{time}.txt").read())
    return sp_energy


@dataclass
class Residue:
    name: str
    index: int
    n_ats: int
    coordinates_mean: float
    coordinates: list  # možná smazat?
    pdb_lines: list
    ca_index: int



class Molecule:
    def __init__(self,
                 args):
        print(f"Loading of molecule from {args.pdb_file}...")
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        residues = list(list(pdb_parser.get_structure("whole pdb", args.pdb_file)[0].get_chains())[0].get_residues())
        pdb_lines = [line for line in open(args.pdb_file, "r").readlines() if line.split()[0] == "ATOM"]
        self.residues = []
        pdb_line_index = 0
        for residuum_index, residuum in enumerate(residues):
            atom_names = [atom.name for atom in residuum.get_atoms()]
            self.residues.append(Residue(residuum.resname,
                                         residuum_index,
                                         len(atom_names),
                                         residuum.center_of_mass(geometric=True),
                                         [a.coord for a in residuum.get_atoms()],
                                         pdb_lines[pdb_line_index:pdb_line_index + len(atom_names)],
                                         atom_names.index("CA") + 1))
            pdb_line_index += len(atom_names)
        self.num_of_res = len(self.residues)
        self.res_kdtree = kdtreen([res.coordinates_mean for res in self.residues], leaf_size=50)
        print(f"   {self.num_of_res} residues loaded\n   Estimated calculation time xyz seconds")
        print(colored("ok\n", "green"))


class Substructure:
    def __init__(self,
                 optimized_residuum: Residue,
                 molecule: Molecule,
                 args):
        self.residuum_index = optimized_residuum.index + 1
        self.constrained_atoms_indices = [optimized_residuum.ca_index]
        self.pdb = "".join([at_line for at_line in optimized_residuum.pdb_lines])
        self.num_of_atoms = optimized_residuum.n_ats
        indices = molecule.res_kdtree.query_radius([optimized_residuum.coordinates_mean], 16)[0]
        for i in indices:
            if i == optimized_residuum.index: # the optimizing residue is already processed
                continue
            res_i = molecule.residues[i]
            if cdist(optimized_residuum.coordinates, res_i.coordinates).min() < 6:
                c = 1
                self.constrained_atoms_indices.append(self.num_of_atoms + res_i.ca_index)
                for constrained_atom_index in range(self.num_of_atoms + 1,
                                                    self.num_of_atoms + res_i.n_ats + 1):
                    if min(cdist([res_i.coordinates[c - 1]], optimized_residuum.coordinates)[0]) > 4:
                        self.constrained_atoms_indices.append(constrained_atom_index)
                    c += 1
                self.num_of_atoms += res_i.n_ats
                self.pdb += "".join([at_line for at_line in res_i.pdb_lines])
        self.data_dir = f"{args.data_dir}/sub_{optimized_residuum.index + 1}"
        self.pdb_file = f"{self.data_dir}/sub_{optimized_residuum.index + 1}.pdb"

    def optimize(self):
        system(f"mkdir {self.data_dir}")
        xtb_settings_file_name = f"{self.data_dir}/xtb_settings.inp"
        with open(xtb_settings_file_name, "w") as xtb_settings_file:
            xtb_settings_file.write(xtb_settings.replace("xxx", ", ".join([str(x) for x in self.constrained_atoms_indices])))
        with open(self.pdb_file, "w") as pdb_file:
            pdb_file.write(self.pdb)
        system(f"xtb --gfnff {self.pdb_file} --input {xtb_settings_file_name} --opt --alpb water --verbose > "
               f"{self.data_dir}/xtb_output.txt 2>&1 ;"
               f" mv xtbopt.pdb xtbopt.log {self.data_dir} ;"
               f" rm gfnff_adjacency gfnff_topo gfnff_charges")

    def update_PDB(self, args, pdb, residues):
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)  # zkusit dát mimo
        structure_atoms = list(pdb_parser.get_structure("pdb", pdb)[0].get_atoms())
        substructure = pdb_parser.get_structure("substructure", f"{args.data_dir}/sub_{self.residuum_index}/xtbopt.pdb")[0]
        substructure_residues = list(list(substructure.get_chains())[0].get_residues())  # rewrite for more chains
        substructure_pdb_lines = open(f"{args.data_dir}/sub_{self.residuum_index}/sub_{self.residuum_index}.pdb", "r").readlines()  # přepsat
        non_constrained_atoms = []
        constrained_atoms = []
        structure_constrained_atoms = []
        c = 1
        for residue in substructure_residues:
            for atom in list(residue.get_atoms()):
                if c in self.constrained_atoms_indices:
                    constrained_atoms.append(atom)
                    structure_constrained_atoms.append(structure_atoms[int(substructure_pdb_lines[c - 1].split()[1]) - 1])
                else:
                    non_constrained_atoms.append(atom)
                c += 1
        sup = Bio.PDB.Superimposer()
        sup.set_atoms(structure_constrained_atoms, constrained_atoms)
        sup.apply(non_constrained_atoms + constrained_atoms)
        io = Bio.PDB.PDBIO()
        io.set_structure(substructure)
        io.save(f"{args.data_dir}/sub_{self.residuum_index}/imposed_{self.residuum_index}.pdb")

        for residuum in substructure_residues:
            residues[residuum.get_id()[1] - 1].coordinates_mean = residuum.center_of_mass(geometric=True)

        new_pdb_lines = defaultdict(list)
        pdb_lines = open(pdb, "r").readlines()
        for data_line, index_line in zip(open(f"{args.data_dir}/sub_{self.residuum_index}/imposed_{self.residuum_index}.pdb", "r").readlines(),
                                         substructure_pdb_lines):
            newline = index_line[:26] + data_line[26:]
            pdb_lines[int(index_line.split()[1]) - 1] = newline
            new_pdb_lines[int(index_line.split()[5]) - 1].append(newline)

        # update residues pdb lines
        for key in new_pdb_lines.keys():
            residues[key].pdb_lines = new_pdb_lines[key]
        open(pdb, "w").write("".join(pdb_lines))





if __name__ == '__main__':
    s = time()
    args = load_arguments()
    original_pdb, optimized_pdb = prepare_directory(args)
    molecule = Molecule(args)
    print("Structure optimization...")

    for residuum in molecule.residues:
        print(f"Optimization of {residuum.index + 1}. residuum...", end="\r")
        substructure = Substructure(residuum,
                                    molecule,
                                    args)
        substructure.optimize()
        substructure.update_PDB(args,
                                optimized_pdb,
                                molecule.residues)
    print("                                               ", end="\r")
    print(colored("ok\n", "green"))
    print("\nRESULTS:")
    if args.sp_calculation:
        print(f"\nOriginal structure energy: {sp_calculation(original_pdb, 'before_optimization')}")
        print(f"Optimized structure energy: {sp_calculation(optimized_pdb, 'after_optimization')}")
    print(f"\nTotal time: {time() - s}s")
    print(f"Time per residue: {(time() - s) / molecule.num_of_res}s")


    # import pytraj as pt
    # pytraj_pdb = pt.load(original_pdb)
    # print(f"\n\n\noriginal structure number of H bonds: {pt.search_hbonds(pytraj_pdb)}")
    #
    # pytraj_pdb = pt.load(optimized_pdb)
    # print(f"optimized structure number of H bonds: {pt.search_hbonds(pytraj_pdb)}\n\n\n")

    #
    # script = f"""load "myfiles" \"{optimized_pdb}\" \"{original_pdb}\" ; model 0 ;
    #                     select all; wireframe 0.08; spacefill 0.12;
    #
    #                     select 1.1 ; color [204,0,0];
    #                     select 2.1 ; color [102,204,0];"""
    # open(f"{args.data_dir}/jmol_script.txt", "w").write(script)
    # system(f" jmol -g1000x1000 -s {args.data_dir}/jmol_script.txt > /dev/null 2>&1 &")


# přepsat ať v první iteraci se optimalizují vodíky (a možná i peptidická páteř?)
# zkontrolovat optimalizaci kodu
# přidat virtuální vodíky
# dopsat aby pdb file mohl být kdekoliv


#  předoptimalizovat vodíky - ano
#  více iterací - ano, zjistit kolik


# https://pubs.acs.org/doi/10.1021/ct400065j


# do článku
# ukázat zrychlení
# ukázat, že to konverguje ke stejnému minimu
# graf, kde ukážeme energii před optimalizací, po optimalizaci vodíků, po optimalizaci prostrop (možná více iterací), energie celé struktury
# porovnání úhlů, délek vazeb atd..
