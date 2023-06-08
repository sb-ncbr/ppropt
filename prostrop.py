"""export MKL_NUM_THREADS=1 ; export OMP_NUM_THREADS=1,1 ; export OMP_STACKSIZE=10G ; ulimit -s unlimited"""
from shutil import rmtree
from os import system, path
from time import time
import argparse
import numpy as np
from sklearn.neighbors import KDTree as kdtreen
import Bio.PDB
from collections import defaultdict
from termcolor import colored
from dataclasses import dataclass

xtb_settings = """$constrain
   atoms: xxx
   force constant=1.0
$end
"""

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


def load_arguments():
    print("\nParsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str, required=True,
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument('--data_dir', type=str, default="data_dir_prostrop",
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument("--rewrite", help="Rewrite data directory if exists.",
                        action="store_true")
    args = parser.parse_args()
    print(colored("ok", "green"))
    return args


def prepare_directory(args):
    print("\nPreparing a data directory...")
    if path.exists(args.data_dir):
        if args.rewrite:
            print(f"   Removing old data directory {args.data_dir}")
            rmtree(args.data_dir)
        else:
            exit(f"Error! Directory with name {args.data_dir} exists. Remove existed directory or change --data_dir argument.")
    system(f"mkdir {args.data_dir}")
    system(f"mkdir {args.data_dir}/inputed_PDB")
    system(f"mkdir {args.data_dir}/optimized_PDB")
    system(f"cp {args.pdb_file} {args.data_dir}/inputed_PDB")
    system(f"cp {args.pdb_file} {args.data_dir}/optimized_PDB")
    print(colored("ok\n", "green"))
    return f"{args.data_dir}/inputed_PDB/{args.pdb_file}", f"{args.data_dir}/optimized_PDB/{args.pdb_file}"


def sp_calculation(pdb_file, time):
    path = "/".join(pdb_file.split("/")[:-1])
    system(f"xtb --gfnff {pdb_file} --alpb water --verbose 2>&1 | grep 'TOTAL ENERGY' | awk '{{ print $4 }}' > {path}/xtb_sp_energy_{time}.txt ")
    system("rm gfnff_adjacency gfnff_topo gfnff_charges")
    sp_energy = float(open(f"{path}/xtb_sp_energy_{time}.txt").read())
    return sp_energy


@dataclass
class Residue:
    name: str
    n_ats: int
    coordinates_mean: float
    pdb_lines: list
    a_carbon_index: int


class Molecule:
    def __init__(self,
                 args):
        print(f"Loading of molecule from {args.pdb_file}...")
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("whole pdb", args.pdb_file)[0]
        pdb_lines = [line for line in open(args.pdb_file, "r").readlines() if line.split()[0] == "ATOM"]
        self.residues = []
        i = 0
        for residuum in list(list(structure.get_chains())[0].get_residues()):
            atom_names = [atom.fullname for atom in residuum.get_atoms()]
            alpha_carbon_index = atom_names.index(" CA ") + 1
            self.residues.append(Residue(residuum.resname,
                                         len(atom_names),
                                         residuum.center_of_mass(geometric=True),
                                         pdb_lines[i:i+len(atom_names)],
                                         alpha_carbon_index))
            i += len(atom_names)
        self.num_of_res = len(self.residues)
        self.res_kdtree = kdtreen([res.coordinates_mean for res in self.residues], leaf_size=50)
        print(f"   {self.num_of_res} residues loaded\n   Estimated calculation time xyz seconds")
        print(colored("ok\n", "green"))





class Substructure:
    def __init__(self,
                 optimized_residuum: Residue,
                 residuum_index: int,
                 molecule: Molecule,
                 args):
        self.residues = [optimized_residuum]
        self.constrained_atoms_indices = [2] # self.constrained_atoms_indices = [optimized_residuum.a_carbon_index]
        self.num_of_atoms = optimized_residuum.n_ats
        distances, indices = molecule.res_kdtree.query([optimized_residuum.coordinates_mean], k=molecule.num_of_res) # upravit na nějaké normální číslo a ne num of res
        for d,i in zip(distances[0][1:], indices[0][1:]):
            if d < amk_radius[optimized_residuum.name] + amk_radius[molecule.residues[i].name] + 2:
                self.residues.append(molecule.residues[i])
                for constrained_atom_index in range(self.num_of_atoms + 1, self.num_of_atoms + molecule.residues[i].n_ats + 1):
                    self.constrained_atoms_indices.append(constrained_atom_index)
                self.num_of_atoms += molecule.residues[i].n_ats

                # if fixed:
                #     for constrained_atom_index in range(self.num_of_atoms + 1, self.num_of_atoms + residuum.n_ats + 1):
                #         self.constrained_atoms_indices.append(constrained_atom_index)
                # else:
                #     self.constrained_atoms_indices.append(self.num_of_atoms + residuum.a_carbon_index + 1)


        self.pdb = "".join([at_line for res in self.residues for at_line in res.pdb_lines])

        self.data_dir = f"{args.data_dir}/sub_{residuum_index}"
        self.pdb_file = f"{self.data_dir}/sub_{residuum_index}.pdb"


    def optimize(self):
        system(f"mkdir {self.data_dir}")
        xtb_settings_file = f"{self.data_dir}/xtb_settings.inp"
        open(xtb_settings_file, "w").write(xtb_settings.replace("xxx", ", ".join([str(x) for x in self.constrained_atoms_indices])))
        open(substructure.pdb_file, "w").write(substructure.pdb)
        system(f"xtb --gfnff {self.pdb_file} --input {xtb_settings_file} --opt --alpb water --verbose > {self.data_dir}/xtb_output.txt 2>&1 ;"
               f" mv xtbopt.pdb xtbopt.log {self.data_dir} ;"
               f" rm gfnff_adjacency gfnff_topo gfnff_charges")


    def update_PDB(self, args, res_i, pdb, residues):
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("whole pdb", pdb)[0]
        substructure = pdb_parser.get_structure("substructure", f"{args.data_dir}/sub_{res_i}/xtbopt.pdb")[0]
        substructure_residues = list(list(substructure.get_chains())[0].get_residues())  # rewrite for more chains
        optimized_residuum_atoms = list(substructure_residues[0].get_atoms())
        nonoptimized_residues_atoms = [atom for residuum in substructure_residues[1:] for atom in residuum.get_atoms()]
        constrained_residues = [residuum.get_id()[1] for residuum in substructure_residues[1:]]

        structure_atoms = []
        for constrained_res_i in constrained_residues:
            for structure_res in list(structure.get_chains())[0]:
                if structure_res.get_id()[1] == constrained_res_i:
                    for structure_atom in structure_res:
                        structure_atoms.append(structure_atom)

        # superimpose
        sup = Bio.PDB.Superimposer()
        sup.set_atoms(structure_atoms, nonoptimized_residues_atoms)
        sup.apply(optimized_residuum_atoms + nonoptimized_residues_atoms)
        io = Bio.PDB.PDBIO()
        io.set_structure(substructure)
        io.save(f"{args.data_dir}/sub_{res_i}/imposed_{res_i}.pdb")

        # update residues
        for residuum in substructure_residues:
            res_coordinates = [substructure_atom.get_coord() for substructure_atom in residuum]
            res_mean_coordinates = np.mean(res_coordinates, axis=0)
            residues[residuum.get_id()[1] - 1].coordinates = res_coordinates
            residues[residuum.get_id()[1] - 1].coordinates_mean = res_mean_coordinates

        new_pdb_lines = defaultdict(list)

        pdb_lines = open(pdb, "r").readlines()
        for data_line, index_line in zip(open(f"{args.data_dir}/sub_{res_i}/imposed_{res_i}.pdb", "r").readlines(),
                                         open(f"{args.data_dir}/sub_{res_i}/sub_{res_i}.pdb", "r").readlines()):
            newline = index_line[:26] + data_line[26:]
            pdb_lines[int(index_line.split()[1]) - 1] = newline

            new_pdb_lines[int(index_line.split()[5]) - 1].append(newline)

        # update redisues pdb lines
        for key in new_pdb_lines.keys():
            residues[key].pdb_lines = new_pdb_lines[key]

        open(pdb, "w").write("".join(pdb_lines))


if __name__ == '__main__':
    args = load_arguments()
    original_pdb, optimized_pdb = prepare_directory(args)
    molecule = Molecule(args)

    print("Structure optimization...")
    original_structure_energy = sp_calculation(original_pdb, "before_optimization")
    s = time()
    for residuum_index, residuum in enumerate(molecule.residues, start=1):
        print(f"Optimization of {residuum_index}. residuum...", end="\r")
        substructure = Substructure(residuum,
                                    residuum_index,
                                    molecule,
                                    args)

        substructure.optimize()


        substructure.update_PDB(args, residuum_index, optimized_pdb, molecule.residues)




    print("                                               ", end="\r")
    print(colored("ok\n", "green"))



    optimized_structure_energy = sp_calculation(optimized_pdb, "after_optimization")


    print(f"\nRESULTS\nOriginal energy: {original_structure_energy}")
    print(f"Optimized energy: {optimized_structure_energy}")

    print(f"\nTotal time: {time()-s}s")
    print(f"Time per residue: {(time()-s) / molecule.num_of_res}s")














    # import pytraj as pt
    # pytraj_pdb = pt.load(original_pdb)
    # print(f"\n\n\noriginal structure number of H bonds: {pt.search_hbonds(pytraj_pdb)}")
    #
    # pytraj_pdb = pt.load(optimized_pdb)
    # print(f"optimized structure number of H bonds: {pt.search_hbonds(pytraj_pdb)}\n\n\n")



    # script = f"""load "myfiles" \"{optimized_pdb}\" \"{original_pdb}\" ; model 0 ;
    #                     select all; wireframe 0.08; spacefill 0.12;
    #
    #                     select 1.1 ; color [204,0,0];
    #                     select 2.1 ; color [102,204,0];"""
    # open(f"{args.data_dir}/jmol_script.txt", "w").write(script)
    # system(f" jmol -g1000x1000 -s {args.data_dir}/jmol_script.txt > /dev/null 2>&1 &")








"""
optimalizovat
1) uvolnít více atomů (celou peptidovou vazbu, i vedlejší AMK)
2) cutoff distance
3) level optimalizace


porovnávat s:
počet vodíkových vazeb
celková energie z xtb
nějaký experimentální údaj?
experimentální struktury?



todo:
předělat xtb na python library
přidat náboj (rdkit?)
přidat alphafill

užitečné odkazy:
https://xtb-python.readthedocs.io/en/latest/
https://github.com/grimme-lab/xtb-python
https://en.wikipedia.org/wiki/Alpha_helix
"""


# todo 1) odstranit rdkit
#      2) předělat xtb na python
#      3) udělat substructure