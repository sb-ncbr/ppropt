import argparse
import Bio.PDB
from os import system, path
from time import time
from sklearn.neighbors import KDTree as kdtreen
from scipy.spatial.distance import cdist

xtb_settings_template = """$constrain
   atoms: xxx
   force constant=1.0
$end
"""

def load_arguments():
    print("\nParsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str, required=True,
                        help='PDB file with structure, which should be optimized.')
    parser.add_argument('--data_dir', type=str, default="data_dir_prostrop",
                        help='Directory for saving results.')
    parser.add_argument("--rewrite", help="Rewrite data directory if exists.",
                        action="store_true")
    args = parser.parse_args()
    print("ok")
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
    print("ok\n")
    return f"{args.data_dir}/inputed_PDB/{args.pdb_file}", f"{args.data_dir}/optimized_PDB/{args.pdb_file}"


class SelectIndexedResidues(Bio.PDB.Select):
    def accept_residue(self, residue):
        if residue.id[1] in self.indices:
            return 1
        else:
            return 0


class Substructure:
    def __init__(self,
                 optimized_residuum,
                 structure,
                 kdtree: kdtreen,
                 data_dir: str):
        self.optimized_residuum = optimized_residuum
        self.optimized_residuum_index = self.optimized_residuum.id[1]
        self.structure = structure
        self.data_dir = data_dir
        self.constrained_atoms_indices = []
        self.structure_constrained_atoms = []
        self.substructure_residues_indices = []
        print(f"Optimization of {self.optimized_residuum_index}. residuum...", end="\r")
        atom_counter = 0
        optimized_residuum_coordinates = [atom.coord for atom in optimized_residuum.get_atoms()]
        indices = sorted(kdtree.query_radius([self.optimized_residuum.center_of_mass(geometric=True)], 15)[0] + 1)
        for i in indices:
            res_i = self.structure[int(i)]
            res_i_atoms = list(res_i.get_atoms())
            res_i_atom_names = [atom.name for atom in res_i.get_atoms()]
            res_i_coordinates = [atom.coord for atom in res_i.get_atoms()]

            if i == self.optimized_residuum_index:
                # self.constrained_atoms_indices.append(atom_counter + res_i_atom_names.index("CA") + 1)
                # self.structure_constrained_atoms.append(res_i["CA"])
                self.substructure_residues_indices.append(i)
                atom_counter += len(res_i_atom_names)

            elif cdist(optimized_residuum_coordinates, res_i_coordinates).min() < 6:
                c = 0
                for constrained_atom_index in range(atom_counter + 1,
                                                    atom_counter + len(res_i_atom_names) + 1):
                    if min(cdist([res_i_coordinates[c]], optimized_residuum_coordinates)[0]) > 4:
                        self.constrained_atoms_indices.append(constrained_atom_index)
                        self.structure_constrained_atoms.append(res_i_atoms[c])
                    # elif list(res_i.get_atoms())[c].name == "CA":
                    #     self.constrained_atoms_indices.append(constrained_atom_index)
                    #     self.structure_constrained_atoms.append(res_i_atoms[c])
                    c += 1
                self.substructure_residues_indices.append(i)
                atom_counter += len(res_i_atoms)

        self.substructure_data_dir = f"{self.data_dir}/sub_{self.optimized_residuum_index}"
        self.pdb_file = f"{self.substructure_data_dir}/sub_{self.optimized_residuum_index}.pdb"
        io = Bio.PDB.PDBIO()
        io.set_structure(self.structure)
        selector = SelectIndexedResidues()
        selector.indices = self.substructure_residues_indices
        system(f"mkdir {self.substructure_data_dir}")
        io.save(self.pdb_file, selector)


    def optimize(self):
        xtb_settings_file_name = f"{self.substructure_data_dir}/xtb_settings.inp"
        with open(xtb_settings_file_name, "w") as xtb_settings_file:
            xtb_settings_file.write(xtb_settings_template.replace("xxx", ", ".join([str(x) for x in self.constrained_atoms_indices])))
        system(f"xtb --gfnff {self.pdb_file} --input {xtb_settings_file_name} --opt --alpb water --verbose > "
               f"{self.substructure_data_dir}/xtb_output.txt 2>&1 ;"
               f" mv xtbopt.pdb xtbopt.log {self.substructure_data_dir} ;"
               f" rm gfnff_adjacency gfnff_topo gfnff_charges")

    def update_PDB(self):
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)  # zkusit dát mimo
        substructure = pdb_parser.get_structure("substructure", f"{self.substructure_data_dir}/xtbopt.pdb")[0]
        substructure_residues = list(list(substructure.get_chains())[0].get_residues())  # rewrite for more chains
        non_constrained_atoms = []
        constrained_atoms = []
        c = 1
        for residue in substructure_residues:
            for atom in list(residue.get_atoms()):
                if c in self.constrained_atoms_indices:
                    constrained_atoms.append(atom)
                else:
                    non_constrained_atoms.append(atom)
                c += 1
        sup = Bio.PDB.Superimposer()
        sup.set_atoms(self.structure_constrained_atoms, constrained_atoms)
        sup.apply(non_constrained_atoms + constrained_atoms)
        substructre_atoms = list(substructure.get_atoms())
        c = 0
        for res_index in self.substructure_residues_indices:
            for atom in structure[int(res_index)]:
                if c+1 not in self.constrained_atoms_indices:
                    atom.set_coord(substructre_atoms[c].coord)
                c += 1

        io = Bio.PDB.PDBIO()
        io.set_structure(substructure)
        io.save(f"{self.substructure_data_dir}/imposed_{self.optimized_residuum_index}.pdb")


def load_molecule(pdb_file):
    print(f"Loading of molecule from {pdb_file}...")
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure("structure", pdb_file)[0]["A"]
    residues = list(structure.get_residues())
    kdtree = kdtreen([residuum.center_of_mass(geometric=True) for residuum in residues], leaf_size=50)
    print("ok\n")
    return structure, residues, kdtree


if __name__ == '__main__':
    args = load_arguments()
    original_pdb, optimized_pdb = prepare_directory(args)
    structure, residues, kdtree = load_molecule(args.pdb_file)
    s = time()
    print("Structure optimization...")
    for residuum in structure.get_residues():
        substructure = Substructure(residuum,
                                    structure,
                                    kdtree,
                                    args.data_dir)
        substructure.optimize()
        substructure.update_PDB()
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(optimized_pdb)
    print("                                               ", end="\r")
    print("ok\n")
    print("\nRESULTS:")
    print(f"\nTotal time: {time() - s}s")
    print(f"Time per residue: {(time() - s) / len(list(structure.get_residues()))}s")








# rozhodnout
# jaký použít cutoff
# přidávat vodíky?


# implementovat
# parallelizovatelnost
# dopsat aby pdb file mohl být kdekoliv
# zkontrolovat optimalizaci kodu
# zkontrolovat čitelnost kodu
# nahradit kdtree z bio
# předělat ať je to schopno pracovat s více chainy


# do článku
# ukázat zrychlení a změnu energie mezi full ff optimalizací a prostorpem (rozhodnout kolik a jakých struktur?)
# ukázat zrychlení a změnu energie mezi full ff optimalizací a full ff optimalizací po prostropu
# je to paralelizovatelné
