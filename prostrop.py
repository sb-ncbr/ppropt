import argparse
from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from os import system, path
from time import time
from scipy.spatial.distance import cdist

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


class Substructure:
    def __init__(self,
                 optimized_residuum,
                 structure,
                 data_dir: str):
        self.optimized_residuum = optimized_residuum
        self.optimized_residuum_index = self.optimized_residuum.id[1]
        self.structure = structure
        self.data_dir = data_dir
        self.constrained_atoms_indices = []
        self.structure_constrained_atoms = []
        self.substructure_residues_indices = []
        print(f"Optimization of {self.optimized_residuum_index}. residuum...", end="\r")
        atom_counter = 1
        optimized_residuum_coordinates = [atom.coord for atom in optimized_residuum.get_atoms()]
        kdtree = NeighborSearch(list(structure.get_atoms()))
        near_residues = sorted(kdtree.search(self.optimized_residuum.center_of_mass(geometric=True), 15, level="R"))
        for residuum_i in near_residues:
            residuum_i_coordinates = [atom.coord for atom in residuum_i.get_atoms()]
            if residuum_i.id[1] == self.optimized_residuum_index:
                self.substructure_residues_indices.append(residuum_i.id[1])
                atom_counter += len(residuum_i)
            elif cdist(optimized_residuum_coordinates, residuum_i_coordinates).min() < 6:
                c = 0
                residuum_i_atoms = list(residuum_i.get_atoms())
                for constrained_atom_index, structure_atom in enumerate(residuum_i.get_atoms(), start=atom_counter):
                    if min(cdist([residuum_i_coordinates[c]], optimized_residuum_coordinates)[0]) > 4:
                        self.constrained_atoms_indices.append(constrained_atom_index)
                        self.structure_constrained_atoms.append(residuum_i_atoms[c])
                    c += 1
                self.substructure_residues_indices.append(residuum_i.id[1])
                atom_counter += len(residuum_i)
        self.substructure_data_dir = f"{self.data_dir}/sub_{self.optimized_residuum_index}"
        self.pdb_file = f"{self.substructure_data_dir}/sub_{self.optimized_residuum_index}.pdb"
        io = PDBIO()
        io.set_structure(self.structure)
        selector = SelectIndexedResidues()
        selector.indices = self.substructure_residues_indices
        system(f"mkdir {self.substructure_data_dir}")
        io.save(self.pdb_file, selector)


    def optimize(self):
        substructure_settings = xtb_settings_template.replace("xxx", ", ".join([str(x) for x in self.constrained_atoms_indices]))
        with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
            xtb_settings_file.write(substructure_settings)
        system(f"cd {self.substructure_data_dir} ;"
               f"xtb sub_{self.optimized_residuum_index}.pdb "
               f"--gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1")

    def update_PDB(self):
        pdb_parser = PDBParser(QUIET=True)
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
        sup = Superimposer()
        sup.set_atoms(self.structure_constrained_atoms, constrained_atoms)
        sup.apply(non_constrained_atoms + constrained_atoms)
        substructre_atoms = list(substructure.get_atoms())
        c = 0
        for res_index in self.substructure_residues_indices:
            for atom in structure[int(res_index)]:
                if c+1 not in self.constrained_atoms_indices:
                    atom.set_coord(substructre_atoms[c].coord)
                c += 1
        io = PDBIO()
        io.set_structure(substructure)
        io.save(f"{self.substructure_data_dir}/imposed_{self.optimized_residuum_index}.pdb")


def load_molecule(pdb_file):
    print(f"Loading of molecule from {pdb_file}... ", end="")
    structure = PDBParser(QUIET=True).get_structure("structure", pdb_file)[0]["A"]
    residues = list(structure.get_residues())
    print("ok\n")
    return structure, residues


if __name__ == '__main__':
    args = load_arguments()
    original_pdb, optimized_pdb = prepare_directory(args)
    structure, residues = load_molecule(args.pdb_file)
    s = time()
    for residuum in structure.get_residues():
        substructure = Substructure(residuum,
                                    structure,
                                    args.data_dir)
        substructure.optimize()
        substructure.update_PDB()
    io = PDBIO()
    io.set_structure(structure)
    io.save(optimized_pdb)
    print(f"Structure sucessfully optimized after {round(time() - s)} seconds.\n")







# rozhodnout
# jaký použít cutoff
# přidávat vodíky?



# implementovat
# parallelizovatelnost
# předělat ať je to schopno pracovat s více chainy
# zkontrolovat optimalizaci kodu
# zkontrolovat čitelnost kodu
# pdb parser a pdb vstup/výstup

# xtb by mělo běžet v adresáři
# dopsat aby pdb file mohl být kdekoliv
# nahradit kdtree z bio


# do článku
# ukázat zrychlení a změnu energie mezi full ff optimalizací a prostorpem (rozhodnout kolik a jakých struktur?)
# ukázat zrychlení a změnu energie mezi full ff optimalizací a full ff optimalizací po prostropu
# je to paralelizovatelné
# ukázat že to funguje na 50t atomech
