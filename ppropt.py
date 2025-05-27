import argparse
import json
from rdkit import Chem
from Bio import SeqUtils
from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from os import system, path
import tqdm
from math import dist
from glob import glob
from multiprocessing import Pool
from time import time


def load_arguments():
    print("\nParsing arguments... ", end="")
    parser = argparse.ArgumentParser()
    parser.add_argument('--PDB_file',
                        type=str,
                        required=True,
                        help='PDB file with structure, which should be optimised.')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory for saving results.')
    parser.add_argument('--cpu',
                        type=int,
                        required=False,
                        default=1,
                        help='How many CPUs should be used for the calculation.')
    parser.add_argument('--run_full_xtb_optimisation',
                        action="store_true",
                        help='For testing the methodology. It also runs full xtb optimization with alpha carbons fixed.')
    parser.add_argument("--delete_auxiliary_files",
                        help="Auxiliary calculation files can be large. With this argument, "
                             "the auxiliary files will be continuously deleted during the calculation.")

    args = parser.parse_args()
    if not path.isfile(args.PDB_file):
        print(f"\nERROR! File {args.PDB_file} does not exist!\n")
        exit()
    if path.exists(args.data_dir):
        exit(f"\n\nError! Directory with name {args.data_dir} exists. "
             f"Remove existed directory or change --data_dir argument.")
    print("ok")
    return args

class AtomSelector(Select):
    """
    Support class for Biopython.
    After initialization, a set with all full ids of the atoms to be written into the substructure must be stored in self.full_ids.
    """
    def accept_atom(self, atom):
        return int(atom.full_id in self.full_ids)



def optimise_substructure(substructure, optimised_coordinates, iteration):
    if substructure.converged[-3:] == [True, True, True]:
        return "skipped"

    for atom in substructure.get_atoms():
        atom.coord = optimised_coordinates[atom.serial_number - 1]

    io = PDBIO()
    io.set_structure(substructure)

    io.save(file=f"{substructure.data_dir}/substructure_{iteration}.pdb",
            preserve_atom_numbering=True)

    # optimise substructure by xtb
    xtb_settings_template = f"""$constrain
    atoms: xxx
    force constant=1.0
    $end
    $opt
    maxcycle=10
    $end
    """
    substructure_settings = xtb_settings_template.replace("xxx", ", ".join(
        [str(i) for i in substructure.constrained_atoms_indices]))
    with open(f"{substructure.data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
        xtb_settings_file.write(substructure_settings)
    run_xtb = (f"cd {substructure.data_dir} ;"
               f"ulimit -s unlimited ;"
               f"export OMP_STACKSIZE=1G ; "
               f"export OMP_NUM_THREADS=1,1 ;"
               f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
               f"export MKL_NUM_THREADS=1 ;"
               f"xtb substructure_{iteration}.pdb --gfnff --input xtb_settings.inp --opt vtight --alpb water --verbose > xtb_output_{iteration}.txt 2> xtb_error_output_{iteration}.txt  ; rm gfnff_*")
    system(run_xtb)
    with open(f"{substructure.data_dir}/xtb_output_{iteration}.txt", "r", encoding='iso-8859-2') as xtb_output_file:
        if xtb_output_file.read().find(" GEOMETRY OPTIMIZATION CONVERGED AFTER ") > 0:
            converged = True
        else:
            converged = False

    optimised_substructure = PDBParser(QUIET=True).get_structure("substructure",
                                                                 f"{substructure.data_dir}/xtbopt.pdb")
    optimised_substructure_atoms = list(optimised_substructure.get_atoms())
    op_constrained_atoms = []
    for constrained_atom_index in substructure.constrained_atoms_indices:
        op_constrained_atoms.append(optimised_substructure_atoms[constrained_atom_index - 1])

    sup = Superimposer()
    sup.set_atoms(substructure.constrained_atoms, op_constrained_atoms)
    sup.apply(optimised_substructure.get_atoms())

    optimised_coordinates = []
    for optimised_atom_index in substructure.optimised_atoms_indices:
        optimised_atom = optimised_substructure_atoms[optimised_atom_index - 1]
        original_atom = list(substructure.get_atoms())[optimised_atom_index - 1] # todo neoptimálni
        optimised_coordinates.append((original_atom.serial_number - 1, optimised_atom.coord))
    return optimised_coordinates, converged



class PRO:
    def __init__(self,
                 data_dir: str,
                 PDB_file: str,
                 cpu: int,
                 delete_auxiliary_files: bool):
        self.data_dir = data_dir
        self.PDB_file = PDB_file
        self.cpu = cpu
        self.delete_auxiliary_files = delete_auxiliary_files



    def optimise(self):
        print(f"Loading of structure from {self.PDB_file}... ", end="")
        self._load_molecule()
        print("ok")

        print(f"Creating of substructues... ", end="")
        self._create_substructures()
        print("ok")


        self.optimised_coordinates = [atom.coord for atom in self.structure.get_atoms()]
        with Pool(self.cpu) as pool:

            # for iteration in tqdm.tqdm(range(100),
            #                            desc="Structure optimisation",
            #                            unit="iteration",
            #                            smoothing=0,
            #                            delay=0.1,
            #                            mininterval=0.4,
            #                            maxinterval=0.4):
            for iteration in range(100):
                iteration_results = pool.starmap(optimise_substructure, [(substructure, self.optimised_coordinates, iteration) for substructure in self.substructures])

                for substructure, substructure_results in zip(self.substructures, iteration_results):
                    if substructure_results == "skipped":
                        continue
                    for optimised_atom_index, optimised_coordinates in substructure_results[0]:
                        self.optimised_coordinates[optimised_atom_index] = optimised_coordinates
                    substructure.converged.append(substructure_results[1])




                for atom, coord in zip(self.structure.get_atoms(), self.optimised_coordinates):
                    atom.coord = coord
                self.io.save(f"{self.data_dir}/optimised_PDB/{path.basename(self.PDB_file[:-4])}_optimised_{iteration}.pdb")
                # s1 = PDBParser(QUIET=True).get_structure(id="structure", file="/home/dargen3/TMP/xtb_f1/xtbopt.pdb")
                # s2 = PDBParser(QUIET=True).get_structure(id="structure", file=f"{self.data_dir}/optimised_PDB/{path.basename(self.PDB_file[:-4])}_optimised_{iteration}.pdb")
        return



        print("Storage of the optimised structure... ", end="")
        logs = sorted([json.loads(open(f).read()) for f in glob(f"{self.data_dir}/sub_*/residue.log")],
                      key=lambda x: x['residue index'])
        atom_counter = 0
        for optimised_residue, log in zip(self.residues, logs):
            d = 0
            for optimised_atom in optimised_residue.get_atoms():
                d += dist(optimised_atom.coord, self.original_atoms_positions[atom_counter])**2
                atom_counter += 1
            residual_rmsd = (d / len(list(optimised_residue.get_atoms())))**(1/2)
            log["residual_rmsd"] = residual_rmsd
            if residual_rmsd > 1:
                log["category"] = "Highly optimised residue"
        with open(f"{self.data_dir}/residues.logs", "w") as residues_logs:
            residues_logs.write(json.dumps(logs, indent=2))
        self.io.save(f"{self.data_dir}/optimised_PDB/{path.basename(self.PDB_file[:-4])}_optimised.pdb")
        if self.delete_auxiliary_files:
            system(f"for au_file in {self.data_dir}/sub_* ; do rm -fr $au_file ; done &")
        print("ok\n\n")

    def _load_molecule(self):
        system(f"mkdir {self.data_dir};"
               f"mkdir {self.data_dir}/inputed_PDB;"
               f"mkdir {self.data_dir}/optimised_PDB;"
               f"cp {self.PDB_file} {self.data_dir}/inputed_PDB")
        try:
            structure = PDBParser(QUIET=True).get_structure("structure", self.PDB_file)
            io = PDBIO()
            io.set_structure(structure)
            self.io = io
            self.structure = io.structure
        except KeyError:
            exit(f"\nERROR! PDB file {self.PDB_file} does not contain any structure.\n")
        self.residues = list(self.structure.get_residues())
        self.atoms = list(self.structure.get_atoms())
        self.original_atoms_positions = [atom.coord for atom in self.structure.get_atoms()]



    def _create_substructures(self):
        kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.substructures = []
        for optimised_residue_index, optimised_residue in enumerate(self.residues, start=1):


            # creation of data dir
            substructure_data_dir = f"{self.data_dir}/sub_{optimised_residue_index}"
            system(f"mkdir {substructure_data_dir}")



            atoms_in_6A = set([atom for res_atom in optimised_residue.get_atoms() for atom in kdtree.search(center=res_atom.coord,
                                        radius=6,
                                        level="A")])



            atoms_in_12A = set([atom for res_atom in optimised_residue.get_atoms() for atom in kdtree.search(center=res_atom.coord,
                                        radius=12,
                                        level="A")])
            selector = AtomSelector()
            selector.full_ids = set([atom.full_id for atom in atoms_in_6A])
            self.io.save(file=f"{substructure_data_dir}/atoms_in_6A.pdb",
                        select=selector,
                        preserve_atom_numbering=True)
            selector.full_ids = set([atom.full_id for atom in atoms_in_12A])
            self.io.save(file=f"{substructure_data_dir}/atoms_in_12A.pdb",
                        select=selector,
                        preserve_atom_numbering=True)

            # load substructures by RDKit to determine bonds
            mol_min_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_6A.pdb",
                                                 removeHs=False,
                                                 sanitize=False)
            mol_min_radius_conformer = mol_min_radius.GetConformer()
            mol_max_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_12A.pdb",
                                                 removeHs=False,
                                                 sanitize=False)
            mol_max_radius_conformer = mol_max_radius.GetConformer()

            # dictionaries allow quick and precise matching of atoms from mol_min_radius and mol_max_radius
            mol_min_radius_coord_dict = {}
            for i, mol_min_radius_atom in enumerate(mol_min_radius.GetAtoms()):
                coord = mol_min_radius_conformer.GetAtomPosition(i)
                mol_min_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_min_radius_atom
            mol_max_radius_coord_dict = {}
            for i, mol_max_radius_atom in enumerate(mol_max_radius.GetAtoms()):
                coord = mol_max_radius_conformer.GetAtomPosition(i)
                mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_max_radius_atom

            # find atoms from mol_min_radius with broken bonds
            atoms_with_broken_bonds = []
            for mol_min_radius_atom in mol_min_radius.GetAtoms():
                coord = mol_min_radius_conformer.GetAtomPosition(mol_min_radius_atom.GetIdx())
                mol_max_radius_atom = mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)]
                if len(mol_min_radius_atom.GetNeighbors()) != len(mol_max_radius_atom.GetNeighbors()):
                    atoms_with_broken_bonds.append(mol_max_radius_atom)

            # create a substructure that will have only C-C bonds broken
            carbons_with_broken_bonds_coord = []  # hydrogens will be added only to these carbons
            substructure_coord_dict = mol_min_radius_coord_dict
            while atoms_with_broken_bonds:
                atom_with_broken_bonds = atoms_with_broken_bonds.pop(0)
                bonded_atoms = atom_with_broken_bonds.GetNeighbors()
                for bonded_atom in bonded_atoms:
                    coord = mol_max_radius_conformer.GetAtomPosition(bonded_atom.GetIdx())
                    if (coord.x, coord.y, coord.z) in substructure_coord_dict:
                        continue
                    else:
                        if atom_with_broken_bonds.GetSymbol() == "C" and bonded_atom.GetSymbol() == "C":
                            carbons_with_broken_bonds_coord.append(
                                mol_max_radius_conformer.GetAtomPosition(atom_with_broken_bonds.GetIdx()))
                            continue
                        else:
                            atoms_with_broken_bonds.append(bonded_atom)
                            substructure_coord_dict[(coord.x, coord.y, coord.z)] = bonded_atom

            # create substructure in Biopython library
            # we prefer to use kdtree because serial_id may be discontinuous in some pdbs files
            # for example, a structure with PDB code 107d and its serial numbers 218 and 445
            substructure_atoms = [kdtree.search(center=coord,
                                                radius=0.1,
                                                level="A")[0] for coord in substructure_coord_dict.keys()]
            selector.full_ids = set([atom.full_id for atom in substructure_atoms])
            self.io.save(file=f"{substructure_data_dir}/substructure.pdb", # todo je nutné ukládat?
                        select=selector,
                        preserve_atom_numbering=True)

            substructure = PDBParser(QUIET=True).get_structure(id="structure",
                                                               file=f"{substructure_data_dir}/substructure.pdb")


            constrained_atoms_indices = []
            optimised_atoms_indices = []
            constrained_atoms = []
            for atom_index, atom in enumerate(substructure.get_atoms(), start=1):
                if atom.get_parent().id == optimised_residue.id and atom.name != "CA":
                    optimised_atoms_indices.append(atom_index)
                else:
                    constrained_atoms_indices.append(atom_index)
                    constrained_atoms.append(atom)

            substructure.constrained_atoms_indices = constrained_atoms_indices
            substructure.constrained_atoms = constrained_atoms
            substructure.optimised_atoms_indices = optimised_atoms_indices
            substructure.data_dir = substructure_data_dir
            substructure.converged = []
            self.substructures.append(substructure)
        self.substructures = sorted(self.substructures, key=lambda x: len(list(x.get_atoms())), reverse=True)


def mean_absolute_deviation_of_coordinates(pdb_file1, pdb_file2):
    s1 = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file1)
    s2 = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file2)
    sup = Superimposer()
    sup.set_atoms(list(s1.get_atoms()), list(s2.get_atoms()))
    sup.apply(s2.get_atoms())
    d = []
    for a1, a2 in zip(s1.get_atoms(), s2.get_atoms()):
        d.append(a1 - a2)
    return sum(d) / len(d)



def run_full_xtb_optimisation(PRO):
    system(f"mkdir {PRO.data_dir}/full_xtb_optimisation ;")
    alpha_carbons_indices = []
    structure = PDBParser(QUIET=True).get_structure("structure", PRO.PDB_file)
    for i, atom in enumerate(structure.get_atoms(), start=1):
        if atom.name == "CA":
            alpha_carbons_indices.append(str(i))
    with open(f"{PRO.data_dir}/full_xtb_optimisation/xtb_settings.inp", "w") as xtb_settings_file:
        xtb_settings_file.write(f"$constrain\n    force constant=1.0\n    atoms: {",".join(alpha_carbons_indices)}\n$end""")

    system(f"""cd {PRO.data_dir}/full_xtb_optimisation;
               export OMP_NUM_THREADS=1,1 ;
               export MKL_NUM_THREADS=1 ;
               export OMP_MAX_ACTIVE_LEVELS=1 ;
               export OMP_STACKSIZE=200G ;
               ulimit -s unlimited ;
               xtb ../inputed_PDB/{path.basename(PRO.PDB_file)} --opt --alpb water --verbose --gfnff --input xtb_settings.inp --verbose > xtb_output.txt 2> xtb_error_output.txt""")








if __name__ == '__main__':
    args = load_arguments()
    t = time()
    Proptimus = PRO(args.data_dir, args.PDB_file, args.cpu, args.delete_auxiliary_files)
    Proptimus.optimise()
    proptimus_time = time() - t

    if args.run_full_xtb_optimisation:
        print("Running full xtb optimisation...", end="")
        t = time()
        run_full_xtb_optimisation(Proptimus)
        full_optimisation_time = time() - t
        print("ok\n")
        print(f"Proptimus time:         {proptimus_time}s")
        print(f"Full optimisation time: {full_optimisation_time}s")
        mad = mean_absolute_deviation_of_coordinates(f"{Proptimus.data_dir}/full_xtb_optimisation/xtbopt.pdb",
                                                     f"{Proptimus.data_dir}/optimised_PDB/{path.basename(Proptimus.PDB_file[:-4])}_optimised_99.pdb")
        print(f"MAD: {mad}")

        print("\n\n")

        for x in range(100):
            mad = mean_absolute_deviation_of_coordinates(f"{Proptimus.data_dir}/full_xtb_optimisation/xtbopt.pdb",
                                                         f"{Proptimus.data_dir}/optimised_PDB/{path.basename(Proptimus.PDB_file[:-4])}_optimised_{x}.pdb")
            print(x, mad)




# předělat ať to funguje s více chainama

# přepsat coordinates na array ??

# zparalelizovat dělání substruktur?

# prvních několik iterací povolit alpha uhlíky ;)

