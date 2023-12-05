PPRopt is a Python application for the fast protein structure optimization. It is rapid alternative to protein structure optimization with constrained alpha carbons. During the PPROpt method, the entire structure is not optimized at once but individual residues are optimized sequentially. The main advantage of such approach is its linear time complexity with respect to the number of atoms, where the optimization time using a regular laptop is usually about 0.14 second per atom. 

## How to install

To run PPROpt optimization you will need to have [Python 3.11](https://www.python.org/downloads/) and package manager  [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installed.

Then, clone project and install the project dependencies by running:

```bash
$ git clone https://github.com/sb-ncbr/ppropt
$ conda install scipy=1.11.3 biopython=1.81 openbabel=3.1.1 xtb=6.6.1
```

## How to run
Run the PPROpt optimization by running the following command inside github repository:

```bash
$ python3.11 ppropt.py --PDB_file <path_to_file> --data_dir <directory_to_store_data>
```

## License
MIT
