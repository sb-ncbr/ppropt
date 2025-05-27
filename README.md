PPRopt is a Python application for the fast protein structure optimisation. It is rapid alternative to protein structure optimisation with constrained alpha carbons. During the PPROpt method, the entire structure is not optimised at once but individual residues are optimised sequentially. The main advantage of such approach is its linear time complexity with respect to the number of atoms. 

## How to install

To run PPROpt optimisation you will need to have [Python 3.12](https://www.python.org/downloads/) and package manager  [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installed.

Then, clone project and install the project dependencies by running:

```bash
$ git clone https://github.com/sb-ncbr/ppropt
$ conda install biopython=1.81 xtb=6.6.1
```

## How to run
Run the PPROpt optimisation by running the following command inside github repository:

```bash
$ python3.11 ppropt.py --PDB_file <path_to_file> --data_dir <directory_to_store_data>
```

For example, you can use the example provided:


```bash
$ python3.11 ppropt.py --PDB_file example/L8BU87.pdb --data_dir L8BU87_optimisation
```

and then compare example/L8BU87_optimised.pdb with L8BU87_optimisation/optimised_PDB/L8BU87_optimised.pdb

## License
MIT

