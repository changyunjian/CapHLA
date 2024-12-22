# CapHLA
A comprehensive tools to predict peptide presentation and binding to HLA class I and class II

## Installation

### Get the CapHLA Source
```
git clone https://github.com/changyunjian/CapHLA.git
```
### Environment and Dependencies

* [python](https://www.python.org) (3.8.13)
* [pytorch](https://pytorch.org) (1.12.0)
* [numpy](https://numpy.org) (1.24.2)
* [pandas](https://pandas.pydata.org) (2.0.0)
* [tqdm](https://tqdm.github.io/) (4.64.0)

#### Optional Dependencies
* [cuda](https://developer.nvidia.com/cuda-downloads) (11.7)  Required for GPU usage
* [scikit-learn](https://scikit-learn.org/) (1.1.1) Required for calculate performance metrics

### Build
Users can configure the environment themselves, or use the Conda YAML file provided by us to configure the environment.
```
conda env create -f caphla.yaml
conda activate caphla
```

### Test
To test your installation, make sure you are in the CapHLA directory and run the command
```
python CapHLA.py --input test.csv --output test_out.csv
```

## Usage
`CapHLA.py` is used for making predictions using CapHLA-EL and CapHLA-BA.

### Command
```
python CapHLA.py --input test.csv --output test_out.csv --gpu False --BA True
```
* `--input` type=str, the path of the .csv file contains peptide and HLA allele
input file must be .csv format contained two columns or three columns with no header.
The first column must be peptide sequences, peptide length should range from 7-25 and is composed of normal amino acid.
The second column must contain HLA allele within HLA library.
The third column could be annotation or nothing.
HLAI and HLAII could mix in a file.

* `--output` type=str, the path of the output file
The score column represent the presentation probability of peptide. The rank column represent a queried peptide has achieved a prediction score surpassing that scores observed in random natural peptide.
* `--gpu` type=bool, default=False, whether use gpu to predict
* `--BA` type=bool, default=False, whether to predict binding affinity
