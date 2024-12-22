from EL_model import CapHLA_EL
from BA_model import CapHLA_BA
from utils import Logger, load_data, predict_ms, predict_ba
import os
import pickle
import pandas as pd
import argparse
import sys
import torch
from tqdm import tqdm

# script help
description = """peptide HLA-I and HLA-II binding prediction
input file must be .csv format with no header.
The first column must be peptide sequences, peptide length should range from 7-25 and is composed of normal amino acid.
The second column must contain HLA allele names within HLA library"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--input', type=str, help='the path of the .csv file contains peptide and HLA allele name')
parser.add_argument('--output', type=str, help='the path of the output file')
parser.add_argument('--gpu', type=str, default='False', help='whether use gpu to predict')
parser.add_argument('--BA', type=str, default='False', help='whether to predict binding affinity by BA data')
args = parser.parse_args()

pwd = os.getcwd()
logpath = os.path.join(pwd, 'error.log')
if not args.input:
    log = Logger(logpath)
    log.logger.critical('your input file path is empty')
    sys.exit()
if not args.output:
    log = Logger(logpath)
    log.logger.critical('your output file path is empty')
    sys.exit()

# reading file and check it quality
upscaleAA = {'A', 'R', 'N', 'D', 'V', 'Q', 'E', 'G', 'H', 'I',
             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'C'}

main_dir = os.path.dirname(__file__)
hla_df = pd.read_csv(os.path.join(main_dir, 'HLA_library.csv'))
hla_lib = dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))

input_df = pd.read_csv(args.input, header=None)
if input_df.shape[1] == 3:
    three = True
    input_df.columns = ['peptide', 'Allele Name', 'Annotation']
else:
    three = False
    input_df.columns = ['peptide', 'Allele Name']

try:
    input_df['MHC pseudo-seq'] = input_df['Allele Name'].apply(lambda x: hla_lib[x])
except KeyError:
    log = Logger(logpath)
    log.logger.critical('The HLA allele name is invalid, please check whether your HLA allele names are contained in the HLA allele library.')
    sys.exit()
for pep in input_df['peptide']:
    if len(pep) > 25 or len(pep) < 7:
        log = Logger(logpath)
        log.logger.critical('The peptide is invalid, please check whether their length ranges from 7-25.')
        sys.exit(0)
    if not set(list(pep)).issubset(upscaleAA):
        log = Logger(logpath)
        log.logger.critical('The peptide is invalid, please check whether they contain abnormal amino acid.')
        sys.exit(0)
input_iter = load_data(input_df)
print('file QC achieved!')

# load random10k peptide backgroud
allele_dict_path = os.path.join(main_dir, 'allele_dict.pickle')
allele_dict = pickle.load(open(allele_dict_path, 'rb'))
allele_list = input_df['Allele Name'].value_counts().index
allele_10k_dict = {}
for allele in allele_list:
    rand10k_path = os.path.join(main_dir, 'random_data', allele_dict[allele].replace("/", "_").replace("*", "_").replace(':', '_') +'.pickle')
    list_file = open(rand10k_path, 'rb')
    allele_10k_dict[allele] = pickle.load(list_file)

# define rank function
def rank(x):
    k = allele_10k_dict[x['Allele Name']].copy()
    k.insert(0, x['score'])
    rank = (sorted(k).index(k[0])) / 10000
    return rank

# ms model load and predict input peptides
device = torch.device('cuda' if args.gpu.lower() == 'true' else 'cpu')
params_dir = os.path.join(main_dir, 'params')
result_el = pd.DataFrame()
print('5-fold model prediction start!')
for fold in tqdm(range(5)):
    net = CapHLA_EL().to(device)
    params_path = os.path.join(params_dir, f'el_fold{fold}.params')
    net.load_state_dict(torch.load(params_path, map_location=device))
    net.eval()
    score = predict_ms(net, input_iter, device)
    result_el[f'fold{fold}'] = score
input_df['score'] = result_el.mean(axis=1)
# ba model load and predict input peptides
if three:
    output_df = input_df.loc[:, ['peptide', 'Allele Name', 'Annotation', 'score']]
else:
    output_df = input_df.loc[:, ['peptide', 'Allele Name', 'score']]
                                 
print('rank score in 10k random peptide start!')
output_df['rank'] = output_df.apply(rank, axis=1)

if args.BA.lower() == 'true':
    result_ba = pd.DataFrame()
    for fold in tqdm(range(5)):
        net = CapHLA_BA().to(device)
        params_path = os.path.join(params_dir, f'ba_fold{fold}.params')
        net.load_state_dict(torch.load(params_path, map_location=device))
        net.eval()
        score = predict_ba(net, input_iter, device)
        result_ba[f'fold{fold}'] = score
    output_df['affinity'] = result_ba.mean(axis=1)

# output
output_path = os.path.join(pwd, args.output)
output_df.to_csv(output_path, index=False)
print('Successful finished')