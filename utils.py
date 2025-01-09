import logging
from torch.utils import data
import torch
from torch.nn import functional as F


# error log funtion
class Logger(object):
    def __init__(self, filename, level='info', fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.INFO)
        sh = logging.FileHandler(filename, mode='w')
        sh.setFormatter(format_str)
        self.logger.addHandler(sh)


# create pytorch input format dataset
aa_dict_tokens = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
                  'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
                  'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
aa_dict_one_hot = {}
for i in aa_dict_tokens.keys():
    aa_dict_one_hot[i] = [
        1 if j == aa_dict_tokens[i] else 0 for j in range(len(aa_dict_tokens))]


class _Pep_MHC_dataset(data.Dataset):
    def __init__(self, dataset, aa_dict, vocab=None):
        self.pep = torch.tensor(
            [[aa_dict[i] for i in line] for line in dataset[0]])
        self.hla = torch.tensor(
            [[aa_dict[i] for i in line] for line in dataset[1]])

    def __getitem__(self, idx):
        return self.pep[idx], self.hla[idx]

    def __len__(self):
        return len(self.pep)


def load_data(df, aa_dict=aa_dict_one_hot, batch_size=128, max_len=25):
    df['padding'] = df['peptide'].apply(
        lambda x: x + 'X' * (max_len - len(x)))
    data_MHC = [list(i) for i in df['MHC pseudo-seq'].tolist()]
    data_pep = [list(i) for i in df['padding'].tolist()]
    dataset = _Pep_MHC_dataset([data_pep, data_MHC], aa_dict)
    data_iter = data.DataLoader(dataset, batch_size,
                                shuffle=False, num_workers=8)
    return data_iter


# predict
def predict_ms(net, test_iter, device):
    with torch.no_grad():
        softmax_metric = torch.tensor([]).to(device)
        for pep, hla in test_iter:
            pep = pep.to(device)
            hla = hla.to(device)
            y_hat = net(pep, hla)
            y_hat_softmax = F.softmax(y_hat, dim=1)
            softmax_metric = torch.cat((softmax_metric, y_hat_softmax), dim=0)
        softmax_metric = softmax_metric[:, 1].cpu().numpy()
    return softmax_metric


def predict_ba(net, test_iter, device):
    with torch.no_grad():
        result_metric = torch.tensor([]).to(device)
        for pep, hla in test_iter:
            pep = pep.to(device)
            hla = hla.to(device)
            y_hat = net(pep, hla)
            result_metric = torch.cat((result_metric, y_hat), dim=0)
    return result_metric.cpu().numpy()