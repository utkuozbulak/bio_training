import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class SeqDataset(Dataset):
    def read_dna_file(self, file_location, class_id):
        current_file = open(file_location)
        dna_list = []

        for line in current_file:
            stripped_line = line.strip()
            # Ignore empty lines
            if stripped_line == '':
                pass
            else:
                dna_list.append((stripped_line, class_id))
        current_file.close()
        return dna_list

    def __init__(self, data_loc_and_label):
        # Initialization
        self.dna_label_list = []
        for data_loc, label in data_loc_and_label:
                self.dna_label_list.extend(self.read_dna_file(data_loc, label))
        self.data_len = len(self.dna_label_list)
        print('Dataset init with', self.data_len, 'samples')

    def __getitem__(self, index):
        # Read data
        dna_data, label = self.dna_label_list[index]
        label = int(label)
        # pdb.set_trace()
        dna_data = dna_data.replace('A', '0')
        dna_data = dna_data.replace('N', '0')
        dna_data = dna_data.replace('G', '1')
        dna_data = dna_data.replace('C', '2')
        dna_data = dna_data.replace('T', '3')
        # A [1 0 0 0]
        # G [0 1 0 0]
        # C [0 0 1 0]
        # T [0 0 0 1]

        dna_data = torch.tensor([int(digit) for digit in dna_data])
        dna_data_as_ten = torch.nn.functional.one_hot(dna_data, 4).float()
        dna_data_as_ten.unsqueeze_(dim=0)
        return dna_data_as_ten, label

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    tis_pos = SeqDataset([('../data/splice_train_pos.txt', 1),
                          ('../data/splice_train_neg.txt', 0)])

    x, y = tis_pos[0]
    print(x.shape)
    print(y)
