from datasets import Dataset, DatasetDict
import pickle
from tqdm import tqdm
from constants import DATASET_PATH


class DatasetNER(object):
    def __init__(self):
        self.dataset_type = ['train', 'test', 'dev']
        
    def open_data(self, type_d:str):
        with open(DATASET_PATH + "_" + type_d, "rb") as fp:   
            self.data =  pickle.load(fp)
        if type_d == 'train':
            self.label_all = sorted(list(set([token[1] for line in self.data[0] for token in line])))
        self.label_map = {label: i for i, label in enumerate(self.label_all)}

    def make_dataset(self, type_d:str) -> list:

        data_format = {"tokens": [], "tags": []}

        for lines in self.data[0]:
            tokens = [word[0] for word in lines]
            tags = [self.label_map[word[1]] for word in lines]
            data_format["tokens"].append(tokens)
            data_format["tags"].append(tags)
        
        print(len(tokens), len(tags))
        
        return [type_d, Dataset.from_dict(data_format)]
    
    def make_dataset_dict(self) -> DatasetDict:
        ds_dict = {}
        for type_d in tqdm(self.dataset_type):
            self.open_data(type_d)
            ds = self.make_dataset(type_d)
            ds_dict[ds[0]] = ds[1]
        
        return DatasetDict(ds_dict)
