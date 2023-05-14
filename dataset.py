import torch
from torch.utils.data import Dataset, DataLoader
from online_dataset import OnlineDataset
from api_dataset import API
from transformers import BertModel, BertTokenizer
from torch.utils.data import random_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.manual_seed(0)
class CustomDataset(Dataset):
    def __init__(self, text_list, label_list, tokenizer, max_seq_length):
        self.text_list = text_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.label_list[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_length, add_special_tokens=True)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, label
    
def merge_both_datasets(dataset1, dataset2):
    '''
    Merge both datasets together.
    Return a tuple (merged_dataset, merged_labels).
    Label 0 represents the dataset that is human-written.
    Label 1 represents the dataset that is AI-generated.
    '''
    merged_dataset = dataset1 + dataset2
    merged_labels = [0]*len(dataset1) + [1]*len(dataset2)
    return (merged_dataset, merged_labels)
    
def get_merged_dataset(dataset1, dataset2, max_seq_length):
    '''
    Return a dataloader for the merged dataset.
    '''
    merged_dataset, merged_labels = merge_both_datasets(dataset1, dataset2)
    final_dataset = CustomDataset(merged_dataset, merged_labels, tokenizer, max_seq_length)
    
    return final_dataset


def train_test_split(merged_dataset, train_ratio, batch_size):
    '''
    Split the dataloader into train and test dataloader.
    '''
    train_size = int(train_ratio * len(merged_dataset))
    test_size = len(merged_dataset) - train_size
    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return (train_dataloader, test_dataloader)


def testing_dataloader():
    online_dataset = OnlineDataset()
    api = API()
    dataset1 = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    dataset2 = api.convert_text_to_lst("new_train_dataset_api.txt")
    merged_dataset = get_merged_dataset(dataset1, dataset2, 250)
    train_dataloader, test_dataloader = train_test_split(merged_dataset, 0.8, 4)
    


if __name__ == "__main__":
    testing_dataloader()