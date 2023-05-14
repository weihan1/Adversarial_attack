import torch
from dataset import *
from random import randint
from transformers import BertForSequenceClassification, BertTokenizer
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def test(best_model_path):
    '''
    Test the best model on some random examples.
    '''
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    online_dataset = OnlineDataset()
    api = API()
    dataset1 = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    dataset2 = api.convert_text_to_lst("new_train_dataset_api.txt")
    merged_dataset = get_merged_dataset(dataset1, dataset2, 128)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, test_dataloader = train_test_split(merged_dataset, 0.8, 1)

    for batch in test_dataloader:
        
        # print(f"random sample: {tokenizer.decode(batch[0][0])}")
        print(f"random sample label: {batch[2]}")
        
        # print(f"prediction: {torch.argmax(model(batch[0], batch[1], batch[2]).logits, dim=1)}")
        random_sample_label = batch[2]
        random_sample_prediction = torch.argmax(model(batch[0], batch[1], batch[2]).logits, dim=1)
        # if random_sample_label == random_sample_prediction:
        #     print("correct prediction")    

if __name__ =="__main__":
    test("models/best_model.pt")