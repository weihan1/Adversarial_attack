from dataset import *
import tqdm
from training import *
from transformers import BertForSequenceClassification
from config import parser

def main():
    args = parser.parse_args()
    model = BertForSequenceClassification.from_pretrained(args.model_type)
    online_dataset = OnlineDataset()
    api = API()
    dataset1 = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    dataset2 = api.convert_text_to_lst("new_train_dataset_api.txt")
    merged_dataset = get_merged_dataset(dataset1, dataset2, args.max_seq_length)
    train_dataloader, test_dataloader = train_test_split(merged_dataset, args.train_test_split, args.batch_size)
    # print(len(train_dataloader.dataset))
    # print(len(test_dataloader.dataset))
    train(model, train_dataloader, test_dataloader, epochs=args.num_epochs, lr=args.learning_rate, seed=args.seed, saved_model_dir=args.saved_model_dir)


if __name__ == "__main__":
    main()