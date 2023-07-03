import os
FILE_PATH = 'arg_search_framework/data/essay/train.json'
import argparse

parser = argparse.ArgumentParser(description='BERT model training')

# Required parameters
parser.add_argument('--model_type', type=str, default='bert-base-uncased',
                help='The type of BERT model to use')
parser.add_argument('--saved_model_dir', type=str, default='models/best_model2.pt',
                help='The output directory where the model predictions and checkpoints will be written')

# Other parameters
parser.add_argument('--train_test_split', type=float, default=0.8, 
                help='The ratio of train dataset to test dataset')
parser.add_argument('--max_seq_length', type=int, default=128,
                help='The maximum total input sequence length after tokenization')
parser.add_argument('--batch_size', type=int, default=32,
                help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                help='The initial learning rate for AdamW optimizer')
parser.add_argument('--num_epochs', type=int, default=3,
                help='Total number of training epochs to perform')
parser.add_argument('--warmup_steps', type=int, default=0,
                help='Number of steps for linear warmup')
parser.add_argument('--seed', type=int, default=42,
                help='Random seed for initialization')
