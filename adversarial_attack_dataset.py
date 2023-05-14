import torch
from dataset import *
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from api_dataset import API
import numpy as np
import logging
import textattack
logging.getLogger("transformers").setLevel(logging.ERROR)
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack import Attack
from textattack.datasets import Dataset

class HuggingFaceSentimentAnalysisPipelineWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, text_inputs):
        raw_outputs = self.model(text_inputs)
        outputs = []
        for output in raw_outputs:
            score = output["score"]
            if output["label"] == "POSITIVE":
                outputs.append([1 - score, score])
            else:
                outputs.append([score, 1 - score])
        return np.array(outputs)


if __name__ == "__main__":
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    
# Configure the TextAttack attack recipe
    
    dataset2 = API().convert_text_to_lst("new_train_dataset_api.txt")
    dataset_transformed = [(text, 1) for text in dataset2]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    # goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
    # constraints = [RepeatModification(),
    #                  StopwordModification(),
    #                     WordEmbeddingDistance(min_cos_sim=0.8)]
    # transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
    # search_method = textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")
    attack_args = textattack.AttackArgs(
        num_examples=20,
        log_to_csv = "log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints/",
        disable_stdout=False,
    )
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    dataset = Dataset(dataset_transformed)
    
    attacker = textattack.Attacker(attack, dataset, attack_args)
    print(attacker.attack_dataset())
    
