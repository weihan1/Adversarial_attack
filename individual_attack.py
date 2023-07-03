import textattack
import transformers
from textattack.attack import Attack
from textattack.transformations.word_swaps.word_swap_embedding import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
# Load model, tokenizer, and model_wrapper
def turn_victim_into_str(victim_path):
    '''
    Turn the text file into a string.
    '''
    victim_str = ""
    with open(victim_path, 'r') as file:
        for line in file:
            victim_str += line
    return victim_str


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("textattack").setLevel(logging.ERROR)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# model.load_state_dict(torch.load("models/best_model.pt"))
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# # Construct our four components for `Attack`
# from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
# from textattack.constraints.semantics import WordEmbeddingDistance

# goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
# constraints = [RepeatModification(),
#                    StopwordModification(),
#                    WordEmbeddingDistance(min_cos_sim=0.9)]
# transformation = WordSwapEmbedding(max_candidates=50)
# search_method = GreedyWordSwapWIR(wir_method="delete")

# # Construct the actual attack
# attack = Attack(goal_function, constraints, transformation, search_method)

# input_text = turn_victim_into_str("victim.txt")
# label = 1 #Positive
# attack_result = attack.attack(input_text, label)
# print(attack_result)


import textattack
import transformers

# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [RepeatModification(),
                   StopwordModification(),
                   WordEmbeddingDistance(min_cos_sim=0.9)]
transformation = WordSwapEmbedding(max_candidates=50)
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = Attack(goal_function, constraints, transformation, search_method)

input_text = "I really enjoyed the new movie that came out last month."
label = 1 #Positive
attack_result = attack.attack(input_text, label)
print(attack_result)