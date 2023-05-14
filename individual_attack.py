from adversarial_attack_dataset import *


if __name__ == "__main__":
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    
# Configure the TextAttack attack recipe
    
    dataset2 = API().convert_text_to_lst("new_train_dataset_api.txt")
    dataset_transformed = [("Birds, with their mesmerizing ability to take to the skies, have always captivated the human imagination. Flight, once an exclusive domain of birds, has been a source of wonder and inspiration for millennia. While humans have made remarkable strides in aviation technology, birds continue to remain the undisputed masters of the sky. The question of why birds can fly is rooted in their unique anatomy, physiology, and evolutionary adaptations. By delving into these fascinating aspects, we can unravel the secrets behind their extraordinary ability to soar effortlessly through the air.", 1)]


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    # goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
    # constraints = [RepeatModification(),
    #                  StopwordModification(),
    #                     WordEmbeddingDistance(min_cos_sim=0.8)]
    # transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
    # search_method = textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")
    attack_args = textattack.AttackArgs(
        num_examples=1,
        log_to_csv = "log2.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints2/",
        disable_stdout=False,
    )
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    dataset = Dataset(dataset_transformed)
    
    attacker = textattack.Attacker(attack, dataset, attack_args)
    print(attacker.attack_dataset())