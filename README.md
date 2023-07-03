# Adversarial_attack
Steps:
Create your own dataset of essays with various topics.
Use api_dataset.py to generate AI-generated essays based on the same topics of your dataset.
Then, you can use main.py to train a BERT model to learn to classify whether a piece of text is written by AI or by a human.
Finally, run individual_attack.py to perform an adversarial attack to break the model's predictions.

# Note:
The adversarial attack module is using TextAttack (https://textattack.readthedocs.io/en/latest/) and the AI-generated process is using the OPENAI API (https://openai.com).