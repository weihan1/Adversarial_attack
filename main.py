import pyinputplus as pyip
from individual_attack_dataset import perform_individual_attack_on_dataset
if __name__ == "__main__":
    result = pyip.inputStr('Write your copied essay from ChatGPT> ')
    perform_individual_attack_on_dataset(result, 1)

