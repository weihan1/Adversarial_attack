import json

class OnlineDataset:
    
    def extract_topics(self,json_file_path):
        '''
        Extract the different topics given a json file path.
        Works on the "essay" dataset.
        '''
        with open(json_file_path) as f:
            data = json.load(f)
        num_topics = len(data)
        list_of_topics = []
        for i in range(num_topics):
            list_of_topics.append(data[i]['topic'])
        # print("dataset length", len(list_of_topics))
        # subtopic_titles = [subtopic['title'] for subtopic in data['subtopics']]
        
        return list_of_topics


    def extract_argument(self,json_file_path):
        '''
        Extract the argument given a json file path.
        Works on the "essay" dataset.
        '''
        with open(json_file_path) as f:
            data = json.load(f)
        num_topics = len(data)
        
        list_of_arguments = []
        for i in range(num_topics):
            num_subtopics = len(data[i]['subtopics'])
            # print(f"for topic {i}, there are {num_subtopics} subtopics")
            for j in range(num_subtopics):
                num_arguments = len(data[i]['subtopics'][j]['arguments'])
                # print(f"for subtopic {j}, there are {num_arguments} arguments")
                for k in range(num_arguments):
                    list_of_arguments.append(data[i]['subtopics'][j]['arguments'][k]['premise'])
        
        return list_of_arguments

    def dataset_characteristics(self,lst_of_arguments):
        '''
        Calculate the average length of an argument given a list of arguments.
        '''
        total_length = 0
        for i in range(len(lst_of_arguments)):
            total_length += len(lst_of_arguments[i])
        average_length = total_length / len(lst_of_arguments)
        print(f"Average length of an argument: {average_length}")
        print(f"Total number of arguments: {len(lst_of_arguments)}")

    def generate_labels(self,lst_of_arguments):
        '''
        Generate labels for the input_lst. The labels are just [1] * len(input_lst).
        '''
        return [1]*len(lst_of_arguments)

if __name__ == '__main__':
    online_dataset = OnlineDataset()
    topic_title = online_dataset.extract_topics('arg_search_framework/data/essay/train.json')
    lst_of_arguments = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    online_dataset.dataset_characteristics(lst_of_arguments)
    
        