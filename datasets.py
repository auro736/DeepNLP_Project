import json
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class PersianDataset(Dataset):
    def __init__(self, f, shuffle):
        '''
        f = path to json file
        sep_token = token separatore(?)
        input_format = perchè i vari json sono organizzati in modo diverso, capire come fare questo
        shuffle = shuffle del dataloader
        '''

        ''' content : list() = all the possible combinations (question, answer)
            labels : list() = list with 0 and 1, 1 where we have the correct answer'''
        content, labels = [], []
        x = open(f).readlines()
        if shuffle:
            random.shuffle(x)
        
        for line in x:
            '''create a python dict from this line of the jsonl'''
            instance = json.loads(line)

            ''' possible answers, length may vary from 3 to 5 only one is correct
                for network purposes we  need to have the number of choices fixed
                aribitrary chosen as 4, so we proceed only if len(choices) == 4
            '''
            choices = [a for a in instance["candidates"]]
            if len(choices) == 4:
                '''question'''
                question = instance["question"]
                '''possible answers, length may vary from 3 to 5 only one is correct'''
                choices = [a for a in instance["candidates"]]
                '''position of the correct answer'''
                correct_answer_id = int(instance["answer"])
                # '''correct answer in string'''
                # correct_answer = choices[correct_answer_id]

                '''create all the possible combinations (question, answer)'''
                for c in choices:
                    content.append("{} {}".format(question,c))
                

                '''1, 2, 3, 4 '''
                answers = [1,2,3,4]
                y = [0,0,0,0]
                y[answers.index(correct_answer_id)] = 1    #crea una lista concatenata di 0 e 1, 1  alla posizione della risposta corretta
                labels += y
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self,index):
        s1,s2 = self.content[index], self.labels[index]
        return s1,s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
            
