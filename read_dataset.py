from dataclasses import dataclass
from typing import List
from utils import table_row_to_text

@dataclass
class FinQA_entry:
    id:str
    question:str
    all_positive:List[str]
    pre_text:str
    post_text:str
    table:str

    def from_entry(entry):
        return FinQA_entry(entry['id'],
                             entry['qa']['question'],
                             entry['qa'].get('gold_inds',[]),
                             entry['pre_text'],
                             entry['post_text'],
                             entry['table'])
    def convert_for_train(self):
        question = self.question
        all_text = self.pre_text+self.post_text

        pos_features = []
        neg_features = []

        for gold_ind in self.all_positive:
            this_gold_sent = self.all_positive[gold_ind]
            features = {}
            features["text_1"] = question
            features["text_2"] = this_gold_sent            
            features['id'] = self.id
            features['ind'] = gold_ind
            features['label'] = 1
            pos_features.append(features)

        num_pos_pair = len(self.all_positive)

        pos_text_ids = []
        pos_table_ids = []

        for gold_ind in self.all_positive:
            if "text" in gold_ind:
                pos_text_ids.append(int(gold_ind.replace("text_", "")))
            elif "table" in gold_ind:
                pos_table_ids.append(int(gold_ind.replace("table_", "")))

        all_text_ids = range(len(self.pre_text) + len(self.post_text))
        all_table_ids = range(1, len(self.table))
        
        all_negs_size = len(all_text) + len(self.table) - len(self.all_positive)
        if all_negs_size < 0:
            all_negs_size = 0
                    
        # test: all negs
        # text
        for i in range(len(all_text)):
            if i not in pos_text_ids:
                this_text = all_text[i]

                features = {}
                features["text_1"] = question
                features["text_2"] = this_text
                features["id"] = self.id
                features["ind"] = "text_" + str(i)
                features['label'] = 0
                neg_features.append(features)
            # table      
        for this_table_id in range(len(self.table)):
            if this_table_id not in pos_table_ids:
                this_table_row = self.table[this_table_id]
                this_table_line = table_row_to_text(self.table[0], self.table[this_table_id])

                features = {}
                features["text_1"] = question
                features["text_2"] = this_table_line
                features["id"] = self.id
                features["ind"] = "table_" + str(this_table_id)
                features['label'] = 0
                neg_features.append(features)

        return pos_features, neg_features
    
    def convert_for_test(self):
        question = self.question
        all_text = self.pre_text+self.post_text

        pos_features = []
        neg_features = []
                    
        for i in range(len(all_text)):
            this_text = all_text[i]
            features = {}
            features["text_1"] = question
            features["text_2"] = this_text
            features["id"] = self.id
            features["ind"] = "text_" + str(i)
            features['label'] = -1
            neg_features.append(features)
        for this_table_id in range(len(self.table)):
            this_table_row = self.table[this_table_id]
            this_table_line = table_row_to_text(self.table[0], self.table[this_table_id])

            features = {}
            features["text_1"] = question
            features["text_2"] = this_table_line
            features["id"] = self.id
            features["ind"] = "table_" + str(this_table_id)
            features['label'] = -1
            neg_features.append(features)

        return pos_features, neg_features