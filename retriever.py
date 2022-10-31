from utils import load_json_list
from read_dataset import FinQA_entry
from tokenizers import Tokenizer
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


# CONFIGS
TRANSFORMER_BACKBONE = "bert"
TRANSFORMER_BACKBONE_TYPE = "bert-base-uncased"
NEG_RATE = 3
BATCH_SIZE = 32
EPOCHS = 10
MAX_LENGTH = 512
LR = 2e-5
DEVICE = "cuda"


def tokenize_features_fast(features, tokenizer):
    pos = []
    neg = []
    for p, n in features:
        pos.extend(p)
        neg.extend(n)

    text_pos = [(i["text_1"], i["text_2"]) for i in pos]
    text_neg = [(i["text_1"], i["text_2"]) for i in neg]

    feats_pos = tokenizer.encode_batch(text_pos)
    feats_neg = tokenizer.encode_batch(text_neg)

    for i in range(len(pos)):
        p = pos[i]
        p["input_ids"] = feats_pos[i].ids
        p["input_mask"] = feats_pos[i].attention_mask
        p["segment_ids"] = feats_pos[i].type_ids

    for i in range(len(neg)):
        n = neg[i]
        n["input_ids"] = feats_neg[i].ids
        n["input_mask"] = feats_neg[i].attention_mask
        n["segment_ids"] = feats_neg[i].type_ids

    return pos, neg


class DataLoader:
    def __init__(self, is_training, pos, neg, batch_size=8, shuffle=True):
        self.data_pos = pos
        self.data_neg = neg
        self.batch_size = batch_size
        self.is_training = is_training
        if self.is_training:
            random.shuffle(self.data_neg)
            num_neg = int(len(self.data_pos) * NEG_RATE)
            self.data = self.data_pos + self.data_neg[:num_neg]
        else:
            self.data = self.data_pos + self.data_neg
        self.data_size = len(self.data)
        self.num_batches = (
            int(self.data_size / batch_size)
            if self.data_size % batch_size == 0
            else int(self.data_size / batch_size) + 1
        )
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # drop last batch
        if self.is_training:
            bound = self.num_batches - 1
        else:
            bound = self.num_batches
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        random.shuffle(self.data_neg)
        num_neg = int(len(self.data_pos) * NEG_RATE)
        self.data = self.data_pos + self.data_neg[:num_neg]
        random.shuffle(self.data)
        return

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)
        self.count += 1
        batch_data = {
            "input_ids": [],
            "input_mask": [],
            "segment_ids": [],
            "id": [],
            "label": [],
            "ind": [],
        }
        for each_data in self.data[start_index:end_index]:
            batch_data["input_ids"].append(each_data["input_ids"])
            batch_data["input_mask"].append(each_data["input_mask"])
            batch_data["segment_ids"].append(each_data["segment_ids"])
            batch_data["id"].append(each_data["id"])
            batch_data["label"].append(each_data["label"])
            batch_data["ind"].append(each_data["ind"])
        return batch_data


tok = Tokenizer.from_pretrained(TRANSFORMER_BACKBONE_TYPE)
tok.enable_truncation(max_length=MAX_LENGTH)
tok.enable_padding(length=MAX_LENGTH)


# train
train_data = load_json_list("train")
train_entries = [FinQA_entry.from_entry(i) for i in train_data]
train_features = [i.convert_for_train() for i in train_entries]
train_pos, train_neg = tokenize_features_fast(train_features, tok)


# valid
valid_data = load_json_list("dev")
valid_entries = [FinQA_entry.from_entry(i) for i in valid_data]
valid_features = [i.convert_for_test() for i in valid_entries]
valid_pos, train_neg = tokenize_features_fast(valid_features, tok)


# test
test_data = load_json_list("test")
test_entries = [FinQA_entry.from_entry(i) for i in test_data]
test_features = [i.convert_for_test() for i in test_entries]
test_pos, train_neg = tokenize_features_fast(test_features, tok)


train_iterator = DataLoader(
    is_training=True, pos=train_pos, neg=train_neg, batch_size=BATCH_SIZE, shuffle=True
)




if TRANSFORMER_BACKBONE == "bert":
    from transformers import BertModel
elif TRANSFORMER_BACKBONE == "roberta":
    from transformers import RobertaModel



class BertRetriever(nn.Module):

    def __init__(self, hidden_size, dropout_rate):
        super(BertRetriever, self).__init__()
        self.hidden_size = hidden_size
        if TRANSFORMER_BACKBONE == "bert":
            self.transformer_backbone = BertModel.from_pretrained(TRANSFORMER_BACKBONE_TYPE)
        elif TRANSFORMER_BACKBONE == "roberta":
            self.transformer_backbone = RobertaModel.from_pretrained(TRANSFORMER_BACKBONE_TYPE)

        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)
        self.cls_final = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, is_training, input_ids, input_mask, segment_ids, device):
        transformer_outputs = self.transformer_backbone(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        transformer_sequence_output = transformer_outputs.last_hidden_state
        transformer_pooled_output = transformer_sequence_output[:, 0, :]
        pooled_output = self.cls_prj(transformer_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)
        logits = self.cls_final(pooled_output)
        return logits


if TRANSFORMER_BACKBONE== "bert":
    from transformers import BertConfig
    model_config = BertConfig.from_pretrained(TRANSFORMER_BACKBONE_TYPE)
elif TRANSFORMER_BACKBONE == "roberta":
    from transformers import RobertaConfig
    model_config = RobertaConfig.from_pretrained(TRANSFORMER_BACKBONE_TYPE)


model = BertRetriever(hidden_size=model_config.hidden_size,dropout_rate=0.1)
model = nn.DataParallel(model)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), LR)
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
model.train()


k = 0
record_loss = 0.0
record_k = 0
for _ in range(EPOCHS):
    train_iterator.reset()
    for x in train_iterator:

        input_ids = torch.tensor(x['input_ids']).to(DEVICE)
        input_mask = torch.tensor(x['input_mask']).to(DEVICE)
        segment_ids = torch.tensor(x['segment_ids']).to(DEVICE)
        label = torch.tensor(x['label']).to(DEVICE)

        model.zero_grad()
        optimizer.zero_grad()

        this_logits = model(True, input_ids, input_mask,
                            segment_ids, device=DEVICE)

        this_loss = criterion(
            this_logits.view(-1, this_logits.shape[-1]), label.view(-1))

        this_loss = this_loss.sum()
        record_loss += this_loss.item() * 100
        record_k += 1
        k += 1

        this_loss.backward()
        optimizer.step()

        if k > 1 and k % 100 == 0:
            print("%d : loss = %.3f" %(k, record_loss / record_k))
            record_loss = 0.0
            record_k = 0