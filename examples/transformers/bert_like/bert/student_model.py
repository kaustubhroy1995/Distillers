from __init__ import *
from transformers import BertConfig, BertModel

class StudentModel(BertModel):
    def __init__(self, config):
        super().__init__(**config)
        self.bert_model = BertModel(**config)

    def forward(self, input):
        return self.bert_model(**input)
    