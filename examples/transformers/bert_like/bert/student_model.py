from __init__ import *
from transformers import BertConfig, BertModel
from configparser import ConfigParser

parser = ConfigParser()
parser.read('student_config.cfg')

config = {}
config['num_hidden_layers'] = int(parser['TEACHER_CONFIG']['num_hidden_layers'])
config['num_attention_heads'] = int(parser['TEACHER_CONFIG']['num_attention_heads'])
config['hidden_size'] = int(parser['TEACHER_CONFIG']['hidden_size'])
config['intermediate_size'] = int(parser['TEACHER_CONFIG']['intermediate_size'])

STUDENT_CONFIG = BertConfig(**config)

class StudentModel(BertModel):
    def __init__(self, config):
        super().__init__(**config)
        self.bert_model = BertModel(**config)

    def forward(self, input):
        return self.bert_model(**input)
    
STUDENT_MODEL = StudentModel(STUDENT_CONFIG)