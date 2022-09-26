from transformers import pipeline
from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR)

ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

ner_pipeline("株式会社はJurabi、東京都台東区に本社を置くIT企業である。")