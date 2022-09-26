import json
from transformers import BertJapaneseTokenizer
from label import label2id

MAX_LENGTH = 128  # 一文あたりの最大トークン数
BERT_MODEL = "cl-tohoku/bert-base-japanese-v2"  # 使用する学習済みモデル
DATASET_PATH = "./dataset/ner.json"
TAGGED_DATASET_PATH = "./dataset/ner_tagged.json"

# 1. データ読み込み

with open(DATASET_PATH) as f:
  ner_data_list = json.load(f)


# 2. 固有表現タグづけ

# 半角スペースによってエンティティの開始位置がずれるので、エンティティの開始・終了位置を調整する（トークナイズで半角スペースが削除されるため）
def adjust_entity_span(text, entities):
  white_spece_posisions = [i for i, c in enumerate(text) if c == " "]
  for entity in entities:
    start_pos = entity["span"][0]
    end_pos = entity["span"][1]
    start_diff = sum(white_spece_pos < start_pos for white_spece_pos in white_spece_posisions)
    end_diff = sum(white_spece_pos < end_pos for white_spece_pos in white_spece_posisions)
    entity["span"] = [start_pos - start_diff, end_pos - end_diff]

for ner_data in ner_data_list:
  adjust_entity_span(ner_data["text"], ner_data["entities"])

sentence_list = [ner_data["text"] for ner_data in ner_data_list]

tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)

encoded_sentence_list = [tokenizer(sentence, max_length=MAX_LENGTH, padding="max_length", truncation=True) for sentence in sentence_list]

def calc_token_length(token):
  return len(token) -2 if token.startswith("##") else len(token)

def warn_start_pos(pos, token, entity, curid):
  print("[WARN] トークンの開始位置がエンティティの開始位置を超えました。エンティティの開始=<" + str(entity["span"][0]) + "> トークンの開始=<" + str(pos) + "> curid=<" + curid + "> token=<" + token + "> entity=<" + entity["name"] + ">")

def warn_end_pos(pos, token, entity, curid):
  token_length = calc_token_length(token)
  print("[WARN] トークンの終了位置がエンティティの終了位置を超えました。エンティティの終了=<" + str(entity["span"][1]) + "> トークンの終了=<" + str(pos + token_length) + "> curid=<" + curid + "> token=<" + token + "> entity=<" + entity["name"] + ">")

def search_tokens(tokens, entity, curid):
  ret = {}

  entity_type = entity["type"]
  entity_span = entity["span"]
  entity_start_pos = entity_span[0]
  entity_end_pos = entity_span[1]

  pos = 0
  is_inside_entity = False
  for i, token in enumerate(tokens):
    if token in ["[UNK]", "[SEP]", "[PAD]"]:
      break
    elif token == "[CLS]":
      continue

    token_length = calc_token_length(token)
    if not is_inside_entity: # まだエンティティの中に入っていない場合
      if pos == entity_start_pos: # トークンの開始がエンティティの開始に一致した場合
        ret[i] = "B-" + entity_type
        if pos + token_length == entity_end_pos: # トークンの終了がエンティティの終了に一致した場合
          break
        elif pos + token_length < entity_end_pos:
          is_inside_entity = True
        else: # [WARN]トークンの終了がエンティティの終了を超えた場合
          warn_end_pos(pos, token, entity, curid)
          print(tokens)
      elif pos > entity_start_pos: # [WARN]トークンの開始がエンティティの開始を超えた場合
        warn_start_pos(pos, token, entity, curid)
        print(tokens)
        break
    else: # エンティティの中に入っている場合
      if pos + token_length == entity_end_pos: # トークンの終わりがエンティティの終わりに一致した場合
        ret[i] = "I-" + entity_type
        is_inside_entity = False
        break
      elif pos + token_length < entity_end_pos: # トークンがまだエンティティの終わりに達していない場合
        ret[i] = "I-" + entity_type
      else: # [WARN]トークンがエンティティの終わりを超えた場合
        warn_end_pos(pos, token, entity, curid)
        print(tokens)
        ret.clear()
        is_inside_entity = False
        break
    pos += token_length

  return ret

# トークンにタグ付けをする
tags_list = []
for i, encoded_sentence in enumerate(encoded_sentence_list):
  tokens = tokenizer.convert_ids_to_tokens(encoded_sentence["input_ids"])

  tags = ["O"] * MAX_LENGTH

  ner_data = ner_data_list[i]
  curid = ner_data["curid"]

  entities = ner_data["entities"]

  for entity in entities:
    found_token_pos_tags = search_tokens(tokens, entity, curid)
    for pos, tag in found_token_pos_tags.items():
      tags[pos] = tag

  tags_list.append(tags)

  # 固有表現タグをIDに変換
  encoded_tags_list = [[label2id[tag] for tag in tags] for tags in tags_list] # 学習で利用

# タグづけしたデータの保存
tagged_sentence_list = []
for encoded_sentence, encoded_tags in zip(encoded_sentence_list, encoded_tags_list):
  tagged_sentence = {}
  tagged_sentence['input_ids'] = encoded_sentence['input_ids']
  tagged_sentence['token_type_ids'] = encoded_sentence['token_type_ids']
  tagged_sentence['attention_mask'] = encoded_sentence['attention_mask']
  tagged_sentence['labels'] = encoded_tags
  tagged_sentence_list.append(tagged_sentence)

with open(TAGGED_DATASET_PATH, 'w') as f:
  json.dump(tagged_sentence_list, f)