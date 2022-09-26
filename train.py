import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, BertConfig
from label import label2id, id2label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LENGTH = 128  # 一文あたりの最大トークン数
BERT_MODEL = "cl-tohoku/bert-base-japanese-v2"  # 使用する学習済みモデル
TAGGED_DATASET_PATH = "./dataset/ner_tagged.json"
MODEL_DIR = "./model"
LOG_DIR = "./logs"

# データの読み込み
with open(TAGGED_DATASET_PATH, 'r') as f:
  encoded_tagged_sentence_list = json.load(f)

# 3. データセットの作成
from sklearn.model_selection import train_test_split

class NERDataset(torch.utils.data.Dataset):
  def __init__(self, encoded_tagged_sentence_list):
    self.encoded_tagged_sentence_list = encoded_tagged_sentence_list

  def __len__(self):
    return len(self.encoded_tagged_sentence_list)

  def __getitem__(self, idx):
    # 辞書の値をTensorに変換
    item = {key: torch.tensor(val).to(device) for key, val in self.encoded_tagged_sentence_list[idx].items()}
    return item

# データを学習用、検証用に分割
train_encoded_tagged_sentence_list, eval_encoded_tagged_sentence_list = train_test_split(encoded_tagged_sentence_list)
# データセットに変換
train_data = NERDataset(train_encoded_tagged_sentence_list)
eval_data = NERDataset(eval_encoded_tagged_sentence_list)


# 4. Trainerの作成
from transformers import Trainer, TrainingArguments

import numpy as np
from datasets import load_metric

# 事前学習モデル
config = BertConfig.from_pretrained(BERT_MODEL, id2label=id2label, label2id=label2id)
model = BertForTokenClassification.from_pretrained(BERT_MODEL, config=config).to(device)
tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)

# 学習用パラメーター
training_args = TrainingArguments(
    output_dir = MODEL_DIR,
    num_train_epochs = 2,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    warmup_steps = 500,  # 学習係数が0からこのステップ数で上昇
    weight_decay = 0.01,  # 重みの減衰率
    logging_dir = LOG_DIR,
)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # ラベルのIDをラベルに変換
    predictions = [
        [id2label[p] for p in prediction] for prediction in predictions
    ]
    labels = [
        [id2label[l] for l in label] for label in labels
    ]

    results = metric.compute(predictions=predictions, references=labels)
    print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Trainerの初期化
trainer = Trainer(
    model = model, # 学習対象のモデル
    args = training_args, # 学習用パラメーター
    compute_metrics = compute_metrics, # 評価用関数
    train_dataset = train_data, # 学習用データ
    eval_dataset = eval_data, # 検証用データ
    tokenizer = tokenizer
)

# 5. 学習
trainer.train()
trainer.evaluate()

trainer.save_model(MODEL_DIR)
