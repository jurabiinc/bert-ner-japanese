# bert-ner-japanese
BERTによる日本語固有表現抽出のファインチューニング用プログラム

## 使用方法

### パッケージのインストール
requirements.txtに書かれているパッケージをインストールします。

### 学習用データのダウンロード
[stockmarkteam/ner-wikipedia-dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)から学習用データセット（ner.json）をダウンロードし、リポジトリ直下のdatasetディレクトリに保存します。

### BERT入力データの作成
create_tagged_token.pyを実行して、ダウンロードした学習用データを、BERTの入力データの形式に変換します。
作成されたデータは、dataset/ner_tagged.jsonに出力されます。

### 学習の実行
train.pyを実行して、ファインチューニングを行います。
学習済みのモデルは、modelディレクトリに出力されます。

### テスト
test.pyを実行して、固有表現が抽出できることを確認して下さい。
