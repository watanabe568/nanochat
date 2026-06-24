````md
# nanochat データローダー実装調査

## 概要

本資料では、nanochat の `dataloader.py` に実装されているデータローダーについて調査し、どのような処理がどのような流れで実行されているのかをまとめる。

対象となる実装は以下である。

- `_document_batches()`
- `tokenizing_distributed_data_loader_with_state_bos_bestfit()`
- `tokenizing_distributed_data_loader_bos_bestfit()`

このデータローダーは、学習用テキストデータをトークン化し、GPTモデルへ入力できる形式に変換する役割を持つ。

---

# 1. データローダーの目的

GPTの学習では、単なる文章データではなく、

```text
入力(x)
→ 次の単語を予測
→ 正解(y)
```

という形式のデータが必要になる。

本データローダーは、

1. データセットから文章を取得
2. トークンへ変換
3. 学習用バッチ作成
4. GPUへ転送

までを担当している。

---

# 2. _document_batches()

## 役割

Parquet形式で保存されたデータセットから文章を読み込む。

## 処理の流れ

### ① Parquetファイル一覧取得

```python
parquet_paths = list_parquet_files()
```

データセットとして保存されているParquetファイル一覧を取得する。

### ② train / val の切り分け

学習用(train)

```python
parquet_paths = parquet_paths[:-1]
```

検証用(val)

```python
parquet_paths = parquet_paths[-1:]
```

最後のファイルを検証用として利用している。

### ③ DDPによる分散学習対応

```python
ddp_rank
ddp_world_size
```

を利用し、各GPUが異なるデータを担当する。

例

```text
GPU0 → RowGroup 0
GPU1 → RowGroup 1
GPU2 → RowGroup 2
GPU3 → RowGroup 3
```

### ④ RowGroupの読み込み

```python
rg = pf.read_row_group(rg_idx)
```

Parquetファイル内のRowGroup単位でデータを取得する。

### ⑤ テキスト取得

```python
batch = rg.column("text").to_pylist()
```

文章列をPythonリストへ変換する。

例

```python
[
    "Hello world",
    "How are you",
    "I am fine"
]
```

### ⑥ tokenizer用バッチへ分割

```python
yield batch[i:i+tokenizer_batch_size]
```

一定数ごとに分割して返す。

---

# 3. tokenizing_distributed_data_loader_with_state_bos_bestfit()

## 役割

この関数が実際の学習用データローダー本体である。

文章をトークン列へ変換し、

```text
入力(x)
正解(y)
```

を生成する。

---

# 4. BOSトークン付与

まず文章をトークン化する。

```python
tokenizer.encode(...)
```

さらに

```python
prepend=bos_token
```

が指定されている。

## BOSとは

BOS（Beginning Of Sequence）は文章開始を表す特殊トークンである。

例

文章

```text
Hello world
```

↓

トークン化

```text
[101, 55, 87]
```

↓

BOS追加

```text
[BOS, 101, 55, 87]
```

## 効果

全ての文章が同じ開始位置から始まるため、モデルは文書全体の文脈を理解しやすくなる。

---

# 5. Best-Fitアルゴリズム

この実装最大の特徴である。

## 行の容量

```python
row_capacity = T + 1
```

例えば

```text
T = 2048
```

なら

```text
2049トークン
```

を1行として扱う。

## 文書をバッファへ保存

```python
doc_buffer
```

にトークン列を蓄積する。

例

```text
文書A = 300
文書B = 500
文書C = 700
文書D = 120
```

## 最も大きく収まる文書を選択

```python
if doc_len <= remaining
```

残り容量に収まる文書の中から最も長い文書を選択する。

例

```text
残り容量 600

300
500
700
120
```

↓

```text
500
```

を選択する。

## 空きを最小化

この方法により、行内の未使用領域をできるだけ減らしている。

---

# 6. クロッピング処理

どの文書も残り容量に収まらない場合、

```python
best_idx < 0
```

となる。

その場合、

```python
shortest_idx = min(...)
```

によって最も短い文書を選択する。

## 残りを埋める

```python
doc[:remaining]
```

で必要な長さだけ切り取る。

例

```text
残り容量 100

文書長 150
```

↓

```text
先頭100トークンのみ利用
```

## 特徴

この方法により、

```text
利用率100%
```

となる。

つまり Padding を使用せずに学習できる。

---

# 7. 入力データと正解データ生成

行データ完成後、

```python
row_buffer
```

に格納される。

例

```text
[BOS A B C D]
```

## 入力(x)

```python
row_buffer[:, :-1]
```

結果

```text
[BOS A B C]
```

## 正解(y)

```python
row_buffer[:, 1:]
```

結果

```text
[A B C D]
```

## GPT学習

モデルは以下の予測を学習する。

```text
BOS → A
A → B
B → C
C → D
```

---

# 8. GPU転送

まずCPU側へ格納する。

```python
cpu_inputs
cpu_targets
```

↓

GPUへ転送

```python
gpu_buffer.copy_()
```

↓

学習へ利用

```python
yield inputs, targets
```

---

# 9. 学習全体の流れ

```text
Parquetファイル読込
        ↓
文章取得
        ↓
トークン化
        ↓
BOS付与
        ↓
doc_bufferへ保存
        ↓
Best-Fit配置
        ↓
長さT+1の行作成
        ↓
入力(x)生成
        ↓
正解(y)生成
        ↓
GPU転送
        ↓
GPT学習
```

---

# 10. まとめ

このデータローダーは、Parquet形式の文章データを読み込み、トークン化した後に Best-Fit アルゴリズムを用いて効率的に学習バッチを生成している。

主な特徴は以下の通りである。

- BOSトークンを各文書先頭へ付与
- DDPによる分散学習対応
- Best-Fitアルゴリズムによる高効率配置
- Paddingを使用せず利用率100%
- GPT学習用の入力(x)と正解(y)を自動生成
- GPUへ効率的に転送

これにより、大量のテキストデータを無駄なく高速に学習へ利用できる仕組みとなっている。
````
