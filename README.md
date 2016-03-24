# README.md

ロボケンの皆様
============


get_mlquesitons_praw.py を公開します。使い方は

```bash
python get_mlquestions_praw.y NUM
```

です。このコードを実行する前に **praw** をインストールしてください

```bash
pip intall praw
```
Q に相当する文章は，title と selftext, A に相当する文章が comment no. NUM --> 以下になっています。
多分これで html のタグは完全に取り除かれているはずです。


# 1. ptb

ptb.{train,valid,test}.txt を JSAI2016 に使えるように，便宜的に以下のルール
に従って変形しました。ptb.{train,valid,test}.txt 内の奇数行目の文(n mod 2
== 1, where n means line number for each sentence) の文章を問いの文 Q, 後続
する偶数行目の文(n mod 2 == 0)を答え A とみなして

1. 奇数行目の文末に <cntnxt> を挿入し，改行コードを削除，
2. 直後の奇数行目と連結させる

以上をする sed script を convert.sed に書いて，以下の変換を実施

```bash
for f in ptb.*; do
    ff=jsai2016${f}
    sed -f convert.sed ${f} > ${ff}
done
```

# 2. ベースラインモデル

jsai2016ptb.py:

モデルの構成:
モデルは4層のニューラルネットワークになっています。
- 第1層：単語埋込み層 650ニューロン
- 第2層：LSTM 650ニューロン
- 第3層：LSTM 650ニューロン
- 第4層：ソフトマックス層 10000ニューロン

変更したハイパーパラメータは以下のとおり:

```python
# n_epoch = 39   # number of epochs
n_epoch = 10
```
どこまで学習させるか。気が短いから少なくした。

```python
# batchsize = 20   # minibatch size
batchsize = 500
```
ミニバッチのサイズ。長期の系列を学習させる必要がるため。25 倍の長さにした。

```python
# bprop_len = 35   # length of truncated BPTT
bprop_len = 35
```
BPTT の過去へのさかのぼり。35 だと十分だろうけれど，学習の高速化のため 5 に変更

```python
# grad_clip = 5    # gradient norm threshold to clip
grad_clip = 1
```
勾配クリップの値 Graves に従って 1 に変更

使用するデータセット：
* jsai2016ptb.train.txt
* jsai2016ptb.test.txt
* jsai2016ptb.valid.txt

コマンドライン:
```python
python jsa2015ptb.py
```

# 3. 対話モデル

ファイル名: jsai2016ptb_dialogue.py

上記を，Q, A に 2 話者に分け，それぞれの発話に <pad> を埋めることを行った。
対話モデルを図示すると下図のようになる。図では上段の LSTM が Q すなわち
質問文，下段の LSTM が A すなわち対応する応答文である。

```python
# <sos>     ...     <eos>    <pad>     ...     <pad>    <sos>    ...
# LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
#  |        |        |        |        |        |        |        |
#  |        |        |        |        |        |        |        |
#  v        v        v        v        v        v        v        v
# <pad>    ....     <pad>    <sos>    ...     <eos>    <pad>    ....
# LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
```
上の LSTM の第2層の文脈情報が下の LSTM の第1層への入力となる。

# 4. S2Sモデル

Sutskever らのモデルに従えば厳密な対話モデルは，LSTM から LSTM への
矢印が一回だけです。
```python
# <sos>     ...     <eos>    <pad>     ...     <pad>    <sos>    ...
# LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
#                   |
#                   |
#                   v
# <pad>    ....     <pad>    <sos>    ...     <eos>    <pad>    ....
# LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
```

これからつくります。ゴメンナサイ。

