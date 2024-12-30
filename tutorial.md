チュートリアル
====

このドキュメントでは、コンテスト参加者がサンプルのBotを動かし、改善する際の参考手順を説明します。

まずは [問題概要スライド](abstract.pdf) や [問題概要](problem.md) に目を通し、ゲームの概要を把握してから始めてください。

## 1. TOKENを取得する

[ポータルサイト](https://2024.gcp.tenka1.klab.jp/portal/index.html) にユーザー登録してログインし、画面中央に表示されているトークンをメモしておきます。

> 詳しくは[ポータルサイトの使い方](portal.md)を参照してください

## 2. サンプルbotをダウンロードする
以下のいずれかから、自分が使用したいプログラミング言語のサンプルをダウンロードしてください。
手元のPCにビルド環境・実行環境が整っている必要があります。

- [C#](cs)
- [C++](cpp)
- [Go](go)
- [Python](py)
- [Rust](rust)

このチュートリアルでは Python を使用して説明します。

## 3. サンプルbotを動かす

ダウンロードしたサンプルプログラムを編集して、`YOUR_TOKEN` という文字列をメモしておいた自分のトークンで置き換えます。

編集したプログラムを実行します。サンプルbotはゲームに自動的に参加して回答を送信します。

Pythonの例:
```
py$ python -m pip install -r requirements.txt # 依存パッケージのインストール
py$ python main.py # サンプル実行
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
https://2024.gcp.tenka1.klab.jp/api/plan/xxxxxxxxxxxxxxxxxxxxxxxxx/397
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
https://2024.gcp.tenka1.klab.jp/api/state/xxxxxxxxxxxxxxxxxxxxxxxxx
...
```

ゲームに参加することができました。コンテスト中は新しいゲームが順次開始されるので、botプログラムを実行し続けておく必要があります。

参加したゲームのスコアやリプレイ表示をポータルサイトから確認することができます。<br/>
[ポータルサイトの使い方](portal.md#リプレイ表示) も合わせてご覧ください。

## 4. プログラムを改良する
あらかじめ、実行用と改良用でプログラム（ソースまたは実行ファイル）分けておきます。<br/>
実行用のプログラムを使ってゲームに参加しつつ、その間に改良用のプログラムを編集して改良を行います。<br/>
改良が終ったら、実行用プログラムを止めて改良したものに置き換え、再度実行します。<br/>
この手順を繰り返すことで、ゲームへの参加とプログラムの改良を並行することができます。

改良時のプログラムの実行方法と、結果をビジュアライズする方法を説明します。

### 1. 入力ファイルを用意
[state API (ゲーム情報取得API)](problem.md#state-api-ゲーム情報取得api) のレスポンス内容をテキストファイルに保存します。

ポータルの [リプレイ表示](portal.md#リプレイ表示) の入力テキストフィールドに表示されているものをコピーして保存するか、サンプルとして [example.json](example.json) を用意していますので、これを使用できます。

### 2. プログラムを実行
サンプルbotの引数として入力ファイルのパスを与えると、与えられたゲームに対して回答(plan)を考えて、 [eval API (動作確認用評価API)](problem.md#eval-api-動作確認用評価api) を使用して動作確認を行うようになっています。

```
py$ python main.py ../example.json
https://2024.gcp.tenka1.klab.jp/api/eval/xxxxxxxxxxxxxxxxxxxxxxxxx
score = 6
```

また、実行時にはplanの配列を`eval_plan.json`というファイルに出力しています。

### 3. 動作結果のビジュアライズ

ポータルの [リプレイ表示](portal.md#リプレイ表示) の入力欄に入力ファイルの内容、出力欄に `eval_plan.json` の内容を貼り付けし、**EVAL**ボタンをクリックすることで視覚的に動作確認を行うことが出来ます。

![チュートリアル](/img/tutorial.png)
