# 天下一 Game Battle Contest 2024

- [公式サイト](https://tenka1.klab.jp/2024/)
- [YouTube配信](https://www.youtube.com/watch?v=jYSpojU0xXU)

## ドキュメント

- [問題概要スライド](abstract.pdf)
- [問題概要およびAPI仕様](problem.md)
- [ポータルの使い方](portal.md)
- [チュートリアル](tutorial.md)

## サンプルコード

- [Python](py)
  - Python 3.11 以上と、 `pip install requests` が必要
- [C#](cs)
  - .NET 8 で動作確認
- [Go](go)
  - go1.17.13 で動作確認
- [Rust](rust)
  - cargo 1.82.0 で動作確認
- [C++](cpp)
  - g++ (Ubuntu 11.4.0-1ubuntu1~22.04) で動作確認
  - libcurl4-openssl-dev 7.81.0-1ubuntu1.20
  - nlohmann-json3-dev 3.10.5-2

## コンテスト結果

- [最終ランキング](lottery/result.tsv)
- [各ゲームの結果](result/ranking.tsv)
- [各ゲームの入出力](https://tenka1.klab.jp/2024/result.zip) (106.8 MiB)


## ローカル実行
ゲームサーバーを動作させるdockerファイルを用意しました。

docker をインストールした環境で、以下のコマンドを実行してください。

起動
```
$ docker compose up
```

ユーザー登録
```
# ユーザID: user0001 トークン: token0001 のユーザーを作成
$ docker compose exec gamedb redis-cli HSET user_token token0001 user0001
```

以下のURLでAPI、SVGビジュアライザ、WebGLビジュアライザにアクセス可能です。
- http://localhost:8080/api/state/token0001
- http://localhost:8080/portal/index.html
- http://localhost:8080/visualizer/index.html?user_id=user0001&token=token0001

BOTプログラムはGAME_SERVERのURLとして `http://localhost:8080` を設定してください。


## ビジュアライザで使用したライブラリ等
- SVGビジュアライザ
  - [license.csv](lb/portal/license.csv) を参照してください。

- WebGLビジュアライザ本体 © 2024 KLab Inc. All rights reserved.
  - Game BGM and SE by KLab Sound Team © 2023 KLab Inc. All rights reserved.
  - [Hurisake.JsonDeserializer by hasipon](https://github.com/hasipon/Hurisake.JsonDeserializer)
  - [TextShader(MIT) © gam0022](https://qiita.com/gam0022/items/f3b7a3e9821a67a5b0f3)
  - [Rajdhani (OFL) © Indian Type Foundry](https://fonts.google.com/specimen/Rajdhani)
  - [Courier Prime (OFL) © The Courier Prime Project Authors](https://fonts.google.com/specimen/Courier+Prime)
  - [Source Sans 3 (OFL) © 2010-2020 Adobe](https://fonts.google.com/specimen/Source+Sans+3)

## ルール

- コンテスト期間
  - 2024年12月30日(月) 14:00～18:00 (日本時間)
- 参加資格
  - 学生、社会人問わず、どなたでも参加可能です。他人と協力せず、個人で取り組んでください。
- 使用可能言語
  - 言語の制限はありません。ただしHTTPSによる通信ができる必要があります。
- SNS等の利用について
  - 本コンテスト開催中にSNS等にコンテスト問題について言及して頂いて構いませんが、ソースコードを公開するなどの直接的なネタバレ行為はお控えください。
ハッシュタグ: #klabtenka1

## その他

- [ギフトカード抽選プログラム](lottery/lottery.py)
  - [実行結果](lottery/output.txt)
