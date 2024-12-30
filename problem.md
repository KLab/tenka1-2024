問題概要
======

[問題概要スライド](abstract.pdf)

これは二次元平面上で質点を操作して目的地を巡回するゲームです。

ゲームは繰り返し実施され、 1 ゲームは開始から 18 秒で終了します。
ゲームは 20 秒ごとに開始し、ゲームID $1$ からゲームID $720$ まで、720 ゲームを実施する予定です。

目的地には、必須 (required) と 任意 (optional) があります。
必須の目的地はゲーム開始時に与えられます。
任意の目的地は、一つ前のゲームで上位 10 位以内の参加者が、参加者一人あたり 1 つ追加することができます。
任意の目的地の受付はゲーム開始から 8 秒間です。

必須の目的地に与えられた順序で到達することでスコアが加算されます。
任意の目的地は、最後の必須の目的地のスコア加算のときに一周したものとして、一周あたり一回のみ到達時にスコアが加算されます。

参加者は、質点の操作計画を提出し、スコアを競います。
質点の操作計画は長さ 300 の実数の列 $\theta_0 \ , \theta_1 \ , \ldots , \theta_{299}$ で表されます。
また、 $-10 < \theta_i < 10$ である必要があります。

時刻 $i$ から 時刻 $i+1$ の間、質点には加速度 $(\cos \theta_i \ , \sin \theta_i)$ がかかります。
ここで、 $\theta_i$ の単位はラジアンとします。

これによって、時刻 $i$ のときの質点の位置を $(p_x, p_y)$ 、速度を $(v_x, v_y)$ としたとき、
時刻 $i+t$ ( ただし、 $0 \le t \le 1$ ) のとき、
質点の位置は $({1 \over 2} \cos \theta_i \ t^2 + v_x \ t + p_x \ , {1 \over 2} \sin \theta_i \ t^2 + v_y \ t + p_y)$ となります。

質点から目的地までの距離が $0.01$ 以下となるような時刻 $i+t$ が存在するとき、その目的地に到達したものとします。

また、時刻 $i+1$ のときの速度は $(v_x + \cos \theta_i \ , v_y + \sin \theta_i )$ となります。

時刻 $0$ のとき、質点の位置は $(0, 0)$ であり、速度は $(0, 0)$ です。

スコア計算のプログラムは [evaluate.cpp](game/evaluate.cpp) です。
これをゲームサーバが実行した結果を正とします。

## 制約

必須の目的地は 10 個です。目的地の座標を $(x, y)$ としたとき、 $-10 < x < 10$,  $-10 < y < 10$ であることが保証されます。
必須の目的地間の距離は $2$ 以上であることが保証されます。
また、最後の必須の目的地は、原点からの距離が $0.5$ 未満であることが保証されます。

必須の目的地を生成するプログラムは [target_gen.py](game/target_gen.py) です。

任意の目的地は、座標を $(x, y)$ としたとき、 $-10 < x < 10$,  $-10 < y < 10$ であり、原点からの距離が $1$ 以上である必要があります。

## 順位決定方法

1 ゲームに対して質点の操作計画を複数回提出できます。
一人の参加者の提出のうち、最良の提出（順位が最も高くなる提出）をその参加者のそのゲームに対する提出とします。

ゲームの順位は、以下のルールに従って決定します。
1. 時刻 $300$ でのスコアが高いほうを上位とします。
2. 時刻 $300$ でのスコアが等しい場合は、時刻 $299$ でのスコアが高いほうを上位とします。
3. 時刻 $299$ でのスコアも等しい場合は、時刻 $298$ でのスコアが高いほうを上位とします。
4. 上記のルールを同様に続けていきます。
5. $1$ 以上 $300$ 以下の整数の時刻でのスコアがすべて等しい場合は、提出時刻が早いほうを上位とします。

ゲームの順位を決定後、そのゲームでの順位を $N$ として、以下のルールに従って基本順位点を計算します。
- $N \le 16$ の場合、基本順位点は $3^{N-1} \times 4^{16-N}$ とします。
- $N \ge 17$ の場合、基本順位点は $\lfloor 3^{15} \times 15 / (N-1) \rfloor$ とします。

また、獲得できる累積順位点は 30 ゲームごとに 2 倍になります。
すなわち、ゲームID $G$ で基本順位点が $R$ だったとき、獲得できる累積順位点は $R \times 2^{\lfloor {(G - 1) / 30} \rfloor}$ となります。

コンテストの順位は累積順位点に基づいて決定します。
累積順位点が等しい場合は、以下のルールに従って順位を決定します。

1. 最後に参加したゲームが新しいほうを上位とします。
2. 最後に参加したゲームが同じ場合は、そのゲームでの順位が高いほうを上位とします。

---

API仕様
======

以下の3種類のAPIを使用してゲームに参加します。
[Swagger UI](https://2024.gcp.tenka1.klab.jp/api/docs_u5nn3l340) も使用可能です。

```
GET /api/state/{token}
POST /api/optional/{token}/{game_id}
POST /api/plan/{token}/{game_id}
```

`{token}` は ポータルサイトトップに記載されています。  

optional API と plan API で送信する座標・角度の値は JSON としては文字列長 100 以下の文字列であり、
Pythonの正規表現で `[+-]?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([Ee][+-]?[0-9]+)?` にマッチする必要があります。

また、動作確認用評価APIが使用可能です。
詳しくは [チュートリアル](tutorial.md) を参照ください。

```
POST /api/eval/{token}
```

ポータル・ビジュアライザは以下のAPIを使用します。これらに関するドキュメントは提供しませんが、自作プログラムで使用することを禁止はしません。
```
GET /api/ranking/{token}
GET /api/history/{token}
GET /api/result/{token}/{game_id}/{user_id}/{submit_at}
GET /api/game/{token}/{game_id}
```

## state API (ゲーム情報取得API)

state APIは参加者ごとに1,000ミリ秒に1回に制限されます。

**endpoint**

```
GET /api/state/{token}
```

- `{token}` : ポータルサイトに記載されているトークン

**response (成功時)**

```js
{
  "game_id": 100,  // ゲームID
  "optional_time": 2345,  // 任意目的地受付の残り時間 [単位ミリ秒]
  "plan_time": 12345,  // 質点操作計画受付の残り時間 [単位ミリ秒]
  "can_submit_optional": true,  // 任意目的地が追加可能かどうか
  "is_set_optional": false,  // 任意目的地の内容が確定しているかどうか
  "required": [  // 必須の目的地
    {"x": "4.070174699013645", "y": "1.2037712739694695"},
    {"x": "4.766165309234155", "y": "-0.7269829690242524"},
    ...
  ],
  "optional": [  // 任意の目的地。is_set_optional が false の場合は空
    {"x": "1.4197498328669211", "y": "2.5827462559997603"},
    ...
  ],
  "checkpoint_size": "0.01", // チェックポイントに到達したとみなす距離のしきい値。0.01固定
  "plan_length": 300 // 質点操作計画の長さ。300固定
}
```

---
## optional API (任意目的地追加API)

optional APIは参加者ごとに1ゲームごとに1回に制限されます。

**endpoint**

```
POST /api/optional/{token}/{game_id}
```

- `{token}` : ポータルサイトに記載されているトークン
- `{game_id}` : ゲームID

**request body の例**

```js
{"x": "1.2", "y": "3.4"}
```

- $(x, y)$ を任意の目的地として追加する。
- $-10 < x < 10$,  $-10 < y < 10$ であり、原点からの距離が $1$ 以上である必要があります。

**response (成功時)**

```js
{
  "status": true
}
```

**response (失敗時: 受付時間外の場合 または 受付済の場合)**

```js
{
  "status": false
}
```

---
## plan API (質点操作計画提出API)

plan APIは参加者ごとに1,000ミリ秒に1回に制限されます。

**endpoint**
```
POST /api/plan/{token}/{game_id}
```

- `{token}` : ポータルサイトに記載されているトークン
- `{game_id}` : ゲームID

**request body の例**

```js
["1", "2.345e-06", "3.141592653589793", ...] // 文字列の array (要素数 300)
```

- $-10 < \theta_i < 10$ である必要があります。

**response (成功時)**

```js
{
  "status": true,
  "result": [
    {
      "p": {"x": "0.2701511529340699", "y": "0.4207354924039483"},
      "v": {"x": "0.5403023058681398", "y": "0.8414709848078965"},
      "score": 0
    },
    {
      "p": {"x": "1.310453458800835", "y": "1.262207649711845"},
      "v": {"x": "1.54030230586539", "y": "0.8414733298078965"},
      "score": 0
    },
    {
      "p": {"x": "2.350755764666225", "y": "2.103680979519741"},
      "v": {"x": "0.5403023058653904", "y": "0.8414733298078966"},
      "score": 0
    },
    ...
  ],
  "is_set_optional": true  // 任意目的地の内容が確定しているかどうか
}
```

**response (失敗時: 提出可能時間外の場合)**

```js
{
  "status": false,
  "result": [],
  "is_set_optional": false
}
```

---
## eval API (動作確認用評価API)

plan APIは参加者ごとに1,000ミリ秒に1回に制限されます。

**endpoint**
```
POST /api/eval/{token}
```

- `{token}` : ポータルサイトに記載されているトークン

**request body の例**

```js
{
  "checkpoint_size": "0.01",  //  0.1 以下の正の値。 evaluate.cpp の checkpoint_size
  "required": [  // 必須の目的地。要素数 1 以上 10 以下
    {"x": "4.070174699013645", "y": "1.2037712739694695"},
    {"x": "4.766165309234155", "y": "-0.7269829690242524"},
    ...
  ],
  "optional": [  // 任意の目的地。要素数 10 以下
    {"x": "1.4197498328669211", "y": "2.5827462559997603"},
    ...
  ],
  "plan": ["1", "2.345e-06", "3.141592653589793", ...] // 質点操作計画。要素数 300 以下
}
```

**response (成功時)**

```js
{
  "result": [
    {
      "p": {"x": "0.2701511529340699", "y": "0.4207354924039483"},
      "v": {"x": "0.5403023058681398", "y": "0.8414709848078965"},
      "score": 0
    },
    {
      "p": {"x": "1.310453458800835", "y": "1.262207649711845"},
      "v": {"x": "1.54030230586539", "y": "0.8414733298078965"},
      "score": 0
    },
    {
      "p": {"x": "2.350755764666225", "y": "2.103680979519741"},
      "v": {"x": "0.5403023058653904", "y": "0.8414733298078966"},
      "score": 0
    },
    ...
  ]
}
```