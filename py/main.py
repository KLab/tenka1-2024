import json
import sys
from dataclasses import dataclass, field
import os
import time
from math import cbrt, sqrt, atan2, cos, sin, hypot, pi
import random
from typing import NamedTuple

import requests

# ゲームサーバのアドレス / トークン
GAME_SERVER = os.getenv('GAME_SERVER', 'https://2024.gcp.tenka1.klab.jp')
TOKEN = os.getenv('TOKEN', 'YOUR_TOKEN')
session = requests.Session()


class Point(NamedTuple):
    x: str
    y: str


class StateResponse(NamedTuple):
    game_id: int
    optional_time: int
    plan_time: int
    can_submit_optional: bool
    is_set_optional: bool
    required: list[Point]
    optional: list[Point]
    checkpoint_size: str
    plan_length: int


def to_state_response(s: dict) -> StateResponse:
    return StateResponse(
        game_id=s['game_id'],
        optional_time=s['optional_time'],
        plan_time=s['plan_time'],
        can_submit_optional=s['can_submit_optional'],
        is_set_optional=s['is_set_optional'],
        required=[Point(p['x'], p['y']) for p in s['required']],
        optional=[Point(p['x'], p['y']) for p in s['optional']],
        checkpoint_size=s['checkpoint_size'],
        plan_length=s['plan_length'],
    )


def get_local_state(file_path: str):
    with open(file_path) as f:
        return to_state_response(json.load(f))


@dataclass
class GameState:
    px: float = 0.0
    py: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    score: int = 0
    target_idx: int = 0
    used_optional: list[bool] = field(default_factory=list)


class Simulator:
    def __init__(self, state: StateResponse):
        self.checkpoint_size = float(state.checkpoint_size)
        self.required = [(float(p.x), float(p.y)) for p in state.required]
        self.optional = [(float(p.x), float(p.y)) for p in state.optional]

    # t^3 + 3bt^2 + 2ct + 2d = 0 を解き昇順ソート済の実数解を返却する
    @classmethod
    def solve_cubic_equation(cls, b: float, c: float, d: float) -> list[float]:
        # カルダノの方法を用いて解く
        # 参考: https://ja.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F
        # t = y - b と置く
        # y^3 + 3hy - 2q = 0
        q = c * b - b * b * b - d
        h = c * 2 / 3 - b * b
        r = q * q + h * h * h
        if r >= 0:
            y = cbrt(q + sqrt(r)) + cbrt(q - sqrt(r))
        else:
            # z = q + sqrt(-r) i とすると、
            # zの絶対値は hypot(q, sqrt(-r))
            # zの偏角は atan2(sqrt(-r), q)
            # であり、yは以下の式で求まる
            y = cbrt(hypot(q, sqrt(-r))) * cos(atan2(sqrt(-r), q) / 3) * 2

        t1 = y - b
        # t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
        s = 3 * b + t1
        u = 2 * c + t1 * s
        # t^2 + st + u = 0 を解く
        z = s * s - 4 * u
        if z >= 0:
            return sorted([t1, (-s + sqrt(z)) / 2, (-s - sqrt(z)) / 2])
        else:
            return [t1]

    # 目的地の座標(tx, ty)を訪れる時刻t(t0 < t <= 1.0)を求める
    # 目的地に到達しないなら-1を返却する
    def calc_visit_time(self, gs: GameState, ax: float, ay: float, tx: float, ty: float, t0: float) -> float:
        def g(t: float) -> bool:
            # 時刻tで質点が目的地に到達しているかを判定する
            x = gs.px + gs.vx * t + 0.5 * ax * t * t - tx
            y = gs.py + gs.vy * t + 0.5 * ay * t * t - ty
            return hypot(x, y) <= self.checkpoint_size

        # 質点の座標を(px, py), 速度を(vx, vy), 加速度を(ax, ay) とし, 目的地の座標を(tx, ty)とする.
        # 時刻tにおける質点の座標は x(t) = px + vx t + ax t^2/2, y(t) = py + vy t + ax t^2/2 である.
        # 時刻tにおける質点の座標と目的地の座標との距離の2乗をf(t)とする. f(t) = (x(t)-tx)^2 + (y(t)-ty)^2 である.
        # f(t)を整理すると以下のようになる
        # f(t) = t^4/4 + bt^3 + ct^2 + 2dt + dx^2 + dy^2
        dx = gs.px - tx
        dy = gs.py - ty
        b = gs.vx * ax + gs.vy * ay
        c = gs.vx * gs.vx + gs.vy * gs.vy + dx * ax + dy * ay
        d = dx * gs.vx + dy * gs.vy

        # f(t)を微分すると f'(t) = t^3 + 3bt^2 + 2ct + 2d となる.
        # f'(t) = 0 の解を求め, f(t)が最小となるtの候補を求める.
        candidates = self.solve_cubic_equation(b, c, d)

        # 求めたtの候補を小さい順に試し, 最初に条件を満たしたものを返却する.
        for i, t in enumerate(candidates):
            # f(candidates[1])は極大値なので無視する
            if i != 1 and t0 < t < 1.0 and g(t):
                return t

        # t=1.0が条件を満たしているケースを考慮する
        if g(1.0):
            return 1.0
        return -1.0

    # 時刻t=0からt=1までの間に訪れる目的地の数を求める
    def calc(self, gs: GameState, ax: float, ay: float) -> tuple[int, int, list[bool]]:
        res = 0
        idx = gs.target_idx
        used = gs.used_optional[:]

        # 必須目的地への到達判定
        t1 = []
        t0 = 0.0
        while t0 < 1.0:
            tx, ty = self.required[idx]
            tt = self.calc_visit_time(gs, ax, ay, tx, ty, t0)
            if tt < 0:
                break
            t0 = tt
            res += 1
            idx += 1
            if idx == len(self.required):
                idx = 0
                t1.append(t0)

        # 任意目的地への到達判定
        for i in range(len(self.optional)):
            if used[i] and not t1:
                continue
            tx, ty = self.optional[i]
            t0 = 0.0
            j = 0
            while t0 < 1.0:
                tt = self.calc_visit_time(gs, ax, ay, tx, ty, t0)
                if tt < 0:
                    break
                t0 = tt
                # 最後の必須目的地に到達した直後にこの任意目的地に到達したケースを考慮
                while j < len(t1) and t1[j] < t0:
                    j += 1
                    used[i] = False
                if not used[i]:
                    res += 1
                    used[i] = True
            # 最後の必須目的地に到達した際に任意目的地が再度利用可能になる
            if j < len(t1):
                used[i] = False

        return res, idx, used

    # 操作θを与え単位時間だけ状態を更新する
    def update(self, gs: GameState, th: float) -> GameState:
        # 加速度を決定する
        ax = cos(th)
        ay = sin(th)
        # 訪れる目的地の数を求める
        s, idx, used = self.calc(gs, ax, ay)
        # 移動後の状態に更新する
        return GameState(
            px=gs.px + gs.vx + 0.5 * ax,
            py=gs.py + gs.vy + 0.5 * ay,
            vx=gs.vx + ax,
            vy=gs.vy + ay,
            score=gs.score + s,
            target_idx=idx,
            used_optional=used,
        )


# ゲームサーバのAPIを呼び出す
def call_api(path: str, post_data: any = None) -> dict | None:
    url = f'{GAME_SERVER}{path}'
    # 5xxエラーまたはRequestExceptionの際は100ms空けて5回までリトライする
    for _ in range(5):
        print(url, flush=True)
        try:
            if post_data:
                response = session.post(url, json=post_data)
            else:
                response = session.get(url)

            if response.status_code == 200:
                return response.json()

            if 500 <= response.status_code < 600:
                print(response.status_code, flush=True)
                time.sleep(0.1)
                continue

            print(f'Api Error status_code:{response.status_code} body:{response.text}', flush=True)
            return None

        except requests.RequestException as e:
            print(e)
            time.sleep(0.1)
    raise Exception('Api Error')


class Program:
    def __init__(self):
        self.state: StateResponse | None = None

    # /api/stateを呼び出す
    def call_state(self):
        s = call_api(f"/api/state/{TOKEN}")
        if s is None:
            return
        self.state = to_state_response(s)

    # /api/optionalを呼び出す 任意目的地を追加する
    def call_optional(self, x: float, y: float):
        call_api(f"/api/optional/{TOKEN}/{self.state.game_id}", post_data={"x": repr(x), "y": repr(y)})

    # /api/plan を呼び出す 移動予定を決定する
    def call_plan(self, plan: list[float]):
        call_api(f"/api/plan/{TOKEN}/{self.state.game_id}", post_data=list(map(repr, plan)))

    def call_eval(self, plan: list[float]) -> dict | None:
        return call_api(f"/api/eval/{TOKEN}", post_data={
            'checkpoint_size': self.state.checkpoint_size,
            'required': [{'x': p.x, 'y': p.y} for p in self.state.required],
            'optional': [{'x': p.x, 'y': p.y} for p in self.state.optional],
            'plan': list(map(str, plan)),
        })

    def think_optional(self) -> tuple[float, float]:
        while True:
            # -5.0 から 5.0 の範囲でランダムに生成する
            x = (random.random() - 0.5) * 10
            y = (random.random() - 0.5) * 10
            if hypot(x, y) >= 1:
                return x, y

    def think_plan(self) -> list[float]:
        plan = []
        gs = GameState(used_optional=[False] * len(self.state.optional))
        sim = Simulator(self.state)

        # 状態の評価を行う
        def evaluate(gs: GameState) -> (int, float):
            tx, ty = sim.required[gs.target_idx]
            return gs.score, hypot(gs.px - tx, gs.py - ty)

        for _ in range(self.state.plan_length):
            best_th = None
            best_s = -1
            best_d = 1e+300
            for _ in range(10):
                # 0~2πの範囲でthをランダムに選び
                # 現在のベストよりもスコアが増加しているか, 次の必須目的地に近づくなら採用する
                th = random.random() * 2 * pi
                s, d = evaluate(sim.update(gs, th))
                if s > best_s or (s == best_s and d < best_d):
                    best_th = th
                    best_s = s
                    best_d = d
            plan.append(best_th)
            gs = sim.update(gs, best_th)
        return plan

    def solve(self):
        game_id = None
        while True:
            # game_idの更新を待つ
            while True:
                self.call_state()
                if game_id != self.state.game_id:
                    game_id = self.state.game_id
                    break
                time.sleep(1.0)

            # 任意目的地の提出が可能な場合に提出する
            if self.state.can_submit_optional:
                x, y = self.think_optional()
                self.call_optional(x, y)

            # ゲーム開始を待つ
            while not self.state.is_set_optional:
                time.sleep(1.0)
                self.call_state()

            # planを考えて提出する
            if self.state.plan_time > 0:
                plan = self.think_plan()
                self.call_plan(plan)

    def test(self, file_path: str):
        self.state = get_local_state(file_path)
        plan = self.think_plan()
        res = self.call_eval(plan)
        assert res is not None
        with open('./eval_plan.json', 'w') as f:
            f.write(json.dumps(list(map(str, plan))))
        print('score =', res['result'][-1]['score'])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        Program().test(sys.argv[1])
    else:
        Program().solve()
