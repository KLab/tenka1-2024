import os
import subprocess
from pathlib import Path
from time import sleep
from typing import NamedTuple

import redis
import json

from main import get_timestamp, get_cur_game_id, redis_pool
from target_gen import target_gen

PLAN_PERIOD = 10000
GAME_PERIOD = 18000
MATCH_PERIOD = 20000
NUM_GAMES_PER_SET = 30

CHECKPOINT_SIZE = '0.01'
PLAN_LENGTH = 300
NUM_OPTIONAL = 10


class Game(NamedTuple):
    game_id: int
    optional_end_time: int
    plan_end_time: int
    required: list[dict[str, str]]


def close_optional(game_id: int) -> list[tuple[str, str]]:
    r = redis.Redis(connection_pool=redis_pool)
    r.sadd('closed_optional', game_id)
    res = []
    for k, v in r.hgetall(f'optional_point_{game_id}').items():
        x, y = json.loads(v)
        res.append((x, y))
    return res


def wait(timestamp: int):
    delay = timestamp - get_timestamp()
    if delay <= 0:
        return
    if delay > 5:
        sleep((delay - 5) * 1e-3)
    while True:
        delay = timestamp - get_timestamp()
        if delay <= 0:
            return
        sleep(1e-3)


def load_game(last_game_id: int) -> Game | None:
    r = redis.Redis(connection_pool=redis_pool)
    game_id = get_cur_game_id(r)
    if game_id is None or game_id == last_game_id:
        return None
    data = r.hget('master_data', str(game_id))
    assert data is not None
    v = json.loads(data)
    print(f'load_game {game_id}', flush=True)
    return Game(
        game_id=game_id,
        optional_end_time=v['optional_end_time'],
        plan_end_time=v['plan_end_time'],
        required=v['required'],
    )


def start_game(last_game_id: int, contest_start_at: int, num_games: int) -> Game | None:
    game_id = last_game_id + 1
    if game_id >= num_games:
        return None

    required = target_gen()

    start_at = max(
        contest_start_at + game_id * MATCH_PERIOD + MATCH_PERIOD - GAME_PERIOD,
        get_timestamp() + 50
    )
    game = Game(
        game_id=game_id,
        optional_end_time=start_at + GAME_PERIOD - PLAN_PERIOD,
        plan_end_time=start_at + GAME_PERIOD,
        required=required,
    )
    data = json.dumps({
        'required': required,
        'checkpoint_size': CHECKPOINT_SIZE,
        'plan_length': PLAN_LENGTH,
        'optional_end_time': game.optional_end_time,
        'plan_end_time': game.plan_end_time,
    }, separators=(',', ':'))

    r = redis.Redis(connection_pool=redis_pool)
    r.hset('master_data', str(game_id), data)
    wait(start_at)
    r.set('cur_game', game_id)
    print(f'start_game {game_id}', flush=True)
    return game


class CalcScore:
    def __init__(self, game: Game, optional: list[tuple[str, str]]):
        self.game_id = game.game_id
        self.plan_end_time = game.plan_end_time
        self.optional = optional
        eval_input = f'{CHECKPOINT_SIZE} {len(game.required)} {len(self.optional)} {PLAN_LENGTH}\n'
        for pos in game.required:
            eval_input += f'{pos["x"]} {pos["y"]}\n'
        for x, y in self.optional:
            eval_input += f'{x} {y}\n'
        self.eval_input = eval_input
        self.user_score: dict[str, list[int]] = {}

    def run(self):
        r = redis.Redis(connection_pool=redis_pool)
        u_plan = r.lrange(f'u_plan_{self.game_id}', 0, -1)
        u_plan_cnt = 0
        for idx, plan_key in enumerate(u_plan):
            user_id, timestamp = plan_key.decode().split('/')
            # noinspection PyTypeChecker
            plan: bytes = r.hget(f'plan_{self.game_id}', plan_key)
            try:
                results, scores = self.evaluate(plan.decode())
            except:
                print(f'evaluate error {self.game_id} {user_id} {timestamp}')
                continue
            r.hset(f'results_{self.game_id}', plan_key, '\n'.join(results))
            r.hset(f'scores_{self.game_id}', plan_key, ' '.join(scores))
            r.rpush(f'timestamp_{user_id}_{self.game_id}', f'{timestamp} {scores[-1]}')
            self.save_score(plan_key, scores, idx)
            u_plan_cnt += 1

        print(f'calc score u_plan finished {u_plan_cnt}', flush=True)

        idx = 0
        closed = False
        idx_dict = {}
        while True:
            # print(f'calc score t_plan idx={idx} len={len(idx_dict)}', flush=True)
            t_plan = r.lrange(f't_plan_{self.game_id}', idx, -1)
            for i, plan_key in enumerate(t_plan, start=idx + len(u_plan)):
                idx_dict[plan_key] = i
            idx += len(t_plan)
            for plan_key in set(idx_dict.keys()):
                # noinspection PyTypeChecker
                scores2: bytes = r.hget(f'scores_{self.game_id}', plan_key)
                if scores2 is not None:
                    self.save_score(plan_key, scores2.decode().split(' '), idx_dict[plan_key])
                    del idx_dict[plan_key]
            if closed and not idx_dict:
                break
            if closed and get_timestamp() >= self.plan_end_time + 1500:
                print('idx_dict =', idx_dict, flush=True)
                break
            if not closed and get_timestamp() >= self.plan_end_time:
                closed = True
                r.sadd('closed_plan', self.game_id)
                print('calc score closed_plan', flush=True)
            sleep(10e-3)

        print('calc score t_plan finished', flush=True)

        if not self.user_score:
            print('calc score user_score empty', flush=True)
            return []

        ranking = [(x, y[0], -y[-2]) for x, y in sorted(self.user_score.items(), key=lambda x: x[1], reverse=True)]
        next_game_id = self.game_id + 1
        for i in range(min(len(ranking), NUM_OPTIONAL)):
            user_id = ranking[i][0]
            r.sadd(f'optional_user_{next_game_id}', user_id)
        for rank, x in enumerate(ranking, start=1):
            user_id, score, timestamp = x
            r.hset(f'best_result_{self.game_id}', user_id, str(timestamp))
            r.rpush(f'rank_score_{user_id}', f'{self.game_id} {rank} {score}')
        r.set(f'ranking_{self.game_id}', '\n'.join(f'{user_id} {score}' for user_id, score, _ in ranking))
        start_time = self.plan_end_time - GAME_PERIOD
        r.rpush(f'history', f'{self.game_id} {start_time}')
        print('calc score ranking finished', flush=True)
        return [x[0] for x in ranking]

    def evaluate(self, plan: str) -> tuple[list[str], list[str]]:
        eval_output = subprocess.run(
            './evaluate', input=f'{self.eval_input}{plan}\n',
            capture_output=True, text=True, check=True, timeout=0.1).stdout
        results: list[str] = []
        scores: list[str] = []
        for line in eval_output.split('\n')[:PLAN_LENGTH]:
            a = line.split(' ')
            results.append(line)
            scores.append(a[4])
        return results, scores

    def save_score(self, plan_key: bytes, scores: list[str], idx: int):
        user_id, timestamp = plan_key.decode().split('/')
        a = [int(x) for x in scores[::-1]]
        a.append(-int(timestamp))
        a.append(-idx)
        if user_id in self.user_score:
            self.user_score[user_id] = max(self.user_score[user_id], a)
        else:
            self.user_score[user_id] = a


def run_game(game: Game) -> list[str]:
    print(f'run_game {game.game_id}', flush=True)
    wait(game.optional_end_time)
    optional = close_optional(game.game_id)
    print(f'optional closed {game.game_id}', flush=True)
    calc_score = CalcScore(game, optional)
    return calc_score.run()


def add_users() -> None:
    users = Path('users.tsv')
    if users.exists():
        items = []
        with users.open() as f:
            for line in f:
                user, token = line.rstrip('\n').split('\t')
                items.extend((token, user))
        r = redis.Redis(connection_pool=redis_pool)
        r.hset('user_token', items=items)


def get_stop_matching() -> bool:
    r = redis.Redis(connection_pool=redis_pool)
    return r.get('stop_matching') is not None


class RankingElement:
    def __init__(self):
        self.point = 0
        self.last_game_id = 0
        self.last_game_rank = 0


class Ranking:
    def __init__(self):
        self.data: dict[str, RankingElement] = {}
        self.last_game_id = 0

    def load(self):
        r = redis.Redis(connection_pool=redis_pool)
        for x in r.lrange('history', 0, -1):
            game_id = int(x.decode().split(' ')[0])
            result = [y.split(' ')[0] for y in r.get(f'ranking_{game_id}').decode().split('\n')]
            self.update(game_id, result)

    def update(self, game_id: int, result: list[str]):
        self.last_game_id = game_id
        for i, user_id in enumerate(result):
            if i <= 15:
                r = (3 ** i) << (30 - 2 * i)
            else:
                r = (3 ** 15) * 15 // i
            s = r << ((game_id - 1) // NUM_GAMES_PER_SET)
            if user_id not in self.data:
                self.data[user_id] = RankingElement()
            self.data[user_id].point += s
            self.data[user_id].last_game_id = game_id
            self.data[user_id].last_game_rank = i

    def save(self):
        r = redis.Redis(connection_pool=redis_pool)
        a = sorted(self.data.items(), key=lambda x: (x[1].point, x[1].last_game_id, -x[1].last_game_rank), reverse=True)
        s = '\n'.join(f'{user_id} {x.point}' for user_id, x in a)
        r.set('ranking', f'{self.last_game_id}\n{s}')
        print(f'ranking saved {self.last_game_id}', flush=True)


def main():
    print(f'batch started', flush=True)
    add_users()

    contest_start_at = int(os.environ.get('GAME_START_AT', 0))
    contest_period = int(os.environ.get('GAME_PERIOD', MATCH_PERIOD * NUM_GAMES_PER_SET * 10000))
    assert contest_period % MATCH_PERIOD == 0
    num_games = contest_period // MATCH_PERIOD
    assert num_games % NUM_GAMES_PER_SET == 0
    print(f'num_games={num_games}', flush=True)

    wait(contest_start_at)

    while get_stop_matching():
        sleep(10)

    ranking = Ranking()
    ranking.load()
    game = load_game(ranking.last_game_id)

    while True:
        if game is not None:
            result = run_game(game)
            ranking.update(game.game_id, result)
            ranking.save()
        while get_stop_matching():
            sleep(10)
        game = start_game(ranking.last_game_id, contest_start_at, num_games)
        if game is None:
            print('contest is over', flush=True)
            break

    sleep(1e8)
