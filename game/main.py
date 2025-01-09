import asyncio
import re
import subprocess
import json
from time import time_ns
from typing import NamedTuple
import redis
import os
from math import hypot

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(root_path="/api", docs_url="/docs_u5nn3l340", openapi_url="/openapi_u5nn3l340.json", redoc_url="/redoc_u5nn3l340")

redis_pool = redis.ConnectionPool(
    host=os.environ.get("GAMEDB_HOST", "localhost"),
    port=os.environ.get("GAMEDB_PORT", "6379"),
    db=0,
)

class HealthCheck(BaseModel):
    status: str = "OK"


@app.get("/health", response_model=HealthCheck, include_in_schema=False)
def get_health() -> HealthCheck:
    r = redis.Redis(connection_pool=redis_pool)
    if not r.ping():
        raise HTTPException(status_code=500, detail="redis connection error")
    return HealthCheck()


float_pattern = re.compile(r'[+-]?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([Ee][+-]?[0-9]+)?')

STATE_TIME_LIMIT = 1000
OPTIONAL_TIME_LIMIT = 1000
PLAN_TIME_LIMIT = 1000
EVAL_TIME_LIMIT = 1000
RANKING_TIME_LIMIT = 1000
HISTORY_TIME_LIMIT = 1000
RESULT_TIME_LIMIT = 1000
GAME_RESULT_TIME_LIMIT = 1000
EVALUATE_TIMEOUT = 0.1
STR_MAX_LENGTH = 100
PLAN_MAX_LENGTH = 300


def validate_float(s: str) -> bool:
    return len(s) <= STR_MAX_LENGTH and float_pattern.fullmatch(s) is not None


class Point(BaseModel):
    x: str
    y: str


class MasterData(NamedTuple):
    game_id: int
    required: list[Point]
    checkpoint_size: str
    plan_length: int
    optional_end_time: int
    plan_end_time: int


def get_timestamp() -> int:
    return time_ns() // 1000000


def get_user_id(token: str) -> str | None:
    r = redis.Redis(connection_pool=redis_pool)
    user_id: bytes | None = r.hget('user_token', token)
    if user_id is None:
        return None
    return user_id.decode()


def get_cur_game_id(r: redis.Redis) -> int | None:
    game_id: bytes | None = r.get('cur_game')
    if game_id is None:
        return None
    return int(game_id)


def get_master_data() -> MasterData | None:
    r = redis.Redis(connection_pool=redis_pool)
    game_id = get_cur_game_id(r)
    if game_id is None:
        return None
    data = r.hget('master_data', str(game_id))
    assert data is not None
    v = json.loads(data)
    return MasterData(
        game_id=game_id,
        required=[Point(x=p['x'], y=p['y']) for p in v['required']],
        checkpoint_size=v['checkpoint_size'],
        plan_length=v['plan_length'],
        optional_end_time=v['optional_end_time'],
        plan_end_time=v['plan_end_time'],
    )


def get_can_submit_optional(game_id: int, user_id: str) -> bool:
    r = redis.Redis(connection_pool=redis_pool)
    return bool(r.sismember(f'optional_user_{game_id}', user_id))


def get_optional(game_id: int) -> tuple[bool, list[Point]]:
    r = redis.Redis(connection_pool=redis_pool)
    if not r.sismember('closed_optional', str(game_id)):
        return False, []

    optional = []
    for v in r.hgetall(f'optional_point_{game_id}').values():
        vv = json.loads(v)
        optional.append(Point(x=vv[0], y=vv[1]))
    return True, optional


def add_optional(game_id: int, user_id: str, x: str, y: str) -> bool:
    r = redis.Redis(connection_pool=redis_pool)
    return bool(r.eval(f'''
local game_id = KEYS[1]
local user_id = KEYS[2]
local pos = KEYS[3]
if redis.call("SISMEMBER", "closed_optional", game_id) == 1 then
  return 0
end
return redis.call("HSETNX", "optional_point_"..game_id, user_id, pos)
    ''', 3, str(game_id), user_id, json.dumps([x, y])))


def add_plan(game_id: int, user_id: str, timestamp: int, plan: str) -> int:
    r = redis.Redis(connection_pool=redis_pool)
    return int(r.eval(f'''
local game_id = KEYS[1]
local user_id = KEYS[2]
local timestamp = KEYS[3]
local plan = KEYS[4]
if redis.call("SISMEMBER", "closed_plan", game_id) == 1 then
  return 0
end
local plan_key = user_id.."/"..timestamp
redis.call("HSET", "plan_"..game_id, plan_key, plan)
if redis.call("SISMEMBER", "closed_optional", game_id) == 1 then
  redis.call("RPUSH", "t_plan_"..game_id, plan_key)
  return 1
end
redis.call("RPUSH", "u_plan_"..game_id, plan_key)
return 2
    ''', 4, str(game_id), user_id, str(timestamp), plan))


def save_result(game_id: int, user_id: str, timestamp: int, results: str, scores: str, score: int) -> None:
    r = redis.Redis(connection_pool=redis_pool)
    plan_key = f'{user_id}/{timestamp}'
    r.hset(f'results_{game_id}', plan_key, results)
    r.hset(f'scores_{game_id}', plan_key, scores)
    r.rpush(f'timestamp_{user_id}_{game_id}', f'{timestamp} {score}')


def get_set_time_limit(lock_type: str, user_id: str, now: int, time_limit: int) -> int:
    r = redis.Redis(connection_pool=redis_pool)
    res = r.eval('''
local field = KEYS[1]
local now = tonumber(KEYS[2])
local time_limit = tonumber(KEYS[3])
local t = redis.call('hget', 'unlock_time', field)
if t and now < tonumber(t) then
  return tostring(t)
end
redis.call('hset', 'unlock_time', field, now + time_limit)
return 'ok'
    ''', 3, f'{lock_type}_{user_id}', str(now), str(time_limit))
    return -1 if res == b"ok" else int(res)


async def wait_unlock(lock_type: str, user_id: str, time_limit: int) -> int:
    now = get_timestamp()
    unlock_time = -1
    while True:
        ut = get_set_time_limit(lock_type, user_id, now, time_limit)
        if ut < 0:
            break
        if unlock_time < 0:
            unlock_time = ut
        elif unlock_time != ut:
            raise HTTPException(status_code=400, detail="error_time_limit")
        await asyncio.sleep(max(1, unlock_time - now) * 1e-3)
        now = get_timestamp()
        while now < unlock_time:
            await asyncio.sleep(1e-3)
            now = get_timestamp()
    return now


async def auth(token: str, lock_type: str, lock_time: int) -> tuple[str, int, MasterData]:
    user_id = get_user_id(token)
    if user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    timestamp = await wait_unlock(lock_type, user_id, lock_time)

    md = get_master_data()
    if md is None:
        raise HTTPException(status_code=404, detail="game not found")

    return user_id, timestamp, md


class StateResponse(BaseModel):
    game_id: int
    optional_time: int
    plan_time: int
    can_submit_optional: bool
    is_set_optional: bool
    required: list[Point]
    optional: list[Point]
    checkpoint_size: str
    plan_length: int


@app.get("/state/{token}", response_model=StateResponse)
async def get_state(token: str) -> StateResponse:
    user_id, timestamp, md = await auth(token, "state", STATE_TIME_LIMIT)

    optional_time = max(0, md.optional_end_time - timestamp)
    plan_time = max(0, md.plan_end_time - timestamp)

    if optional_time > 0:
        is_set_optional = False
        optional = []
        can_submit_optional = get_can_submit_optional(md.game_id, user_id)
    else:
        can_submit_optional = False
        is_set_optional, optional = get_optional(md.game_id)

    return StateResponse(
        game_id=md.game_id,
        optional_time=optional_time,
        plan_time=plan_time,
        can_submit_optional=can_submit_optional,
        is_set_optional=is_set_optional,
        required=md.required,
        optional=optional,
        checkpoint_size=md.checkpoint_size,
        plan_length=md.plan_length,
    )


class SubmitOptionalRequest(BaseModel):
    x: str
    y: str


class SubmitOptionalResponse(BaseModel):
    status: bool


@app.post("/optional/{token}/{game_id}", response_model=SubmitOptionalResponse)
async def submit_optional(token: str, game_id: int, req: SubmitOptionalRequest) -> SubmitOptionalResponse:
    if not (validate_float(req.x) and validate_float(req.y)):
        raise HTTPException(status_code=400, detail="invalid number format")

    x = float(req.x)
    y = float(req.y)
    if not (-10 < x < 10 and -10 < y < 10 and hypot(x, y) >= 1):
        raise HTTPException(status_code=400, detail="not satisfy constraint")

    user_id, timestamp, md = await auth(token, "optional", OPTIONAL_TIME_LIMIT)

    if md.game_id != game_id:
        return SubmitOptionalResponse(status=False)

    if timestamp >= md.optional_end_time:
        return SubmitOptionalResponse(status=False)

    if not get_can_submit_optional(game_id, user_id):
        return SubmitOptionalResponse(status=False)

    if not add_optional(game_id, user_id, req.x, req.y):
        return SubmitOptionalResponse(status=False)

    return SubmitOptionalResponse(status=True)


class EvalElement(BaseModel):
    p: Point
    v: Point
    score: int


class SubmitPlanResponse(BaseModel):
    status: bool
    result: list[EvalElement]
    is_set_optional: bool


@app.post("/plan/{token}/{game_id}", response_model=SubmitPlanResponse)
async def submit_plan(token: str, game_id: int, req: list[str]) -> SubmitPlanResponse:
    if len(req) > PLAN_MAX_LENGTH:
        raise HTTPException(status_code=404, detail="invalid plan length")

    for x in req:
        if not validate_float(x):
            raise HTTPException(status_code=400, detail="invalid number format")

        if not (-10 < float(x) < 10):
            raise HTTPException(status_code=400, detail="not satisfy constraint")

    user_id, timestamp, md = await auth(token, "plan", PLAN_TIME_LIMIT)

    if len(req) != md.plan_length:
        raise HTTPException(status_code=404, detail="invalid plan length")

    if md.game_id != game_id:
        return SubmitPlanResponse(status=False, result=[], is_set_optional=False)

    if timestamp >= md.plan_end_time:
        return SubmitPlanResponse(status=False, result=[], is_set_optional=False)

    plan = ' '.join(req)
    plan_res = add_plan(game_id, user_id, timestamp, plan)
    if plan_res == 0:
        return SubmitPlanResponse(status=False, result=[], is_set_optional=False)

    if plan_res == 1:
        is_set_optional, optional = get_optional(game_id)
        assert is_set_optional
    else:
        is_set_optional, optional = False, []

    eval_input = f'{md.checkpoint_size} {len(md.required)} {len(optional)} {md.plan_length}\n'
    for pos in md.required:
        eval_input += f'{pos.x} {pos.y}\n'
    for pos in optional:
        eval_input += f'{pos.x} {pos.y}\n'
    eval_input += f'{plan}\n'
    try:
        eval_output = subprocess.run(
            './evaluate', input=eval_input,
            capture_output=True, text=True, check=True, timeout=EVALUATE_TIMEOUT).stdout
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr, e)

    result: list[EvalElement] = []
    results: list[str] = []
    scores: list[str] = []
    for line in eval_output.split('\n')[:md.plan_length]:
        a = line.split(' ')
        result.append(EvalElement(
            p=Point(x=a[0], y=a[1]),
            v=Point(x=a[2], y=a[3]),
            score=int(a[4]),
        ))
        results.append(line)
        scores.append(a[4])

    if is_set_optional:
        save_result(game_id, user_id, timestamp, '\n'.join(results), ' '.join(scores), result[-1].score)

    return SubmitPlanResponse(
        status=True,
        result=result,
        is_set_optional=is_set_optional,
    )


class EvalRequest(BaseModel):
    checkpoint_size: str
    required: list[Point]
    optional: list[Point]
    plan: list[str]


class EvalResponse(BaseModel):
    result: list[EvalElement]


@app.post("/eval/{token}", response_model=EvalResponse)
async def evaluate(token: str, req: EvalRequest) -> EvalResponse:
    if not validate_float(req.checkpoint_size):
        raise HTTPException(status_code=400, detail="invalid number format")

    if not (0 < float(req.checkpoint_size) <= 0.1):
        raise HTTPException(status_code=400, detail="invalid checkpoint_size")

    if len(req.required) > 10 or len(req.required) == 0:
        raise HTTPException(status_code=404, detail="invalid required length")

    if len(req.optional) > 10:
        raise HTTPException(status_code=404, detail="invalid optional length")

    if len(req.plan) > PLAN_MAX_LENGTH:
        raise HTTPException(status_code=404, detail="invalid plan length")

    for p in req.required:
        if not (validate_float(p.x) and validate_float(p.y)):
            raise HTTPException(status_code=400, detail="invalid number format")

    for p in req.optional:
        if not (validate_float(p.x) and validate_float(p.y)):
            raise HTTPException(status_code=400, detail="invalid number format")

    for x in req.plan:
        if not validate_float(x):
            raise HTTPException(status_code=400, detail="invalid number format")

    user_id = get_user_id(token)
    if user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    await wait_unlock("eval", user_id, EVAL_TIME_LIMIT)

    eval_input = f'{req.checkpoint_size} {len(req.required)} {len(req.optional)} {len(req.plan)}\n'
    for pos in req.required:
        eval_input += f'{pos.x} {pos.y}\n'
    for pos in req.optional:
        eval_input += f'{pos.x} {pos.y}\n'
    for x in req.plan:
        eval_input += f'{x}\n'
    try:
        eval_output = subprocess.run(
            './evaluate', input=eval_input,
            capture_output=True, text=True, check=True, timeout=EVALUATE_TIMEOUT).stdout
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr, e)

    result: list[EvalElement] = []
    for line in eval_output.split('\n')[:len(req.plan)]:
        a = line.split(' ')
        result.append(EvalElement(
            p=Point(x=a[0], y=a[1]),
            v=Point(x=a[2], y=a[3]),
            score=int(a[4]),
        ))

    return EvalResponse(result=result)


class RankingElement(BaseModel):
    rank: int
    user_id: str
    point: str


class RankingResponse(BaseModel):
    game_id: int
    ranking: list[RankingElement]


def get_ranking_data() -> RankingResponse:
    r = redis.Redis(connection_pool=redis_pool)

    s: bytes = r.get('ranking')
    if s:
        a = s.decode().split('\n')
        game_id = int(a[0])
        ranking: list[RankingElement] = []
        for rank, x in enumerate(a[1:], start=1):
            user_id, point = x.split(' ')
            ranking.append(RankingElement(rank=rank, user_id=user_id, point=point))
    else:
        game_id = 0
        ranking = []

    return RankingResponse(
        game_id=game_id,
        ranking=ranking,
    )


@app.get("/ranking/{token}", response_model=RankingResponse)
async def get_ranking(token: str) -> RankingResponse:
    my_user_id = get_user_id(token)
    if my_user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    await wait_unlock("ranking", my_user_id, RANKING_TIME_LIMIT)

    return get_ranking_data()


class HistoryElement(BaseModel):
    game_id: int
    start_at: int
    rank: int
    score: int


class HistoryResponse(BaseModel):
    history: list[HistoryElement]


def get_history_data(user_id: str) -> HistoryResponse:
    r = redis.Redis(connection_pool=redis_pool)

    rank_score: dict[int, tuple[int, int]] = {}
    for x in r.lrange(f'rank_score_{user_id}', 0, -1)[::-1]:
        game_id, rank, score = map(int, x.decode().split(' '))
        rank_score[game_id] = (rank, score)

    history: list[HistoryElement] = []
    for x in r.lrange(f'history', 0, -1)[::-1]:
        game_id, start_time = map(int, x.decode().split(' '))
        rank, score = rank_score.get(game_id, (0, 0))
        history.append(HistoryElement(
            game_id=game_id,
            start_at=start_time,
            rank=rank,
            score=score,
        ))

    return HistoryResponse(
        history=history,
    )


@app.get("/history/{token}", response_model=HistoryResponse)
async def get_history(token: str) -> HistoryResponse:
    my_user_id = get_user_id(token)
    if my_user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    await wait_unlock("history", my_user_id, HISTORY_TIME_LIMIT)

    return get_history_data(my_user_id)


class ResultElement(BaseModel):
    p: Point
    v: Point
    score: int
    target: int
    opt: int


class ResultResponse(BaseModel):
    state: StateResponse
    plan: list[str]
    result: list[ResultElement]


def get_result_data(game_id: int, user_id: str, timestamp: int) -> ResultResponse:
    r = redis.Redis(connection_pool=redis_pool)

    if timestamp == 0:
        timestamp = r.hget(f'best_result_{game_id}', user_id)
        if timestamp is None:
            raise HTTPException(status_code=404, detail="invalid game_id or user_id")
        timestamp = int(timestamp)

    plan_key = f'{user_id}/{timestamp}'

    # noinspection PyTypeChecker
    plan: bytes = r.hget(f'plan_{game_id}', plan_key)
    if not plan:
        raise HTTPException(status_code=404, detail="invalid game_id or user_id")

    # noinspection PyTypeChecker
    results: bytes = r.hget(f'results_{game_id}', plan_key)
    if not results:
        raise HTTPException(status_code=404, detail="invalid game_id or user_id")

    data = r.hget('master_data', str(game_id))
    if not data:
        raise HTTPException(status_code=404, detail="invalid game_id or user_id")
    md = json.loads(data)

    optional = []
    for v in r.hgetall(f'optional_point_{game_id}').values():
        vv = json.loads(v)
        optional.append(Point(x=vv[0], y=vv[1]))

    result: list[ResultElement] = []
    for line in results.decode().split('\n'):
        a = line.split(' ')
        result.append(ResultElement(
            p=Point(x=a[0], y=a[1]),
            v=Point(x=a[2], y=a[3]),
            score=int(a[4]),
            target=int(a[5]),
            opt=int(a[6]),
        ))

    return ResultResponse(
        state=StateResponse(
            game_id=game_id,
            optional_time=0,
            plan_time=0,
            can_submit_optional=False,
            is_set_optional=True,
            required=[Point(x=p['x'], y=p['y']) for p in md['required']],
            optional=optional,
            checkpoint_size=md['checkpoint_size'],
            plan_length=md['plan_length'],
        ),
        plan=plan.decode().split(' '),
        result=result,
    )


@app.get("/result/{token}/{game_id}/{user_id}/{submit_at}", response_model=ResultResponse)
async def get_result(token: str, game_id: int, user_id: str, submit_at: int) -> ResultResponse:
    my_user_id = get_user_id(token)
    if my_user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    if user_id != my_user_id and submit_at != 0:
        raise HTTPException(status_code=404, detail="invalid game_id or user_id")

    await wait_unlock("result", my_user_id, RESULT_TIME_LIMIT)

    return get_result_data(game_id, user_id, submit_at)


class GameResultElement(BaseModel):
    user_id: str
    rank: int
    score: int


class GameResultSubmit(BaseModel):
    submit_at: int
    score: int
    is_best: bool


class GameResultResponse(BaseModel):
    ranking: list[GameResultElement]
    submit: list[GameResultSubmit]


def get_game_result_data(user_id: str, game_id: int) -> GameResultResponse:
    r = redis.Redis(connection_pool=redis_pool)
    results: bytes = r.get(f'ranking_{game_id}')
    assert results is not None
    ranking: list[GameResultElement] = []
    rank = 1
    for line in results.decode().split('\n'):
        a = line.split(' ')
        ranking.append(GameResultElement(
            user_id=a[0],
            rank=rank,
            score=int(a[1]),
        ))
        rank += 1

    best_timestamp = r.hget(f'best_result_{game_id}', user_id)
    if best_timestamp:
        best_timestamp = int(best_timestamp)
    submit: list[GameResultSubmit] = []
    for x in r.lrange(f'timestamp_{user_id}_{game_id}', 0, -1):
        submit_at, score = map(int, x.decode().split(' '))
        submit.append(GameResultSubmit(
            submit_at=submit_at,
            score=score,
            is_best=submit_at == best_timestamp,
        ))

    return GameResultResponse(ranking=ranking, submit=submit)


@app.get("/game/{token}/{game_id}", response_model=GameResultResponse)
async def get_game_result(token: str, game_id: int) -> GameResultResponse:
    my_user_id = get_user_id(token)
    if my_user_id is None:
        raise HTTPException(status_code=404, detail="invalid token")

    await wait_unlock("game_result", my_user_id, GAME_RESULT_TIME_LIMIT)

    return get_game_result_data(my_user_id, game_id)
