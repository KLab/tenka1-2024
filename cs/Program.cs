using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

internal class GameState
{
    public double Px;
    public double Py;
    public double Vx;
    public double Vy;
    public int Score;
    public int TargetIndex;
    public required bool[] UsedOptional;
}

internal class Simulator
{
    public readonly double CheckpointSize;
    public IReadOnlyList<(double, double)> Required;
    public IReadOnlyList<(double, double)> Optional;

    public Simulator(StateResponse state)
    {
        CheckpointSize = double.Parse(state.CheckpointSize);
        Required = state.Required.Select(p => (double.Parse(p.X), double.Parse(p.Y))).ToList();
        Optional = state.Optional.Select(p => (double.Parse(p.X), double.Parse(p.Y))).ToList();
    }

    // t^3 + 3bt^2 + 2ct + 2d = 0 を解き昇順ソート済の実数解を返却する
    private static List<double> SolveCubicEquation(double b, double c, double d)
    {
        // カルダノの方法を用いて解く
        // 参考: https://ja.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F
        // t = y - b と置く
        // y^3 + 3hy - 2q = 0
        var q = c * b - b * b * b - d;
        var h = c * 2 / 3 - b * b;
        var r = q * q + h * h * h;
        var y = r >= 0 ?
            Math.Cbrt(q + Math.Sqrt(r)) + Math.Cbrt(q - Math.Sqrt(r)) :
            // z = q + sqrt(-r) i とすると、
            // zの絶対値は hypot(q, sqrt(-r))
            // zの偏角は atan2(sqrt(-r), q)
            // であり、yは以下の式で求まる
            Math.Cbrt(double.Hypot(q, Math.Sqrt(-r))) * Math.Cos(Math.Atan2(Math.Sqrt(-r), q) / 3) * 2;
        var t1 = y - b;
        // t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
        var s = 3 * b + t1;
        var u = 2 * c + t1 * s;
        // t^2 + st + u = 0 を解く
        var z = s * s - 4 * u;
        if (z >= 0) {
            var tt = new List<double> { t1, (-s + Math.Sqrt(z)) / 2, (-s - Math.Sqrt(z)) / 2 };
            tt.Sort();
            return tt;
        }
        return new List<double> { t1 };
    }

    // 目的地の座標(tx, ty)を訪れる時刻t(t0 < t <= 1.0)を求める
    // 目的地に到達しないなら-1を返却する
    private double CalcVisitTime(GameState gs, double ax, double ay, double tx, double ty, double t0)
    {
        // 質点の座標を(px, py), 速度を(vx, vy), 加速度を(ax, ay) とし, 目的地の座標を(tx, ty)とする.
        // 時刻tにおける質点の座標は x(t) = px + vx t + ax t^2/2, y(t) = py + vy t + ax t^2/2 である.
        // 時刻tにおける質点の座標と目的地の座標との距離の2乗をf(t)とする. f(t) = (x(t)-tx)^2 + (y(t)-ty)^2 である.
        // f(t)を整理すると以下のようになる
        // f(t) = t^4/4 + bt^3 + ct^2 + 2dt + dx^2 + dy^2
        var dx = gs.Px - tx;
        var dy = gs.Py - ty;
        var b = gs.Vx * ax + gs.Vy * ay;
        var c = gs.Vx * gs.Vx + gs.Vy * gs.Vy + dx * ax + dy * ay;
        var d = dx * gs.Vx + dy * gs.Vy;

        // f(t)を微分すると f'(t) = t^3 + 3bt^2 + 2ct + 2d となる.
        // f'(t) = 0 の解を求め, f(t)が最小となるtの候補を求める.
        var candidates  = SolveCubicEquation(b, c, d);

        // 求めたtの候補を小さい順に試し, 最初に条件を満たしたものを返却する.
        for (var i = 0; i < candidates.Count; i++) {
            var t = candidates[i];
            // f(candidates[1])は極大値なので無視する
            if (i != 1 && t0 < t && t < 1.0 && G(t))
            {
                return t;
            }
        }

        // t=1.0が条件を満たしているケースを考慮する
        if (G(1.0)) {
            return 1.0;
        }
        return -1.0;

        // 時刻tで質点が目的地に到達しているかを判定する
        bool G(double t)
        {
            var x = gs.Px + gs.Vx * t + 0.5 * ax * t * t - tx;
            var y = gs.Py + gs.Vy * t + 0.5 * ay * t * t - ty;
            return double.Hypot(x, y) <= CheckpointSize;
        }
    }

    // 時刻t=0からt=1までの間に訪れる目的地の数を求める
    private (int, int, bool[]) Calc(GameState gs, double ax, double ay)
    {
        var res = 0;
        var idx = gs.TargetIndex;
        var used = gs.UsedOptional.ToArray();

        // 必須目的地への到達判定
        var t1 = new List<double>();
        {
            var t0 = 0.0;
            while (t0 < 1.0)
            {
                var (tx, ty) = Required[idx];
                var tt = CalcVisitTime(gs, ax, ay, tx, ty ,t0);
                if (tt < 0) break;
                t0 = tt;
                ++ res;
                if (++ idx == Required.Count) {
                    idx = 0;
                    t1.Add(t0);
                }
            }
        }

        // 任意目的地への到達判定
        for (var i = 0; i < Optional.Count; ++ i) {
            if (used[i] && t1.Count == 0) continue;
            var (tx, ty) = Optional[i];
            var t0 = 0.0;
            var j = 0;
            while (t0 < 1.0) {
                var tt = CalcVisitTime(gs, ax, ay, tx, ty, t0);
                if (tt < 0) break;
                t0 = tt;
                // 最後の必須目的地に到達した直後にこの任意目的地に到達したケースを考慮
                while (j < t1.Count && t1[j] < t0) {
                    ++ j;
                    used[i] = false;
                }
                if (!used[i]) {
                    ++ res;
                    used[i] = true;
                }
            }
            //  最後の必須目的地に到達した際に任意目的地が再度利用可能になる
            if (j < t1.Count) {
                used[i] = false;
            }
        }
        return (res, idx, used);
    }

    // 操作θを与え単位時間だけ状態を更新する
    public GameState Update(GameState gs, double th)
    {
        // 加速度を決定する
        var ax = Math.Cos(th);
        var ay = Math.Sin(th);
        // 訪れる目的地の数を求める
        var (s, idx, used) = Calc(gs, ax, ay);
        // 移動後の状態に更新する
        return new GameState
        {
            Px = gs.Px + gs.Vx + 0.5 * ax,
            Py = gs.Py + gs.Vy + 0.5 * ay,
            Vx = gs.Vx + ax,
            Vy = gs.Vy + ay,
            Score = gs.Score + s,
            TargetIndex = idx,
            UsedOptional = used
        };
    }
}

internal class Program
{
    private static readonly HttpClient Client = new();
    private readonly Random _random = new();

    private readonly string _gameServer;
    private readonly string _token;
    private StateResponse _state;

    // ゲームサーバのAPIを呼び出す
    private async Task<byte[]?> CallApi(string x, JsonContent? content = null)
    {
        var url = $"{_gameServer}{x}";
        // 5xxエラーまたはHttpRequestExceptionの際は100ms空けて5回までリトライする
        for (var i = 0; i < 5; i++)
        {
            Console.WriteLine(url);
            try
            {
                var res = await (content == null ? Client.GetAsync(url) : Client.PostAsync(url, content));
                if ((int)res.StatusCode == 200)
                {
                    return await res.Content.ReadAsByteArrayAsync();
                }

                if (500 <= (int)res.StatusCode && (int)res.StatusCode < 600)
                {
                    Console.WriteLine($"{res.StatusCode}");
                    Thread.Sleep(100);
                    continue;
                }

                var resContent = await res.Content.ReadAsStringAsync();
                Console.WriteLine($"Api Error status_code:{res.StatusCode} body:{resContent}");
                return null;
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine($"{e.Message}");
                Thread.Sleep(100);
            }
        }
        throw new Exception("Api Error");
    }

    // /api/stateを呼び出す
    private async Task CallState()
    {
        var json = await CallApi($"/api/state/{_token}");
        if (json == null)
        {
            return;
        }

        _state = JsonSerializer.Deserialize<StateResponse>(json);
    }

    // /api/optionalを呼び出す 任意目的地を追加する
    private async Task CallOptional(double x, double y)
    {
        var content = JsonContent.Create(new Dictionary<string, string>
        {
            ["x"] = x.ToString(CultureInfo.InvariantCulture),
            ["y"] = y.ToString(CultureInfo.InvariantCulture),
        });
        await CallApi($"/api/optional/{_token}/{_state.GameId}", content);
    }

    // /api/plan を呼び出す 移動予定を決定する
    private async Task CallPlan(List<double> plan)
    {
        var content = JsonContent.Create(
            plan.Select(x => x.ToString(CultureInfo.InvariantCulture)).ToList());
        await CallApi($"/api/plan/{_token}/{_state.GameId}", content);
    }

    private async Task<byte[]?> CallEval(List<double> plan)
    {
        var content = JsonContent.Create(new Dictionary<string, object>
        {
            ["checkpoint_size"] = _state.CheckpointSize,
            ["required"] = _state.Required,
            ["optional"] = _state.Optional,
            ["plan"] = plan.Select(x => x.ToString(CultureInfo.InvariantCulture)).ToList(),
        });
        return await CallApi($"/api/eval/{_token}", content);
    }

    private (double, double) ThinkOptional()
    {
        for (;;)
        {
            // -5.0 から 5.0 の範囲でランダムに生成する
            var x = (_random.NextDouble() - 0.5) * 10;
            var y = (_random.NextDouble() - 0.5) * 10;
            if (double.Hypot(x, y) >= 1)
            {
                return (x, y);
            }
        }
    }

    private List<double> ThinkPlan()
    {
        var plan = new List<double>();
        var gs = new GameState{UsedOptional = new bool[_state.Optional.Length]};
        var sim = new Simulator(_state);
        for (var i = 0; i < _state.PlanLength; i++)
        {
            var bestTh = 0.0;
            var bestS = -1;
            var bestD = 1e+300;
            for (var j = 0; j < 10; j++)
            {
                // 0~2πの範囲でthをランダムに選び
                // 現在のベストよりもスコアが増加しているか, 次の必須目的地に近づくなら採用する
                var th = _random.NextDouble() * 2 * Math.PI;
                var (s, d) = Evaluate(sim.Update(gs, th));
                if (s > bestS || (s == bestS && d < bestD))
                {
                    bestTh = th;
                    bestS = s;
                    bestD = d;
                }
            }

            plan.Add(bestTh);
            gs = sim.Update(gs, bestTh);
        }

        return plan;

        // 状態の評価を行う
        (int, double) Evaluate(GameState gs)
        {
            var (tx, ty) = sim.Required[gs.TargetIndex];
            return (gs.Score, double.Hypot(gs.Px - tx, gs.Py - ty));
        }
    }

    // BOTのメイン処理
    private async Task Solve()
    {
        var gameId = -1;
        for (;;)
        {
            // game_idの更新を待つ
            for (;;)
            {
                await CallState();
                if (gameId != _state.GameId)
                {
                    gameId = _state.GameId;
                    break;
                }

                await Task.Delay(1000);
            }

            // 任意目的地の提出が可能な場合に提出する
            if (_state.CanSubmitOptional)
            {
                var (x, y) = ThinkOptional();
                await CallOptional(x, y);
            }

            // ゲーム開始を待つ
            while (!_state.IsSetOptional)
            {
                await Task.Delay(1000);
                await CallState();
            }

            // planを考えて提出する
            if (_state.PlanTime > 0)
            {
                var plan = ThinkPlan();
                await CallPlan(plan);
            }
        }

        // ReSharper disable once FunctionNeverReturns
    }

    private async Task Test(string filePath)
    {
        _state = JsonSerializer.Deserialize<StateResponse>(await File.ReadAllBytesAsync(filePath));
        var plan = ThinkPlan();
        var res = await CallEval(plan);
        if (res == null)
        {
            throw new Exception();
        }

        var content = JsonContent.Create(plan.Select(x => x.ToString(CultureInfo.InvariantCulture)).ToList());
        await File.WriteAllBytesAsync("./eval_plan.json", await content.ReadAsByteArrayAsync());

        var json = JsonSerializer.Deserialize<JsonElement>(res);
        var result = json.GetProperty("result");
        var score = result[result.GetArrayLength() - 1].GetProperty("score");
        Console.WriteLine($"score = {score}");
    }

    private Program()
    {
        _gameServer = Environment.GetEnvironmentVariable("GAME_SERVER") ?? "https://2024.gcp.tenka1.klab.jp";
        _token = Environment.GetEnvironmentVariable("TOKEN") ?? "YOUR_TOKEN";
    }

    private static async Task Main(string[] args)
    {
        if (args.Length >= 1)
        {
            await new Program().Test(args[0]);
        }
        else
        {
            await new Program().Solve();
        }
    }
}

internal readonly struct Point
{
    [JsonPropertyName("x")]
    public string X { get; }
    [JsonPropertyName("y")]
    public string Y { get; }
    [JsonConstructor]
    public Point(string x, string y)
    {
        X = x;
        Y = y;
    }
}

internal readonly struct StateResponse
{
    [JsonPropertyName("game_id")]
    public int GameId { get; }
    [JsonPropertyName("optional_time")]
    public int OptionalTime { get; }
    [JsonPropertyName("plan_time")]
    public int PlanTime { get; }
    [JsonPropertyName("can_submit_optional")]
    public bool CanSubmitOptional { get; }
    [JsonPropertyName("is_set_optional")]
    public bool IsSetOptional { get; }
    [JsonPropertyName("required")]
    public Point[] Required { get; }
    [JsonPropertyName("optional")]
    public Point[] Optional { get; }
    [JsonPropertyName("checkpoint_size")]
    public string CheckpointSize { get; }
    [JsonPropertyName("plan_length")]
    public int PlanLength { get; }
    [JsonConstructor]
    public StateResponse(int gameId, int optionalTime, int planTime, bool canSubmitOptional, bool isSetOptional, Point[] required, Point[] optional, string checkpointSize, int planLength)
    {
        GameId = gameId;
        OptionalTime = optionalTime;
        PlanTime = planTime;
        CanSubmitOptional = canSubmitOptional;
        IsSetOptional = isSetOptional;
        Required = required;
        Optional = optional;
        CheckpointSize = checkpointSize;
        PlanLength = planLength;
    }
}
