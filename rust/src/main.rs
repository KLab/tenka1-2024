use anyhow::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::io::Write;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Api Error")]
    ApiError(),
}

#[derive(Default, Debug, Serialize, Deserialize)]
struct StateResponse {
    game_id: i32,
    optional_time: i32,
    plan_time: i32,
    can_submit_optional: bool,
    is_set_optional: bool,
    required: Vec<Point>,
    optional: Vec<Point>,
    checkpoint_size: String,
    plan_length: i32,
}

#[derive(Default, Debug, Serialize, Deserialize)]
struct EvalRequest {
    checkpoint_size: String,
    required: Vec<Point>,
    optional: Vec<Point>,
    plan: Vec<String>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
struct EvalResponse {
    result: Vec<EvalElement>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvalElement {
    p: Point,
    v: Point,
    score: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Point {
    x: String,
    y: String,
}

#[derive(Debug, Default)]
struct GameState {
    px: f64,
    py: f64,
    vx: f64,
    vy: f64,
    score: i32,
    target_idx: i32,
    used_optional: Vec<bool>,
}

impl GameState {
    fn new(optional_len: usize) -> Self {
        Self {
            used_optional: vec![false; optional_len],
            ..Default::default()
        }
    }
}

#[derive(Debug)]
struct Simulator {
    checkpoint_size: f64,
    required: Vec<(f64, f64)>,
    optional: Vec<(f64, f64)>,
}

impl Simulator {
    fn new(state: &StateResponse) -> Self{
        Self {
            checkpoint_size: state.checkpoint_size.parse().unwrap(),
            required: state.required
                .iter()
                .map(|p| (p.x.parse().unwrap(), p.y.parse().unwrap()))
                .collect(),
            optional: state.optional
                .iter()
                .map(|p| (p.x.parse().unwrap(), p.y.parse().unwrap()))
                .collect(),
        }
    }

    /// t^3 + 3bt^2 + 2ct + 2d = 0 を解き昇順ソート済の実数解を返却する
    fn solve_cubic_equation(b: f64, c: f64, d: f64) -> Vec<f64> {
        // カルダノの方法を用いて解く
        // 参考: https://ja.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F
        // t = y - b と置く
        // y^3 + 3hy - 2q = 0
        let q = c * b - b * b * b - d;
        let h = c * 2.0 / 3.0 - b * b;
        let r = q * q + h * h * h;
        let y: f64 = if r >= 0.0 {
            (q + r.sqrt()).cbrt() + (q - r.sqrt()).cbrt()
        } else {
            // z = q + sqrt(-r) i とすると、
            // zの絶対値は hypot(q, sqrt(-r))
            // zの偏角は atan2(sqrt(-r), q)
            // であり、yは以下の式で求まる
            f64::hypot(q, (-r).sqrt()).cbrt() * ((-r).sqrt().atan2(q) / 3.0).cos() * 2.0
        };

        let t1 = y - b;

        // t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
        let s = 3.0 * b + t1;
        let u = 2.0 * c + t1 * s;

        // t^2 + st + u = 0 を解く
        let z = s * s - 4.0 * u;
        if z >= 0.0 {
            let mut solutions = vec![t1, (-s + z.sqrt()) / 2.0, (-s - z.sqrt()) / 2.0];
            solutions.sort_by(|a, b| a.partial_cmp(b).unwrap());
            solutions
        } else {
            vec![t1]
        }
    }

    /// 目的地の座標(tx, ty)を訪れる時刻t(t0 < t <= 1.0)を求める
    /// 目的地に到達しないなら-1を返却する
    fn calc_visit_time(&self, gs: &GameState, ax: f64, ay: f64, tx: f64, ty: f64, t0: f64) -> f64 {
        // 時刻tで質点が目的地に到達しているかを判定する
        let g = |t: f64| -> bool {
            let x = gs.px + gs.vx * t + 0.5 * ax * t * t - tx;
            let y = gs.py + gs.vy * t + 0.5 * ay * t * t - ty;
            x.hypot(y) <= self.checkpoint_size
        };

        // 質点の座標を(px, py), 速度を(vx, vy), 加速度を(ax, ay) とし, 目的地の座標を(tx, ty)とする.
        // 時刻tにおける質点の座標は x(t) = px + vx t + ax t^2/2, y(t) = py + vy t + ax t^2/2 である.
        // 時刻tにおける質点の座標と目的地の座標との距離の2乗をf(t)とする. f(t) = (x(t)-tx)^2 + (y(t)-ty)^2 である.
        // f(t)を整理すると以下のようになる
        // f(t) = t^4/4 + bt^3 + ct^2 + 2dt + dx^2 + dy^2
        let dx = gs.px - tx;
        let dy = gs.py - ty;
        let b = gs.vx * ax + gs.vy * ay;
        let c = gs.vx * gs.vx + gs.vy * gs.vy + dx * ax + dy * ay;
        let d = dx * gs.vx + dy * gs.vy;

        // f(t)を微分すると f'(t) = t^3 + 3bt^2 + 2ct + 2d となる.
        // f'(t) = 0 の解を求め, f(t)が最小となるtの候補を求める.
        let candidates = Simulator::solve_cubic_equation(b, c, d);

        // 求めたtの候補を小さい順に試し, 最初に条件を満たしたものを返却する.
        for (i, &t) in candidates.iter().enumerate() {
            // f(candidates[1])は極大値なので無視する
            if i != 1 && t0 < t && t < 1.0 && g(t) {
                return t;
            }
        }

        // t=1.0が条件を満たしているケースを考慮する
        if g(1.0) {
            return 1.0
        }

        -1.0
    }

    /// 時刻 t=0 から t=1 までの間に訪れる目的地の数を求める
    fn calc(&self, gs: &GameState, ax: f64, ay: f64) -> (i32, i32, Vec<bool>) {
        let mut res = 0;
        let mut idx = gs.target_idx;
        let mut used = gs.used_optional.clone();

        // 必須目的地への到達判定
        let mut t1 = Vec::new();
        let mut t0 = 0.0;
        while t0 < 1.0 {
            let (tx, ty) = self.required[idx as usize];
            let tt = self.calc_visit_time(gs, ax, ay, tx, ty, t0);
            if tt < 0.0 {
                break;
            }
            t0 = tt;
            res += 1;
            idx += 1;
            if idx == self.required.len() as i32 {
                idx = 0;
                t1.push(t0);
            }
        }

        // 任意目的地への到達判定
        for i in 0..self.optional.len() {
            if used[i] && t1.is_empty() {
                continue;
            }
            let (tx, ty) = self.optional[i];
            t0 = 0.0;
            let mut j = 0;
            while t0 < 1.0 {
                let tt = self.calc_visit_time(gs, ax, ay, tx, ty, t0);
                if tt < 0.0 {
                    break;
                }
                t0 = tt;
                // 最後の必須目的地に到達した直後にこの任意目的地に到達したケースを考慮
                while j < t1.len() && t1[j] < t0 {
                    j += 1;
                    used[i] = false;
                }
                if !used[i] {
                    res += 1;
                    used[i] = true;
                }
            }
            // 最後の必須目的地に到達した際に任意目的地が再度利用可能になる
            if j < t1.len() {
                used[i] = false;
            }
        }

        (res, idx, used)
    }

    /// 操作θを与え単位時間だけ状態を更新する
    fn update(&self, gs: &GameState, th: f64) -> GameState {
        // 加速度を決定する
        let ax = th.cos();
        let ay = th.sin();
        // 訪れる目的地の数を求める
        let (s, idx, used) = self.calc(gs, ax, ay);
        // 移動後の状態を返却する
        GameState {
            px: gs.px + gs.vx + 0.5 * ax,
            py: gs.py + gs.vy + 0.5 * ay,
            vx: gs.vx + ax,
            vy: gs.vy + ay,
            score: gs.score + s,
            target_idx: idx,
            used_optional: used,
        }
    }
}

struct Program {
    game_server: String,
    token: String,
    rng: ThreadRng,
    client: reqwest::blocking::Client,
    state: StateResponse,
}

impl Program {
    /// ゲームサーバのAPIを呼び出す
    fn call_api<T: Serialize>(
        &self,
        action: &String,
        post_data: Option<&T>,
    ) -> Result<reqwest::blocking::Response> {
        let url = format!("{}{}", self.game_server, action);

        // 5xxエラーまたはErrの際は100ms空けて5回までリトライする
        let sleep_time = std::time::Duration::from_millis(100);
        for _ in 0..5 {
            println!("{}", url);

            let request = match post_data {
                Some(data) => self.client.post(&url).json(data),
                None => self.client.get(&url),
            };

            match request.send() {
                Ok(response) => {
                    if response.status() == reqwest::StatusCode::OK {
                        return Ok(response);
                    }
                    if response.status().is_server_error() {
                        println!("{}", response.status());
                        std::thread::sleep(sleep_time);
                        continue;
                    }
                    println!("{}", response.status());
                    Err(ApiError::ApiError())?
                }
                Err(e) => {
                    println!("Error: {}", e);
                    std::thread::sleep(sleep_time);
                    continue;
                }
            }
        }

        Err(ApiError::ApiError())?
    }

    /// /api/stateを呼び出す
    fn call_state(&mut self) -> Result<()> {
        let action = format!("/api/state/{}", self.token);
        let game_state: StateResponse = self.call_api::<()>(&action, None)?.json()?;
        self.state = game_state;
        Ok(())
    }

    /// /api/optionalを呼び出す 任意目的地を追加する
    fn call_optional(&self, x: f64, y: f64) -> Result<()> {
        let action = format!("/api/optional/{}/{}", self.token, self.state.game_id);
        let point = Point {
            x: format!("{:.16}", x),
            y: format!("{:.16}", y),
        };
        self.call_api(&action, Some(&point))?;
        Ok(())
    }

    /// /api/plan を呼び出す 移動予定を決定する
    fn call_plan(&self, plan: &[f64]) -> Result<()> {
        let action = format!("/api/plan/{}/{}", self.token, self.state.game_id);
        let plan_repr: Vec<String> = plan.iter().map(|&x| format!("{:.16}", x)).collect();
        self.call_api(&action, Some(&plan_repr))?;
        Ok(())
    }

    fn call_eval(&self, plan: &[f64]) -> Result<EvalResponse> {
        let action = format!("/api/eval/{}", self.token);
        let res: EvalResponse = self.call_api(&action, Some(&EvalRequest{
            checkpoint_size: self.state.checkpoint_size.clone(),
            required: self.state.required.clone(),
            optional: self.state.optional.clone(),
            plan: plan.iter().map(|&x| format!("{:.16}", x)).collect(),
        }))?.json()?;
        Ok(res)
    }

    fn think_optional(&mut self) -> (f64, f64) {
        loop {
            // -5.0 から 5.0 の範囲でランダムに生成する
            let x = self.rng.gen_range(-5.0..5.0);
            let y = self.rng.gen_range(-5.0..5.0);
            if f64::hypot(x, y) >= 1.0 {
                return (x, y)
            }
        }
    }

    fn think_plan(&self) -> Vec<f64> {
        let mut plan = Vec::new();
        let mut gs = GameState::new(self.state.optional.len());
        let sim = Simulator::new(&self.state);
        let mut rng = thread_rng();

        // 状態の評価を行う
        let evaluate = |gs: &GameState| -> (i32, f64) {
            let (tx, ty) = sim.required[gs.target_idx as usize];
            (gs.score, f64::hypot(gs.px - tx, gs.py - ty))
        };

        for _ in 0..self.state.plan_length {
            let mut best_th = 0.0;
            let mut best_s = -1;
            let mut best_d = 1e+300;
            for _ in 0..10 {
                // 0~2πの範囲でthをランダムに選び
                // 現在のベストよりもスコアが増加しているか, 次の必須目的地に近づくなら採用する
                let th = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                let (s, d) = evaluate(&sim.update(&gs, th));
                if s > best_s || (s == best_s && d < best_d) {
                    best_th = th;
                    best_s = s;
                    best_d = d;
                }
            }
            plan.push(best_th);
            gs = sim.update(&gs, best_th);
        }
        plan
    }

    fn solve(&mut self) -> Result<()> {
        let mut game_id = 0;
        loop {
            loop {
                // game_idの更新を待つ
                self.call_state()?;
                if game_id != self.state.game_id {
                    game_id = self.state.game_id;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }

            // 任意目的地の提出が可能な場合に提出する
            if self.state.can_submit_optional {
                let (x, y) = self.think_optional();
                self.call_optional(x, y)?;
            }

            // ゲーム開始を待つ
            while !self.state.is_set_optional {
                self.call_state()?;
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }

            // planを考えて提出する
            if self.state.plan_time > 0 {
                let plan: Vec<f64> = self.think_plan();
                self.call_plan(&plan)?;
            }
        }
    }

    fn test(&mut self, file_path: String) -> Result<()> {
        self.state = get_local_state(file_path)?;
        let plan = self.think_plan();
        let res = self.call_eval(&plan)?;
        let mut plan_file = std::fs::File::create("./eval_plan.json")?;
        let plan_str = serde_json::to_string(&(plan.iter().map(|&x| format!("{:.16}", x)).collect::<Vec<String>>()))?;
        writeln!(plan_file, "{}", plan_str)?;
        println!("score = {}", res.result.last().unwrap().score);
        Ok(())
    }
}

fn get_local_state(file_path: String) -> Result<StateResponse> {
    let file = std::fs::File::open(file_path)?;
    let reader = std::io::BufReader::new(file);
    let state: StateResponse = serde_json::from_reader(reader)?;
    Ok(state)
}

fn main() {
    let mut program = Program {
        game_server: std::env::var("GAME_SERVER")
            .unwrap_or("https://2024.gcp.tenka1.klab.jp".to_string()),
        token: std::env::var("TOKEN").unwrap_or("YOUR_TOKEN".to_string()),
        rng: thread_rng(),
        client: reqwest::blocking::Client::new(),
        state: Default::default(),
    };
    if std::env::args().len() >= 2 {
        match program.test(std::env::args().nth(1).unwrap()) {
            Ok(_) => println!("Finish"),
            Err(e) => panic!("{}", e),
        }
    } else {
        match program.solve() {
            Ok(_) => println!("Finish"),
            Err(e) => panic!("{}", e),
        }
    }
}
