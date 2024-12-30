package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"time"
)

var (
	// ゲームサーバのアドレス / トークン
	GameServer = getEnv("GAME_SERVER", "https://2024.gcp.tenka1.klab.jp")
	Token      = getEnv("TOKEN", "YOUR_TOKEN")
	client     = &http.Client{}
)

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

type Point struct {
	X string `json:"x"`
	Y string `json:"y"`
}

type StateResponse struct {
	GameID            int     `json:"game_id"`
	OptionalTime      int     `json:"optional_time"`
	PlanTime          int     `json:"plan_time"`
	CanSubmitOptional bool    `json:"can_submit_optional"`
	IsSetOptional     bool    `json:"is_set_optional"`
	Required          []Point `json:"required"`
	Optional          []Point `json:"optional"`
	CheckpointSize    string  `json:"checkpoint_size"`
	PlanLength        int     `json:"plan_length"`
}

type EvalRequest struct {
	CheckpointSize string   `json:"checkpoint_size"`
	Required       []Point  `json:"required"`
	Optional       []Point  `json:"optional"`
	Plan           []string `json:"plan"`
}

type EvalElement struct {
	P     Point `json:"p"`
	V     Point `json:"v"`
	Score int   `json:"score"`
}

type EvalResponse struct {
	Result []EvalElement `json:"result"`
}

type GameState struct {
	px           float64
	py           float64
	vx           float64
	vy           float64
	score        int
	targetIdx    int
	usedOptional []bool
}

func NewGameState(optionalLen int) *GameState {
	return &GameState{
		usedOptional: make([]bool, optionalLen),
	}
}

type Simulator struct {
	checkpointSize float64
	required       [][2]float64
	optional       [][2]float64
}

func NewSimulator(state *StateResponse) *Simulator {
	sim := new(Simulator)

	size, err := strconv.ParseFloat(state.CheckpointSize, 64)
	if err != nil {
		log.Fatal(err)
	}
	sim.checkpointSize = size

	sim.required = make([][2]float64, len(state.Required))
	for i := 0; i < len(state.Required); i++ {
		x, err := strconv.ParseFloat(state.Required[i].X, 64)
		if err != nil {
			log.Fatal(err)
		}
		y, err := strconv.ParseFloat(state.Required[i].Y, 64)
		if err != nil {
			log.Fatal(err)
		}
		sim.required[i] = [2]float64{x, y}
	}

	sim.optional = make([][2]float64, len(state.Optional))
	for i := 0; i < len(state.Optional); i++ {
		x, err := strconv.ParseFloat(state.Optional[i].X, 64)
		if err != nil {
			log.Fatal(err)
		}
		y, err := strconv.ParseFloat(state.Optional[i].Y, 64)
		if err != nil {
			log.Fatal(err)
		}
		sim.optional[i] = [2]float64{x, y}
	}

	return sim
}

// t^3 + 3bt^2 + 2ct + 2d = 0 を解き昇順ソート済の実数解を返却する
func solveCubicEquation(b, c, d float64) []float64 {
	// カルダノの方法を用いて解く
	// 参考: https://ja.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F
	// t = y - b と置く
	// y^3 + 3hy - 2q = 0
	q := c*b - b*b*b - d
	h := 2*c/3 - b*b
	r := q*q + h*h*h

	var y float64
	if r >= 0 {
		y = math.Cbrt(q+math.Sqrt(r)) + math.Cbrt(q-math.Sqrt(r))
	} else {
		// z = q + sqrt(-r) i とすると、
		// zの絶対値は hypot(q, sqrt(-r))
		// zの偏角は atan2(sqrt(-r), q)
		// であり、yは以下の式で求まる
		y = math.Cbrt(math.Hypot(q, math.Sqrt(-r))) * math.Cos(math.Atan2(math.Sqrt(-r), q)/3) * 2
	}

	t1 := y - b
	// t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
	s := 3*b + t1
	u := 2*c + t1*s
	// t^2 + st + u = 0 を解く
	z := s*s - 4*u
	if z >= 0 {
		result := []float64{t1, (-s + math.Sqrt(z)) / 2, (-s - math.Sqrt(z)) / 2}
		sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
		return result
	} else {
		return []float64{t1}
	}
}

// 目的地の座標(tx, ty)を訪れる時刻t(t0 < t <= 1.0)を求める
// 目的地に到達しないなら-1を返却する
func (sim *Simulator) calcVisitTime(gs *GameState, ax, ay, tx, ty, t0 float64) float64 {
	g := func(t float64) bool {
		// 時刻tで質点が目的地に到達しているかを判定する
		x := gs.px + gs.vx*t + 0.5*ax*t*t - tx
		y := gs.py + gs.vy*t + 0.5*ay*t*t - ty
		return math.Hypot(x, y) <= sim.checkpointSize
	}

	// 質点の座標を(px, py), 速度を(vx, vy), 加速度を(ax, ay) とし, 目的地の座標を(tx, ty)とする.
	// 時刻tにおける質点の座標は x(t) = px + vx t + ax t^2/2, y(t) = py + vy t + ax t^2/2 である.
	// 時刻tにおける質点の座標と目的地の座標との距離の2乗をf(t)とする. f(t) = (x(t)-tx)^2 + (y(t)-ty)^2 である.
	// f(t)を整理すると以下のようになる
	// f(t) = t^4/4 + bt^3 + ct^2 + 2dt + dx^2 + dy^2
	dx := gs.px - tx
	dy := gs.py - ty
	b := gs.vx*ax + gs.vy*ay
	c := gs.vx*gs.vx + gs.vy*gs.vy + dx*ax + dy*ay
	d := dx*gs.vx + dy*gs.vy

	// f(t)を微分すると f'(t) = t^3 + 3bt^2 + 2ct + 2d となる.
	// f'(t) = 0 の解を求め, f(t)が最小となるtの候補を求める.
	candidates := solveCubicEquation(b, c, d)

	// 求めたtの候補を小さい順に試し, 最初に条件を満たしたものを返却する.
	for i := 0; i < len(candidates); i++ {
		t := candidates[i]
		// f(candidates[1])は極大値なので無視する
		if i != 1 && t0 < t && t < 1.0 && g(t) {
			return t
		}
	}

	// t=1.0が条件を満たしているケースを考慮する
	if g(1.0) {
		return 1.0
	}
	return -1.0
}

// 時刻t=0からt=1までの間に訪れる目的地の数を求める
func (sim *Simulator) calc(gs *GameState, ax, ay float64) (int, int, []bool) {
	res := 0
	idx := gs.targetIdx
	used := make([]bool, len(gs.usedOptional))
	copy(used, gs.usedOptional)

	// 必須目的地への到達判定
	var t1 []float64
	t0 := 0.0
	for t0 < 1.0 {
		tx, ty := sim.required[idx][0], sim.required[idx][1]
		tt := sim.calcVisitTime(gs, ax, ay, tx, ty, t0)
		if tt < 0 {
			break
		}
		t0 = tt
		res += 1
		idx += 1
		if idx == len(sim.required) {
			idx = 0
			t1 = append(t1, t0)
		}
	}

	// 任意目的地への到達判定
	for i := 0; i < len(sim.optional); i++ {
		if used[i] && len(t1) == 0 {
			continue
		}
		tx, ty := sim.optional[i][0], sim.optional[i][1]
		t0 = 0.0
		j := 0
		for t0 < 1.0 {
			tt := sim.calcVisitTime(gs, ax, ay, tx, ty, t0)
			if tt < 0 {
				break
			}
			t0 = tt
			// 最後の必須目的地に到達した直後にこの任意目的地に到達したケースを考慮
			for j < len(t1) && t1[j] < t0 {
				j += 1
				used[i] = false
			}
			if !used[i] {
				res += 1
				used[i] = true
			}
		}
		// 最後の必須目的地に到達した際に任意目的地が再度利用可能になる
		if j < len(t1) {
			used[i] = false
		}
	}

	return res, idx, used
}

// 操作θを与え単位時間だけ状態を更新する
func (sim *Simulator) Update(gs *GameState, th float64) *GameState {
	// 加速度を決定する
	ax := math.Cos(th)
	ay := math.Sin(th)

	// 訪れる目的地の数を求める
	s, idx, used := sim.calc(gs, ax, ay)

	// 移動後の状態に更新する
	return &GameState{
		px:           gs.px + gs.vx + 0.5*ax,
		py:           gs.py + gs.vy + 0.5*ay,
		vx:           gs.vx + ax,
		vy:           gs.vy + ay,
		score:        gs.score + s,
		targetIdx:    idx,
		usedOptional: used,
	}
}

// ゲームサーバのAPIを呼び出す
func callAPI(x string, postData interface{}) ([]byte, error) {
	url := GameServer + x

	var jsonData []byte
	if postData != nil {
		b, err := json.Marshal(postData)
		if err != nil {
			return nil, err
		}
		jsonData = b
	}

	// err != nilの場合 または 5xxエラーの際は100ms空けて5回までリトライする
	for i := 0; i < 5; i++ {
		fmt.Println(url)

		var resp *http.Response
		var err error
		if postData == nil {
			resp, err = client.Get(url)
		} else {
			resp, err = client.Post(url, "application/json", bytes.NewBuffer(jsonData))
		}

		if err != nil {
			log.Printf("%v", err.Error())
			time.Sleep(time.Millisecond * 100)
			continue
		}
		//goland:noinspection GoUnhandledErrorResult
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if resp.StatusCode == 200 {
			return body, nil
		}
		if 500 <= resp.StatusCode && resp.StatusCode < 600 {
			fmt.Println(resp.Status)
			time.Sleep(time.Millisecond * 100)
			continue
		}

		return nil, fmt.Errorf("api Error status_code:%d", resp.StatusCode)
	}
	return nil, fmt.Errorf("api Error")
}

type Program struct {
	state StateResponse
}

// /api/stateを呼び出す
func (p *Program) callState() {
	res, err := callAPI(fmt.Sprintf("/api/state/%s", Token), nil)
	if err != nil {
		log.Print(err)
		return
	}

	err = json.Unmarshal(res, &p.state)
	if err != nil {
		log.Fatal(err)
	}
}

// /api/optionalを呼び出す 任意目的地を追加する
func (p *Program) callOptional(x float64, y float64) {
	point := Point{
		X: strconv.FormatFloat(x, 'g', -1, 64),
		Y: strconv.FormatFloat(y, 'g', -1, 64),
	}

	_, err := callAPI(fmt.Sprintf("/api/optional/%s/%d", Token, p.state.GameID), point)
	if err != nil {
		log.Println("callOptional", err)
	}
}

// /api/plan を呼び出す 移動予定を決定する
func (p *Program) callPlan(plan []float64) {
	var planString []string
	for _, v := range plan {
		planString = append(planString, strconv.FormatFloat(v, 'g', -1, 64))
	}

	_, err := callAPI(fmt.Sprintf("/api/plan/%s/%d", Token, p.state.GameID), planString)
	if err != nil {
		log.Println("callPlan", err)
	}
}

func (p *Program) callEval(plan []float64) (*EvalResponse, error) {
	var planString []string
	for _, v := range plan {
		planString = append(planString, strconv.FormatFloat(v, 'g', -1, 64))
	}
	res, err := callAPI(fmt.Sprintf("/api/eval/%s", Token), &EvalRequest{
		CheckpointSize: p.state.CheckpointSize,
		Required:       p.state.Required,
		Optional:       p.state.Optional,
		Plan:           planString,
	})
	if err != nil {
		return nil, err
	}
	var v EvalResponse
	err = json.Unmarshal(res, &v)
	if err != nil {
		return nil, err
	}
	return &v, nil
}

func (p *Program) thinkOptional() (float64, float64) {
	for {
		x := (rand.Float64() - 0.5) * 10
		y := (rand.Float64() - 0.5) * 10
		if math.Hypot(x, y) >= 1.0 {
			return x, y
		}
	}
}

func (p *Program) thinkPlan() []float64 {
	var plan []float64
	gs := NewGameState(len(p.state.Optional))
	sim := NewSimulator(&p.state)

	// 状態の評価を行う
	evaluate := func(gs *GameState) (int, float64) {
		tx, ty := sim.required[gs.targetIdx][0], sim.required[gs.targetIdx][1]
		return gs.score, math.Hypot(gs.px-tx, gs.py-ty)
	}

	for i := 0; i < p.state.PlanLength; i++ {
		var (
			bestTh = 0.0
			bestS  = -1
			bestD  = 1e300
		)
		for j := 0; j < 10; j++ {
			// 0~2πの範囲でthをランダムに選び
			// 現在のベストよりもスコアが増加しているか, 次の必須目的地に近づくなら採用する
			th := rand.Float64() * 2 * math.Pi
			s, d := evaluate(sim.Update(gs, th))
			if s > bestS || (s == bestS && d < bestD) {
				bestTh = th
				bestS = s
				bestD = d
			}
		}

		plan = append(plan, bestTh)
		gs = sim.Update(gs, bestTh)
	}

	return plan
}

func (p *Program) Solve() {
	gameID := -1
	for {
		// game_idの更新を待つ
		for {
			p.callState()
			if gameID != p.state.GameID {
				gameID = p.state.GameID
				break
			}
			time.Sleep(time.Second)
		}

		// 任意目的地の提出が可能な場合に提出する
		if p.state.CanSubmitOptional {
			x, y := p.thinkOptional()
			p.callOptional(x, y)
		}

		// ゲーム開始を待つ
		for !p.state.IsSetOptional {
			time.Sleep(time.Second)
			p.callState()
		}

		// planを考えて提出する
		if p.state.PlanTime > 0 {
			plan := p.thinkPlan()
			p.callPlan(plan)
		}
	}
}

func (p *Program) Test(filePath string) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	err = json.Unmarshal(data, &p.state)
	if err != nil {
		log.Fatal(err)
	}

	plan := p.thinkPlan()
	res, err := p.callEval(plan)
	if err != nil {
		log.Fatal(err)
	}

	var planStrList []string
	for _, v := range plan {
		planStrList = append(planStrList, strconv.FormatFloat(v, 'g', -1, 64))
	}
	f, err := os.Create("./eval_plan.json")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	planStr, err := json.Marshal(planStrList)
	_, err = f.Write(planStr)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("score =", res.Result[len(res.Result)-1].Score)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) >= 2 {
		new(Program).Test(os.Args[1])
	} else {
		new(Program).Solve()
	}
}
