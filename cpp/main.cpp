#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std::literals;

struct memory {
    char *response;
    size_t size;
};

static size_t curl_cb(void *data, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    auto *mem = (memory *)userp;

    char *ptr = (char *)realloc(mem->response, mem->size + realsize + 1);
    if (ptr == nullptr)
        return 0;

    mem->response = ptr;
    memcpy(&(mem->response[mem->size]), data, realsize);
    mem->size += realsize;
    mem->response[mem->size] = 0;

    return realsize;
}

// ゲームサーバのAPIを叩く
json call_api(const std::string &url, const json *post_data = nullptr) {
    for (int i = 0; i < 5; ++i) {
        std::cout << url << std::endl;
        CURL *curl = curl_easy_init();
        if (curl == nullptr) {
            throw std::runtime_error("curl_easy_init failure");
        }

        memory chunk = {};
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        curl_slist *headers = nullptr;
        if (post_data != nullptr) {
            auto data = post_data->dump();
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());
            curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, data.c_str());
        }

        const auto res = curl_easy_perform(curl);
        curl_slist_free_all(headers);
        if (res != CURLE_OK) {
            std::cerr << "curl error " << res << std::endl;
            std::this_thread::sleep_for(100ms);
            continue;
        }
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_easy_cleanup(curl);
        if (response_code != 200) {
            std::cerr << "http error " << response_code << std::endl;
            if (response_code >= 500) {
                std::this_thread::sleep_for(100ms);
                continue;
            }
            throw std::runtime_error("invalid response");
        }
        return json::parse(chunk.response);
    }
    throw std::runtime_error("call_api failed");
}

struct Point {
    std::string x;
    std::string y;
};

void from_json(const json &j, Point &p) {
    j.at("x").get_to(p.x);
    j.at("y").get_to(p.y);
}

void to_json(json &j, const Point &p) {
    j = json{{"x", p.x}, {"y", p.y}};
}

struct StateResponse {
    int game_id{};
    int optional_time{};
    int plan_time{};
    bool can_submit_optional{};
    bool is_set_optional{};
    std::vector<Point> required;
    std::vector<Point> optional;
    std::string checkpoint_size;
    int plan_length{};
};

void from_json(const json &j, StateResponse &s) {
    j.at("game_id").get_to(s.game_id);
    j.at("optional_time").get_to(s.optional_time);
    j.at("plan_time").get_to(s.plan_time);
    j.at("can_submit_optional").get_to(s.can_submit_optional);
    j.at("is_set_optional").get_to(s.is_set_optional);
    j.at("required").get_to(s.required);
    j.at("optional").get_to(s.optional);
    j.at("checkpoint_size").get_to(s.checkpoint_size);
    j.at("plan_length").get_to(s.plan_length);
}

void to_json(json &j, const StateResponse &s) {
    j = json{
        {"game_id", s.game_id},
        {"optional_time", s.optional_time},
        {"plan_time", s.plan_time},
        {"can_submit_optional", s.can_submit_optional},
        {"is_set_optional", s.is_set_optional},
        {"required", s.required},
        {"optional", s.optional},
        {"checkpoint_size", s.checkpoint_size},
        {"plan_length", s.plan_length},
    };
}

struct EvalElement {
    Point p;
    Point v;
    int score{};
};

void from_json(const json &j, EvalElement &e) {
    j.at("p").get_to(e.p);
    j.at("v").get_to(e.v);
    j.at("score").get_to(e.score);
}

void to_json(json &j, const EvalElement &e) {
    j = json{
        {"p", e.p},
        {"v", e.v},
        {"score", e.score},
    };
}

struct EvalResponse {
    std::vector<EvalElement> result;
};

void from_json(const json &j, EvalResponse &e) {
    j.at("result").get_to(e.result);
}

void to_json(json &j, const EvalResponse &e) {
    j = json{
        {"result", e.result},
    };
}

std::string to_str(double x) {
    char buf[100];
    sprintf(buf, "%.16g", x);
    return std::string(buf);
}

std::vector<std::string> to_vecstr(const std::vector<double>& a) {
    std::vector<std::string> pp;
    for (const auto &p : a) {
        pp.emplace_back(to_str(p));
    }
    return pp;
}

struct GameState {
    double px{}, py{}, vx{}, vy{};
    int score{};
    int target_idx{};
    std::vector<bool> used_optional;

    void init(int optional_len) { used_optional.resize(optional_len, false); }
};

class Simulator {
  public:
    double checkpoint_size;
    std::vector<std::pair<double, double>> required;
    std::vector<std::pair<double, double>> optional;

    Simulator(const std::string &checkpoint_size, const std::vector<Point> &required,
              const std::vector<Point> &optional) {
        this->checkpoint_size = std::stod(checkpoint_size);
        for (const auto &point : required) {
            this->required.emplace_back(std::stod(point.x), std::stod(point.y));
        }
        for (const auto &point : optional) {
            this->optional.emplace_back(std::stod(point.x), std::stod(point.y));
        }
    }

    /// t^3 + 3bt^2 + 2ct + 2d = 0 を解き昇順ソート済の実数解を返却する
    static std::vector<double> solve_cubic_equation(double b, double c, double d) {
        // カルダノの方法を用いて解く
        // 参考:
        // https://ja.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F
        // t = y - b と置く
        // y^3 + 3hy - 2q = 0
        const auto q = c * b - b * b * b - d;
        const auto h = c * 2.0 / 3.0 - b * b;
        const auto r = q * q + h * h * h;
        const auto y =
            r >= 0.0 ? std::cbrt(q + std::sqrt(r)) + std::cbrt(q - std::sqrt(r))
                     // z = q + sqrt(-r) i とすると、
                     // zの絶対値は hypot(q, sqrt(-r))
                     // zの偏角は atan2(sqrt(-r), q)
                     // であり、yは以下の式で求まる
                     : std::cbrt(std::hypot(q, std::sqrt(-r))) * std::cos(std::atan2(std::sqrt(-r), q) / 3.0) * 2.0;

        const auto t1 = y - b;

        // t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
        const auto s = 3.0 * b + t1;
        const auto u = 2.0 * c + t1 * s;

        // t^2 + st + u = 0 を解く
        const auto z = s * s - 4.0 * u;
        if (z >= 0.0) {
            std::vector v{t1, (-s + std::sqrt(z)) / 2, (-s - std::sqrt(z)) / 2};
            std::sort(std::begin(v), std::end(v));
            return v;
        }
        return std::vector{t1};
    }

    // 目的地の座標(tx, ty)を訪れる時刻t(t0 < t <= 1.0)を求める
    // 目的地に到達しないなら-1を返却する
    double calc_visit_time(const GameState &gs, double ax, double ay, double tx, double ty, double t0) const {
        // 時刻tで質点が目的地に到達しているかを判定する
        const auto g = [&](double t) -> bool {
            const auto x = gs.px + gs.vx * t + 0.5 * ax * t * t - tx;
            const auto y = gs.py + gs.vy * t + 0.5 * ay * t * t - ty;
            return std::hypot(x, y) <= checkpoint_size;
        };

        // 質点の座標を(px, py), 速度を(vx, vy), 加速度を(ax, ay) とし, 目的地の座標を(tx, ty)とする.
        // 時刻tにおける質点の座標は x(t) = px + vx t + ax t^2/2, y(t) = py + vy t + ax t^2/2 である.
        // 時刻tにおける質点の座標と目的地の座標との距離の2乗をf(t)とする.
        // f(t) = (x(t)-tx)^2 + (y(t)-ty)^2 である.
        // f(t)を整理すると以下のようになる
        // f(t) = t^4/4 + bt^3 + ct^2 + 2dt + dx^2 + dy^2
        const auto dx = gs.px - tx;
        const auto dy = gs.py - ty;
        const auto b = gs.vx * ax + gs.vy * ay;
        const auto c = gs.vx * gs.vx + gs.vy * gs.vy + dx * ax + dy * ay;
        const auto d = dx * gs.vx + dy * gs.vy;

        // f(t)を微分すると f'(t) = t^3 + 3bt^2 + 2ct + 2d となる.
        // f'(t) = 0 の解を求め, f(t)が最小となるtの候補を求める.
        const auto candidates = solve_cubic_equation(b, c, d);

        // 求めたtの候補を小さい順に試し, 最初に条件を満たしたものを返却する.
        for (int i = 0; i < candidates.size(); i++) {
            const auto t = candidates[i];
            // f(candidates[1])は極大値なので無視する
            if (i != 1 && t0 < t && t < 1.0 && g(t)) {
                return t;
            }
        }

        // t=1.0が条件を満たしているケースを考慮する
        return g(1.0) ? 1.0 : -1.0;
    }

    // 時刻 t=0 から t=1 までの間に訪れる目的地の数を求める
    std::tuple<int, int, std::vector<bool>> calc(const GameState &gs, double ax, double ay) const {
        auto res = 0;
        auto idx = gs.target_idx;
        auto used = gs.used_optional;

        // 必須目的地への到達判定
        std::vector<double> t1;
        auto t0 = 0.0;
        while (t0 < 1.0) {
            const auto [tx, ty] = required[idx];
            const auto tt = calc_visit_time(gs, ax, ay, tx, ty, t0);
            if (tt < 0.0) {
                break;
            }
            t0 = tt;
            res += 1;
            idx += 1;
            if (idx == required.size()) {
                idx = 0;
                t1.emplace_back(t0);
            }
        }

        // 任意目的地への到達判定
        for (size_t i = 0; i < optional.size(); i++) {
            if (used[i] && t1.empty()) {
                continue;
            }
            const auto [tx, ty] = optional[i];
            t0 = 0.0;
            size_t j = 0;
            while (t0 < 1.0) {
                const auto tt = calc_visit_time(gs, ax, ay, tx, ty, t0);
                if (tt < 0.0) {
                    break;
                }
                t0 = tt;
                // 最後の必須目的地に到達した直後にこの任意目的地に到達したケースを考慮
                while (j < t1.size() && t1[j] < t0) {
                    j += 1;
                    used[i] = false;
                }
                if (!used[i]) {
                    res += 1;
                    used[i] = true;
                }
            }
            // 最後の必須目的地に到達した際に任意目的地が再度利用可能になる
            if (j < t1.size()) {
                used[i] = false;
            }
        }

        return {res, idx, used};
    }

    // 操作θを与え単位時間だけ状態を更新する
    GameState update(const GameState &gs, double th) const {
        // 加速度を決定する
        const auto ax = std::cos(th);
        const auto ay = std::sin(th);
        // 訪れる目的地の数を求める
        const auto [s, idx, used] = calc(gs, ax, ay);
        // 移動後の状態を返却する
        return GameState{
            gs.px + gs.vx + 0.5 * ax,
            gs.py + gs.vy + 0.5 * ay,
            gs.vx + ax, gs.vy + ay,
            gs.score + s, idx, used,
        };
    }
};

class Program {
  public:
    std::string game_server;
    std::string token;
    std::default_random_engine rng;
    StateResponse state;

    void call_state() {
        const auto res = call_api(game_server + "/api/state/" + token);
        state = res.get<StateResponse>();
    }

    void call_optional(double x, double y) const {
        const Point p{to_str(x), to_str(y)};
        const json j(p);
        call_api(game_server + "/api/optional/" + token + "/" + std::to_string(state.game_id), &j);
    }

    void call_plan(const std::vector<double> &plan) const {
        const json j(to_vecstr(plan));
        call_api(game_server + "/api/plan/" + token + "/" + std::to_string(state.game_id), &j);
    }

    EvalResponse call_eval(const std::vector<double> &plan) const {
        const json j{
            {"checkpoint_size", state.checkpoint_size},
            {"required", state.required},
            {"optional", state.optional},
            {"plan", to_vecstr(plan)},
        };
        const auto res = call_api(game_server + "/api/eval/" + token, &j);
        return res.get<EvalResponse>();
    }

    void solve() {
        int game_id = 0;
        while (true) {
            while (true) {
                // game_idの更新を待つ
                call_state();
                if (game_id != state.game_id) {
                    game_id = state.game_id;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            // 任意目的地の提出が可能な場合に提出する
            if (state.can_submit_optional) {
                auto [x, y] = think_optional();
                call_optional(x, y);
            }

            // ゲーム開始を待つ
            while (!state.is_set_optional) {
                call_state();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            // planを考えて提出する
            if (state.plan_time > 0) {
                auto plan = think_plan();
                call_plan(plan);
            }
        }
    }

    std::pair<double, double> think_optional() {
        std::uniform_real_distribution distribution(-5.0, 5.0);
        for (;;) {
            const double x = distribution(rng);
            const double y = distribution(rng);
            if (std::hypot(x, y) >= 1.0) {
                return {x, y};
            }
        }
    }

    std::vector<double> think_plan() {
        std::vector<double> plan(state.plan_length);
        GameState gs;
        gs.init(state.optional.size());
        const Simulator sim(state.checkpoint_size, state.required, state.optional);

        // 状態の評価を行います
        const auto evaluate = [&](const GameState &gs) -> std::pair<int, double> {
            const auto [tx, ty] = sim.required[gs.target_idx];
            return {gs.score, std::hypot(gs.px - tx, gs.py - ty)};
        };

        std::uniform_real_distribution<> distribution(0, 2 * M_PI);
        for (int i = 0; i < state.plan_length; i++) {
            auto best_th = 0.0;
            auto best_s = -1;
            auto best_d = 1e+300;
            for (int j = 0; j < 10; j++) {
                // 0~2πの範囲でthをランダムに選び
                // 現在のベストよりもスコアが増加しているか,
                // 次の必須目的地に近づくなら採用する
                const auto th = distribution(rng);
                const auto [s, d] = evaluate(sim.update(gs, th));
                if (s > best_s || s == best_s && d < best_d) {
                    best_th = th;
                    best_s = s;
                    best_d = d;
                }
            }
            plan[i] = best_th;
            gs = sim.update(gs, best_th);
        }

        return plan;
    }

    static StateResponse get_local_state(const std::string &file_path) {
        std::ifstream ifs(file_path);
        auto j = json::parse(ifs);
        return j.get<StateResponse>();
    }

    void test(const char *file_path) {
        state = get_local_state(file_path);
        const auto plan = think_plan();
        const auto res = call_eval(plan);

        std::ofstream ofs("./eval_plan.json");
        ofs << json(to_vecstr(plan)).dump() << std::flush;

        std::cout << res.result.back().score << std::endl;
    }
};

std::string getenv_or_default(std::string const &name, std::string const &def) {
    auto v = std::getenv(name.c_str());
    return v != nullptr ? v : def;
}

int main(int argc, char *argv[]) {
    Program program;
    std::random_device seed_gen;
    program.rng = std::default_random_engine(seed_gen());
    program.game_server = getenv_or_default("GAME_SERVER", "https://2024.gcp.tenka1.klab.jp");
    program.token = getenv_or_default("TOKEN", "YOUR_TOKEN");

    if (argc >= 2) {
        program.test(argv[1]);
    } else {
        program.solve();
    }
    return 0;
}
