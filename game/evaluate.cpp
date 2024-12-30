#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

vector<double> solve_cubic_equation(double b, double c, double d) {
    // t^3 + 3bt^2 + 2ct + 2d = 0 を解く
    // t = y - b と置く
    // y^3 + 3hy - 2q = 0
    auto q = c * b - b * b * b - d;
    auto h = c * 2 / 3 - b * b;
    auto r = q * q + h * h * h;
    auto y = r >= 0 ?
             (cbrt(q + sqrt(r)) + cbrt(q - sqrt(r))) :
             (cbrt(hypot(q, sqrt(-r))) * cos(atan2(sqrt(-r), q) / 3) * 2);
    auto t1 = y - b;
    // t^3 + 3bt^2 + 2ct + 2d = (t - t1)(t^2 + st + u)
    auto s = 3 * b + t1;
    auto u = 2 * c + t1 * s;
    // t^2 + st + u = 0 を解く
    auto z = s * s - 4 * u;
    if (z >= 0) {
        vector<double> tt {t1, (-s + sqrt(z)) / 2, (-s - sqrt(z)) / 2};
        sort(tt.begin(), tt.end());
        return tt;
    } else {
        return {t1};
    }
}

struct CalcScore {
    double checkpoint_size;
    const vector<pair<double, double>>& required;
    const vector<pair<double, double>>& optional;

    double px = 0;
    double py = 0;
    double vx = 0;
    double vy = 0;
    double ax = 0;
    double ay = 0;

    int score = 0;
    unsigned targetIdx = 0;
    vector<bool> usedOptional;

    CalcScore(double checkpoint_size, const vector<pair<double, double>>& required, const vector<pair<double, double>>& optional) :
            checkpoint_size(checkpoint_size), required(required), optional(optional), usedOptional(optional.size(), false) {}

    [[nodiscard]] double calc_visit_time(double t0, double tx, double ty) const {
        auto g = [&](double t) {
            auto x = px + vx * t + 0.5 * ax * t * t - tx;
            auto y = py + vy * t + 0.5 * ay * t * t - ty;
            return hypot(x, y) <= checkpoint_size;
        };

        auto dx = px - tx;
        auto dy = py - ty;
        // f(t) = t^4/4 + bt^3 + ct^2 + 2dt + e
        auto b = vx * ax + vy * ay;
        auto c = vx * vx + vy * vy + dx * ax + dy * ay;
        auto d = dx * vx + dy * vy;

        // f'(t) = t^3 + 3bt^2 + 2ct + 2d = 0 を解く
        auto t = solve_cubic_equation(b, c, d);
        if (t.size() == 3) {
            auto t2 = t[0];
            auto t3 = t[2];
            if (t0 < t2 && t2 < 1.0 && g(t2)) {
                return t2;
            }
            if (t0 < t3 && t3 < 1.0 && g(t3)) {
                return t3;
            }
        } else if (t.size() == 1) {
            auto t1 = t[0];
            if (t0 < t1 && t1 < 1.0 && g(t1)) {
                return t1;
            }
        } else {
            throw std::exception();
        }
        if (g(1.0)) {
            return 1.0;
        }
        return -1.0;
    }

    pair<int, unsigned> calc(unsigned idx, vector<bool>& used) const {
        int res = 0;
        vector<double> t1;
        {
            double t0 = 0;
            while (t0 < 1.0) {
                auto tx = required[idx].first;
                auto ty = required[idx].second;
                auto tt = calc_visit_time(t0, tx, ty);
                if (tt < 0) break;
                t0 = tt;
                ++ res;
                if (++ idx == required.size()) {
                    idx = 0;
                    t1.push_back(t0);
                }
            }
        }
        for (unsigned i = 0; i < optional.size(); ++ i) {
            if (used[i] && t1.empty()) continue;
            auto tx = optional[i].first;
            auto ty = optional[i].second;
            double t0 = 0;
            unsigned j = 0;
            while (t0 < 1.0) {
                auto tt = calc_visit_time(t0, tx, ty);
                if (tt < 0) break;
                t0 = tt;
                while (j < t1.size() && t1[j] < t0) {
                    ++ j;
                    used[i] = false;
                }
                if (!used[i]) {
                    ++ res;
                    used[i] = true;
                }
            }
            if (j < t1.size()) {
                used[i] = false;
            }
        }
        return {res, idx};
    }

    void update(double th) {
        ax = cos(th);
        ay = sin(th);
        auto r = calc(targetIdx, usedOptional);
        score += r.first;
        targetIdx = r.second;
        px = px + vx + 0.5 * ax;
        py = py + vy + 0.5 * ay;
        vx = vx + ax;
        vy = vy + ay;
        cout << setprecision(16) << px << " " << py << " " << vx << " " << vy << " " << score << endl;
    }

    void calc_score(const vector<double>& plan) {
        for (auto th : plan) {
            update(th);
        }
    }
};

int main() {
    double checkpoint_size;
    int num_required, num_optional, plan_length;
    cin >> checkpoint_size >> num_required >> num_optional >> plan_length;

    vector<pair<double, double>> required(num_required);
    for (auto& x : required) cin >> x.first >> x.second;

    vector<pair<double, double>> optional(num_optional);
    for (auto& x : optional) cin >> x.first >> x.second;

    vector<double> plan(plan_length);
    for (auto& x : plan) cin >> x;

    CalcScore s(checkpoint_size, required, optional);
    s.calc_score(plan);
}
