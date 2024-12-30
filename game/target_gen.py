from math import hypot, atan2, sqrt
import random


def f(a11, a12, a21, a22) -> tuple[float, float]:
    r = 10 - (1 + sqrt(2))
    while True:
        x = random.random() * r
        y = random.random() * r
        if x < y:
            x, y = y, x
        x += 1 + sqrt(2)
        y += 1
        if 5 <= hypot(x, y) < 10:
            return a11 * x + a12 * y, a21 * x + a22 * y


def target_gen() -> list[dict[str, str]]:
    target = [
        f(-1, 0, 0, -1),
        f(0, -1, -1, 0),
        f(0, 1, -1, 0),
        f(1, 0, 0, -1),
        f(1, 0, 0, 1),
        f(0, 1, 1, 0),
        f(0, -1, 1, 0),
        f(-1, 0, 0, 1),
    ]
    while True:
        x = random.random() * 20 - 10
        y = random.random() * 20 - 10
        if hypot(x, y) < 3 or abs(x) >= 10 or abs(y) >= 10:
            continue
        ok = True
        for xx, yy in target:
            if hypot(x - xx, y - yy) < 2:
                ok = False
                break
        if ok:
            target.append((x, y))
            break
    target.sort(key=lambda p: atan2(p[1], p[0]))
    while True:
        x = random.random() - 0.5
        y = random.random() - 0.5
        if hypot(x, y) < 0.5:
            target.append((x, y))
            break
    return [{'x': str(x), 'y': str(y)} for x, y in target]
