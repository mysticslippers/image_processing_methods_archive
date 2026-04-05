import math
import random
from typing import List, Tuple


A = 2.0
B = 5.0
N_VALUES = [100, 1000, 10000, 100000]
STRAT_STEPS = [1.0, 0.5]
RUSSIAN_R_VALUES = [0.5, 0.75, 0.95]
SEED = 42


def f(x: float) -> float:
    return x * x


def true_integral(a: float, b: float) -> float:
    return (b ** 3 - a ** 3) / 3.0


I_TRUE = true_integral(A, B)


def print_header(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def print_table(title: str, rows: List[Tuple]) -> None:
    print_header(title)
    print(
        f"{'Параметр':<20}"
        f"{'N':>10}"
        f"{'I_true':>18}"
        f"{'I_MC':>18}"
        f"{'|Ошибка|':>18}"
        f"{'ΔI=I_true/sqrt(N)':>22}"
    )
    print("-" * 110)
    for param, n, i_true, i_mc, err, delta in rows:
        print(
            f"{str(param):<20}"
            f"{n:>10}"
            f"{i_true:>18.8f}"
            f"{i_mc:>18.8f}"
            f"{err:>18.8f}"
            f"{delta:>22.8f}"
        )


def simple_monte_carlo(n: int, rng: random.Random) -> float:
    s = 0.0
    for _ in range(n):
        x = rng.uniform(A, B)
        s += f(x)
    return (B - A) * s / n


def stratified_monte_carlo(n: int, step: float, rng: random.Random) -> float:
    strata = []
    left = A
    while left < B - 1e-12:
        right = min(left + step, B)
        strata.append((left, right))
        left = right

    m = len(strata)

    base = n // m
    rem = n % m
    counts = [base + (1 if i < rem else 0) for i in range(m)]

    estimate = 0.0
    for (l, r), cnt in zip(strata, counts):
        if cnt == 0:
            continue
        local_sum = 0.0
        for _ in range(cnt):
            x = rng.uniform(l, r)
            local_sum += f(x)
        estimate += (r - l) * (local_sum / cnt)

    return estimate


def norm_const_power(k: int, a: float, b: float) -> float:
    return (k + 1) / (b ** (k + 1) - a ** (k + 1))


def pdf_power(x: float, k: int, a: float, b: float) -> float:
    c = norm_const_power(k, a, b)
    return c * (x ** k)


def sample_power_pdf(k: int, a: float, b: float, rng: random.Random) -> float:
    u = rng.random()
    power = k + 1
    return (u * (b ** power - a ** power) + a ** power) ** (1.0 / power)


def importance_sampling(n: int, k: int, rng: random.Random) -> float:
    s = 0.0
    for _ in range(n):
        x = sample_power_pdf(k, A, B, rng)
        p = pdf_power(x, k, A, B)
        s += f(x) / p
    return s / n


def mis_weights_balance(x: float, p1: float, p2: float) -> Tuple[float, float]:
    denom = p1 + p2
    return p1 / denom, p2 / denom


def mis_weights_power2(x: float, p1: float, p2: float) -> Tuple[float, float]:
    p1_2 = p1 * p1
    p2_2 = p2 * p2
    denom = p1_2 + p2_2
    return p1_2 / denom, p2_2 / denom


def multiple_importance_sampling(n: int, weight_mode: str, rng: random.Random) -> float:
    n1 = n // 2
    n2 = n - n1

    if weight_mode == "balance":
        weight_func = mis_weights_balance
    elif weight_mode == "power2":
        weight_func = mis_weights_power2
    else:
        raise ValueError("Неизвестный режим весов")

    part1 = 0.0
    part2 = 0.0

    for _ in range(n1):
        x = sample_power_pdf(1, A, B, rng)
        p1 = pdf_power(x, 1, A, B)
        p2 = pdf_power(x, 3, A, B)
        w1, _ = weight_func(x, p1, p2)
        part1 += w1 * f(x) / p1

    for _ in range(n2):
        x = sample_power_pdf(3, A, B, rng)
        p1 = pdf_power(x, 1, A, B)
        p2 = pdf_power(x, 3, A, B)
        _, w2 = weight_func(x, p1, p2)
        part2 += w2 * f(x) / p2

    result = 0.0
    if n1 > 0:
        result += part1 / n1
    if n2 > 0:
        result += part2 / n2

    return result


def russian_roulette_monte_carlo(n: int, r_cutoff: float, rng: random.Random) -> float:
    q = 1.0 - r_cutoff
    if q <= 0.0:
        raise ValueError("R должно быть меньше 1")

    s = 0.0
    for _ in range(n):
        x = rng.uniform(A, B)
        if rng.random() < q:
            s += f(x) / q
        else:
            s += 0.0

    return (B - A) * s / n


def error_abs(estimate: float) -> float:
    return abs(I_TRUE - estimate)


def delta_estimate(n: int) -> float:
    return I_TRUE / math.sqrt(n)


def main() -> None:
    print(f"Функция: f(x) = x^2")
    print(f"Интервал: [{A}, {B}]")
    print(f"Истинное значение интеграла: {I_TRUE:.8f}")
    print(f"Используемые N: {N_VALUES}")
    print(f"Шаги стратификации: {STRAT_STEPS}")
    print(f"Пороги русской рулетки: {RUSSIAN_R_VALUES}")
    print(f"Seed генератора: {SEED}")

    print_header("АНАЛИТИЧЕСКОЕ РЕШЕНИЕ")
    print("∫[2,5] x^2 dx = (5^3 - 2^3) / 3 = 39.0")

    rows = []
    for n in N_VALUES:
        rng = random.Random(SEED + 1000 + n)
        estimate = simple_monte_carlo(n, rng)
        rows.append((
            "-",
            n,
            I_TRUE,
            estimate,
            error_abs(estimate),
            delta_estimate(n),
        ))
    print_table("ПРОСТОЕ МОНТЕ-КАРЛО", rows)

    rows = []
    for step in STRAT_STEPS:
        for n in N_VALUES:
            rng = random.Random(SEED + 2000 + int(step * 1000) + n)
            estimate = stratified_monte_carlo(n, step, rng)
            rows.append((
                f"step={step}",
                n,
                I_TRUE,
                estimate,
                error_abs(estimate),
                delta_estimate(n),
            ))
    print_table("МОНТЕ-КАРЛО СО СТРАТИФИКАЦИЕЙ", rows)

    rows = []
    for k in [1, 2, 3]:
        for n in N_VALUES:
            rng = random.Random(SEED + 3000 + 10 * k + n)
            estimate = importance_sampling(n, k, rng)
            rows.append((
                f"p(x)~x^{k}",
                n,
                I_TRUE,
                estimate,
                error_abs(estimate),
                delta_estimate(n),
            ))
    print_table("ВЫБОРКА ПО ЗНАЧИМОСТИ", rows)

    rows = []
    for mode in ["balance", "power2"]:
        for n in N_VALUES:
            rng = random.Random(SEED + 4000 + (1 if mode == "balance" else 2) * 100 + n)
            estimate = multiple_importance_sampling(n, mode, rng)
            rows.append((
                mode,
                n,
                I_TRUE,
                estimate,
                error_abs(estimate),
                delta_estimate(n),
            ))
    print_table("МНОГОКРАТНАЯ ВЫБОРКА ПО ЗНАЧИМОСТИ (MIS)", rows)

    rows = []
    for r in RUSSIAN_R_VALUES:
        for n in N_VALUES:
            rng = random.Random(SEED + 5000 + int(r * 1000) + n)
            estimate = russian_roulette_monte_carlo(n, r, rng)
            rows.append((
                f"R={r}",
                n,
                I_TRUE,
                estimate,
                error_abs(estimate),
                delta_estimate(n),
            ))
    print_table("МОНТЕ-КАРЛО С РУССКОЙ РУЛЕТКОЙ", rows)


if __name__ == "__main__":
    main()