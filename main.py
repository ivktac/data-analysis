import math
from typing import Generator
import pandas as pd
import scipy.stats as stats


def residual(beta: int, M: int, a0_prime: int) -> Generator[float, None, None]:
    a_prime = a0_prime
    while True:
        ai_prime = (beta * a_prime) % M
        ai = ai_prime / M
        yield ai
        a_prime = ai_prime


def uniform_random(
    a: float, b: float, generator: Generator[float, None, None]
) -> Generator[float, None, None]:
    while True:
        r = next(generator)
        yield a + (b - a) * r


def conver_to_normal(
    N: int, a: float, D: float, uniform_generator: Generator[float, None, None]
) -> float:
    random_numbers = [next(uniform_generator) for _ in range(N)]
    theta = math.sqrt(12 / N) * (sum(random_numbers) - N / 2)
    nu = a + math.sqrt(D) * theta
    return nu


def construct_interval_series(data: list[float]):
    x_min, x_max = min(data), max(data)
    R = x_max - x_min
    k = 1 + 3.322 * math.log(len(data))
    h = R / k

    series = {}
    for i in range(int(k)):
        x_i = x_min + i * h
        n_i = sum(1 for x in data if x >= x_i and x < x_i + h)
        series[(round(x_i, 2), round(x_i + h, 2))] = n_i

    return series


def compute_empirical_frequency(
    series: dict[tuple[float, float], int], data: list[float]
):
    n = len(data)
    return {k: v / n for k, v in series.items()}


def normal_distribution_probability(
    lower_bound: float,
    upper_bound: float,
    mu: float,
    sigma: float,
    epsilon: float = 1e-3,
) -> float:
    total_probability = 0.0
    x = lower_bound
    while x < upper_bound:
        pdf = (
            1
            / (sigma * math.sqrt(2 * math.pi))
            * (math.exp(-0.5 * ((x - mu) / sigma) ** 2))
        )
        total_probability += pdf * epsilon
        x += epsilon
    return total_probability


def compute_theoretical_frequency(
    intevals: list[tuple[float, float]], mu: float, sigma: float, epsilon: float = 1e-3
):
    thereotical_frequencies = {}
    for interval in intevals:
        a, b = interval
        probability = normal_distribution_probability(a, b, mu, sigma, epsilon)
        thereotical_frequencies[interval] = probability
    return thereotical_frequencies


def compute_criteria_pirson(
    empirical: dict[tuple[float, float], float],
    theoretical: dict[tuple[float, float], float],
):
    criteria = 0
    for interval, empirical_frequency in empirical.items():
        theoretical_frequency = theoretical[interval]
        criteria += (
            empirical_frequency - theoretical_frequency
        ) ** 2 / theoretical_frequency
    return criteria


def main():
    p = 31
    M = 2**p
    a0_prime = M - 1
    beta = 5 ** (2 * p + 1)
    a_generator = residual(beta, M, a0_prime)
    uniform_generator = uniform_random(0, 1, a_generator)

    N = 160
    a = 6
    D = next(uniform_generator)
    print(f"{a=}, {D=}")

    samples = [conver_to_normal(N, a, D, a_generator) for _ in range(N)]
    sorted_samples = sorted(samples)

    interval_series = construct_interval_series(sorted_samples)
    empirical_frequency = compute_empirical_frequency(interval_series, sorted_samples)

    intervals = list(interval_series.keys())
    theoretical_frequency = compute_theoretical_frequency(intervals, a, D)

    df = pd.DataFrame(
        {
            "Intervals": [f"{k[0]} - {k[1]}" for k in interval_series.keys()],
            "Frequency": list(interval_series.values()),
            "Empirical Frequency": list(empirical_frequency.values()),
            "Theoretical Frequency": list(theoretical_frequency.values()),
        }
    )
    print(df)

    critical_value = compute_criteria_pirson(empirical_frequency, theoretical_frequency)
    v = len(intervals) - 2

    alpha = 0.05
    chi2 = stats.chi2.ppf(1 - alpha, v)
    print(
        f"Критичне значення: {critical_value}, критерій пірсона: {chi2} при ступені свободи {v=},"
    )
    if critical_value < chi2:
        print("Гіпотеза H0 приймається, дані розподілені нормально")
    else:
        print("Гіпотеза H0 відхиляється, дані не розподілені нормально")


if __name__ == "__main__":
    main()
