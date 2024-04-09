import math
from typing import Generator
import scipy.stats as stats

type Gen = Generator[float, None, None]
type Sequence = list[float]
type Interval = tuple[float, float]
type VariationSeries = dict[Interval, int]
type Frequency = dict[Interval, float]


def residual(beta: int, m: int, a0_prime: int) -> Gen:
    a_prime = a0_prime
    while True:
        ai_prime = (beta * a_prime) % m
        ai = ai_prime / m
        yield ai
        a_prime = ai_prime


def uniform_random(a: float, b: float, generator: Gen) -> Gen:
    while True:
        r = next(generator)
        yield a + (b - a) * r


def convert_to_normal(N: int, mu: float, sigma: float, uniform_generator: Gen) -> float:
    random_numbers = [next(uniform_generator) for _ in range(N)]
    theta = math.sqrt(12 / N) * (sum(random_numbers) - N / 2)
    nu = mu + math.sqrt(sigma) * theta
    return nu


def create_normal_distribution(
    size: int, mu: float, sigma: float, generator: Gen
) -> Sequence:
    return [convert_to_normal(size, mu, sigma, generator) for _ in range(size)]


def construct_interval_series(data: Sequence) -> VariationSeries:
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


def compute_empirical_frequencies(series: VariationSeries, data: Sequence) -> Frequency:
    n = len(data)
    return {k: v / n for k, v in series.items()}


def calculate_probability(
    bounds: tuple[float, float], mu: float, sigma: float, epsilon: float = 1e-4
) -> float:
    total_probability = 0.0
    x = bounds[0]
    while x < bounds[1]:
        x += epsilon
        total_probability += epsilon * stats.norm.pdf(x, loc=mu, scale=sigma)
    return float(total_probability)


def compute_theoretical_frequency(
    intervals: list[Interval], mu: float, sigma: float
) -> Frequency:
    frequencies = {}
    for interval in intervals:
        probability = calculate_probability(interval, mu, sigma)
        frequencies[interval] = probability
    return frequencies


def compute_chi_square(empirical: Frequency, theoretical: Frequency) -> float:
    return sum(
        ((empirical[k] - theoretical[k]) ** 2) / theoretical[k]
        for k in empirical.keys()
    )


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

    data = create_normal_distribution(N, a, D, uniform_generator)
    data = sorted(data)

    interval_series = construct_interval_series(data)
    empirical_frequencies = compute_empirical_frequencies(interval_series, data)

    intervals = list(interval_series.keys())
    theoretical_frequencies = compute_theoretical_frequency(intervals, a, D)

    critical_value = compute_chi_square(empirical_frequencies, theoretical_frequencies)

    alpha = 0.05
    v = len(intervals) - 2
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
