import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class LinearCongruentialGenerator:
    def __init__(self, beta, M, a0_prime):
        self.beta = beta
        self.M = M
        self.a_i_prime = a0_prime

    def __iter__(self):
        return self

    def __next__(self):
        a_i = self.a_i_prime / self.M
        self.a_i_prime = (self.beta * self.a_i_prime) % self.M
        return a_i


class UniformRandomGenerator:
    def __init__(self, a, b, base_generator):
        self.a = a
        self.b = b
        self.base_generator = base_generator

    def generate(self, n):
        return [
            self.a + (self.b - self.a) * next(self.base_generator) for _ in range(n)
        ]


class StatisticalTests:
    @staticmethod
    def calculate_moments(random_numbers):
        n = len(random_numbers)
        mean = sum(random_numbers) / n
        variance = sum((x - mean) ** 2 for x in random_numbers) / n
        return mean, variance

    @staticmethod
    def perform_tests(random_numbers, delta, y):
        n = len(random_numbers)
        mean, variance = StatisticalTests.calculate_moments(random_numbers)

        xi_1 = mean - 0.5
        xi_2 = variance - 1 / 12

        w1 = math.sqrt(12) * abs(xi_1)
        w2 = xi_2 / math.sqrt(0.0056 / n + 0.0028 / (n**2) - 0.0083 / (n**3))

        if w1 < delta:
            print(f"Accept H0 for mean deviation with confidence probability {y}")
        else:
            print(f"Reject H0 for mean deviation with confidence probability {y}")

        if w2 < delta:
            print(f"Accept H0 for variance deviation with confidence probability {y}")
        else:
            print(f"Reject H0 for variance deviation with confidence probability {y}")

        # Frequency test
        sigma = math.sqrt(variance)
        lower_bound = mean - sigma
        upper_bound = mean + sigma
        count_in_range = sum(lower_bound <= x <= upper_bound for x in random_numbers)
        expected_proportion = 0.577
        actual_proportion = count_in_range / n
        print(
            f"Proportion of numbers within (m - {sigma}, m + {sigma}): {actual_proportion:.4f}"
        )
        print(
            f"Expected proportion for uniform distribution: {expected_proportion:.4f}"
        )


class RandomNumberGenerator:
    @staticmethod
    def generate_normal_random(n, a, D, beta, M, a0_prime):
        lcg = LinearCongruentialGenerator(beta, M, a0_prime)
        uniform_random = UniformRandomGenerator(0, 1, lcg)

        random_numbers = uniform_random.generate(n)
        normal_dist = []

        N = 12
        for i in range(n):
            xi = (sum(random_numbers[i : i + N]) - (N / 2)) * math.sqrt(12 / N)
            etha = a + math.sqrt(D) * xi
            normal_dist.append(etha)

        return normal_dist


# Example usage
if __name__ == "__main__":
    # Define parameters
    n = 10000
    a = 6
    D = 0.2
    beta = 5 ** (2 * 3 + 1)
    M = 2**8
    a0_prime = 13

    # Generator initialization
    lcg = LinearCongruentialGenerator(beta, M, a0_prime)
    uniform_generator = UniformRandomGenerator(0, 1, lcg)
    random_numbers = uniform_generator.generate(n)

    # Perform statistical tests
    delta = 1.96  # For 95% confidence level
    y = 0.95
    StatisticalTests.perform_tests(random_numbers, delta, y)

    n = 100 + 10 * a
    # Generate normal distributed numbers
    normal_numbers = RandomNumberGenerator.generate_normal_random(
        n, a, D, beta, M, a0_prime
    )
    for i in range(n):
        print(f"Normal random number {i + 1}: {normal_numbers[i]:.4f}")

    mean = sum(normal_numbers) / n
    var = sum((x - mean) ** 2 for x in normal_numbers) / n
    print(f"Mean: {mean:.4f}, variance: {var:.4f}")

    # k = round(1 + 3.322 * math.log10(n))
    k = 10 + a
    min = min(normal_numbers)
    max = max(normal_numbers)
    h = (max - min) / k
    intervals = [min + i * h for i in range(k + 1)]
    frequencies = [0] * k
    for x in normal_numbers:
        for i in range(k):
            if intervals[i] <= x < intervals[i + 1]:
                frequencies[i] += 1
                break

    for i in range(k):
        print(
            f"Interval {i + 1}: [{intervals[i]:.4f}, {intervals[i + 1]:.4f}] - Frequency: {frequencies[i]}"
        )

    # find mean, standard deviation, variance and anomalies
    mean = sum(intervals[i] * frequencies[i] for i in range(k)) / n
    print(f"Mean: {mean:.4f}")

    var = sum((intervals[i] - mean) ** 2 * frequencies[i] for i in range(k)) / n

    std = math.sqrt(var)
    print(f"Variance: {var:.4f}, Standard deviation: {std:.4f}")

    var_ddof = sum((intervals[i] - mean) ** 2 * frequencies[i] for i in range(k)) / (
        n - 1
    )
    print(f"Sample variance: {var_ddof:.4f}")

    # find anomalies by rule of three sigmas
    anomalies = []
    for i in range(k):
        if abs(intervals[i] - mean) > 3 * std:
            anomalies.append(i)
    print(f"Anomalies count: {len(anomalies)}")

    # Build an empirical data distribution function
    x = [intervals[i] for i in range(k)]
    y = [sum(frequencies[:i]) / n for i in range(k)]
    plt.figure(figsize=(10, 6))
    plt.step(x, y, where="mid", linestyle="--", label="Empirical Distribution")
    plt.xlabel("Інтервали")
    plt.ylabel("Нормована частота")
    plt.title("Емпірична функція розподілу")
    plt.legend()
    plt.grid(True)
    plt.show()

    numbers = np.array(normal_numbers)
    hist, bin_edges = np.histogram(numbers, bins="auto", density=False)
    empirical_frequencies = hist / len(numbers)

    mean, std_dev = np.mean(numbers), np.std(numbers)

    intervals = len(bin_edges) - 1

    theoretical_frequencies = np.array(
        [
            len(numbers)
            * (
                stats.norm.cdf(bin_edges[i + 1], loc=mean, scale=std_dev)
                - stats.norm.cdf(bin_edges[i], loc=mean, scale=std_dev)
            )
            for i in range(intervals)
        ]
    )
    chi2 = np.sum(
        (empirical_frequencies - theoretical_frequencies) ** 2 / theoretical_frequencies
    )

    k = len(bin_edges) - 1
    s = 2  # Кількість параметрів розподілу для нормального розподілу (середнє та стандартне відхилення)
    v = k - s - 1

    alpha = 0.95
    critical_value = stats.chi2.ppf(1 - alpha, v)
    if chi2 < critical_value:
        print("Accept the hypothesis that the data comes from the normal distribution")
    else:
        print("Reject the hypothesis that the data comes from the normal distribution")
