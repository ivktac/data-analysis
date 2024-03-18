import math


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

        for i in range(n):
            xi = (sum(random_numbers[i : i + 12]) - (n / 2)) * math.sqrt(12 / n)
            etha = a + math.sqrt(D) * xi
            normal_dist.append(etha)

        return normal_dist


# Example usage
if __name__ == "__main__":
    # Define parameters
    n = 10000
    a = 6
    D = 0.02
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
