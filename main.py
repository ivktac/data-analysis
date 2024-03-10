import math


def linear_congruential_generator(beta, M, a0_prime):
    a_i_prime = a0_prime
    while True:
        a_i = a_i_prime / M
        yield a_i
        a_i_prime = (beta * a_i_prime) % M


def generate_uniform_random(a, b, generator):
    while True:
        r = next(generator)
        yield a + (b - a) * r


def generate_random_numbers(n):
    M = 10**8
    a_0_prime = 12345
    beta = 197

    lcg = linear_congruential_generator(beta, M, a_0_prime)
    uniform_random = generate_uniform_random(0, 1, lcg)
    return [next(uniform_random) for _ in range(n)]


def calculate_moments(random_numbers):
    n = len(random_numbers)
    mean = sum(random_numbers) / n
    variance = sum((x - mean) ** 2 for x in random_numbers) / n
    return mean, variance


def perform_tests(random_numbers, delta, y):
    n = len(random_numbers)
    mean, variance = calculate_moments(random_numbers)

    # Compute random deviations
    xi_1 = mean - 0.5
    xi_2 = variance - 1 / 12

    # Calculate test statistics
    w1 = math.sqrt(12) * abs(xi_1)
    w2 = xi_2 / math.sqrt(1 / (12 * n))

    # Test for mean deviation
    if w1 < delta:
        print(f"Accept H0 for mean deviation with confidence probability {y}")
    else:
        print(f"Reject H0 for mean deviation with confidence probability {y}")

    # Test for variance deviation
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
        f"Proportion of numbers within (m - sigma, m + sigma): {actual_proportion:.4f}"
    )
    print(f"Expected proportion for uniform distribution: {expected_proportion:.4f}")


def generate_normal_random(n, a, D):
    M = 10**8
    a_0_prime = 12345
    beta = 197

    lcg = linear_congruential_generator(beta, M, a_0_prime)
    uniform_random = generate_uniform_random(0, 1, lcg)

    random_numbers = []

    for _ in range(n):
        xi = 0
        for _ in range(12):
            xi += next(uniform_random)
        xi -= 6
        nu = a + math.sqrt(D) * xi
        random_numbers.append(nu)

    return random_numbers


def task1():
    M = 10**8  # M = 10^8

    a0_prime = 12345  # Not a multiple of 2 or 5
    beta = 197  # One of the recommended values

    # Create the linear congruential generator
    lcg = linear_congruential_generator(beta, M, a0_prime)

    # Generate uniform random numbers in the interval (0, 1)
    uniform_random = generate_uniform_random(0, 1, lcg)

    # Generate uniform random numbers in a different interval (e.g., (10, 20))
    interval_random = generate_uniform_random(10, 20, lcg)

    # Print the first 10 random numbers
    print("First 10 uniform random numbers in (0, 1):")
    for _ in range(10):
        print(next(uniform_random))

    print("\nFirst 10 uniform random numbers in (10, 20):")
    for _ in range(10):
        print(next(interval_random))


def task2():
    n = 10000
    random_numbers = generate_random_numbers(n)
    delta = 1.96  # For 95% confidence level
    y = 0.95

    perform_tests(random_numbers, delta, y)


def task3():
    N = 6
    n = 100 + 10 * N
    a = N
    D = 4

    normal_random_numbers = generate_normal_random(n, a, D)

    mean = sum(normal_random_numbers) / n
    variance = sum((x - mean) ** 2 for x in normal_random_numbers) / n

    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")


if __name__ == "__main__":
    task1()
    task2()
    task3()
