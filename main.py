class PseudoRandomNumberGenerator:
    def __init__(self, beta, M, a0_prime):
        self.beta = beta
        self.M = M
        self.a_prime = a0_prime

    def generate_next(self):
        self.a_prime = (self.beta * self.a_prime) % self.M
        return self.a_prime / self.M

    def generate_in_range(self, a, b):
        r = self.generate_next()
        return a + r * (b - a)


if __name__ == "__main__":
    beta = 31
    M = 10**5
    a0_prime = 7

    generator = PseudoRandomNumberGenerator(beta, M, a0_prime)

    for _ in range(5):
        print(generator.generate_next())

    print("Random number in range (5, 10):", generator.generate_in_range(5, 10))
