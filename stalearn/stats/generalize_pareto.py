class GPD:
    @staticmethod
    def pdf(x, xi, beta):
        return (1/beta) * (1 + xi * x / beta)**(-1/xi - 1)

    @staticmethod
    def cdf(x, xi, beta):
        return 1 - ( 1 + xi * x / beta) ** (-1 / xi)

