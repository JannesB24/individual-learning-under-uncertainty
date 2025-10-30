import numpy as np


class HGF:
    """Hierarchical Gaussian Filter"""

    def __init__(self, kappa, omega, theta):
        # Store parameters
        self.kappa = kappa
        self.omega = omega
        self.theta = theta

        # Initialize states - see Simulation on page 11
        self.m2 = 0.0  # $\mu_{2}^{(n)}$, n = 0, ..., k - 1
        self.m2 = 0.0  # $\mu_{3}^{(n)}$, n = 0, ..., k - 1
        self.m3 = 0.0  # $\mu_{3}^{(n)}$, n = 0, ..., k - 1
        self.s2 = 1.0  # $\sigma_{2}^{(n)}$, n = 0, ..., k - 1
        self.s3 = 1.0  # $\sigma_{3}^{(n)}$, n = 0, ..., k - 1

        self.m1_hat_prev = "TODO"  # TODO where to update?!

    def _predictions(self) -> tuple:
        """Predictions from observation $u^{(k)}$ upwards the hierarchial model"""
        # Eq 24:$\hat{m})_{1}^{(k)} = sigmoid(\mu_{2}^{(k-1)})$
        m1_hat = sigmoid(self.m2)

        # Eq 26: $\hat{\sigma}_{1}^{(k-1)} = \hat{\mu}_{1}^{(k-1)} * (1 - \hat{\mu}_{1}^{(k-1)})$
        s1_hat = self.m1_hat_prev * (1 - self.m1_hat_prev)

        # Eq 27: $\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}$ -> no $\sigma_{3}^{(k-1)}$ present!
        s2_hat = self.s2 + np.exp(self.kappa * self.m3 + self.omega)

        # Eq 31: $\hat{\pi}_{3}^{(k)} = 1 / (\sigma_{3}^{(k-1)} + \theta)$
        pi3_hat = 1 / (self.s3 + self.theta)

        return s1_hat, s2_hat, pi3_hat, m1_hat

    def update(self, u):
        """Update beliefs given binary input u (0 or 1) for point in time k."""

        # STEP 1: Make predictions (using previous trial's values)
        s1_hat, s2_hat, pi3_hat, m1_hat = self._predictions()

        # STEP 2: Update Level 1
        ## Eq. 21: simply the input $u^{(k)}$
        self.m1 = float(u)

        # STEP 3: Compute prediction error at Level 1
        ## Eq. 25: $\delta_{1}^{(k)} = \mu_{1}^{(k)} - \hat{\mu}_{1}^{(k)}$
        d1 = self.m1 - m1_hat

        # STEP 4: Update Level 2 (m2 and s2)
        ## Eq. 22: $\frac{1}{\frac{1}{\hat{\sigma}_2^{(k)}} + \sigma_{1}^{(k)}}$
        s2_prev = self.s2
        self.s2 = 1 / (1 / (s2_hat) + s1_hat)

        ## Eq. 23: $\mu_{2}^{(k-1)} + \sigma_{2}^{(k)} * \delta_{1}^{(k)}$
        m2_prev = self.m2
        self.m2 = self.m2 + self.s2 * d1

        # STEP 5: Compute prediction error at Level 2
        ## Eq. 34: $\delta_{2}^{(k)} = \frac{\sigma_{2}^{(k)} + (\mu_{2}^{(k)} - \mu_{2}^{(k-1)})^2}{\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}} - 1$
        d2 = (self.s2 + (self.m2 - m2_prev) ** 2) / (
            s2_prev + np.exp(self.kappa * self.s3 + self.omega)
        ) - 1

        ## Eq. 32: $w_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}$
        w2 = np.exp(self.kappa * self.m3 + self.omega) / (
            np.exp(self.kappa * self.m3 + self.omega) + s2_prev
        )

        ## Eq. 33: r_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega} - s_2^{(k-1)}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}
        r2 = (np.exp(self.kappa * self.m3 + self.omega) - s2_prev) / (
            np.exp(self.kappa * self.m3 + self.omega) + s2_prev
        )

        # STEP 6: Update Level 3 (m3 and s3)
        # TODO: Equations 29, 30

        ## Eq. 29: $\pi_{3} = \hat{\pi}_{3}^{(k)} + \frac{\kappa^2}{2} * w_{2}^{(k)} * (w_{2}^{(k)} + r_{2}^{(k)} * \delta_{2}^{(k)})$
        pi3 = pi3_hat + (self.kappa**2 / 2) * w2 * (w2 + r2 * d2)

        self.s3 = 1 / pi3

        ## Eq. 30: $
        self.m3 = self.m3 + self.s3 * (self.kappa / 2) * w2 * d2

        self.m1_hat_prev

        return self.m2, self.m3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
