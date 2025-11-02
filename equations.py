import numpy as np


def equation_21(u: int) -> float:
    """Equation 21: Mapping from observed outcome to perceptual input

    Eq. 21: $\mu_{1}^{(k)}$ = u^{k}

    """
    m1 = float(u)
    return m1


def equation_25(m1: float, m1_hat: float) -> float:
    """Equation 25: Prediction error at Level 1

    Eq. 25: $\delta_{1}^{(k)}$ = $\mu_{1}^{(k)}$ - $\hat{\mu}_{1}^{(k)}$

    """

    if m1 < 1 and m1_hat < 1 and m1 - m1_hat == 0:
        raise ValueError("Underflow detected in equation_25: m1 or m1_hat is too small.")

    d1 = m1 - m1_hat

    return d1


def equation_23(m2_prev: float, s2: float, d1: float) -> float:
    """Equation 23: Update of Level 2 mean

    Eq. 23: $\mu_{2}^{(k-1)} + \sigma_{2}^{(k)} * \delta_{1}^{(k)}$

    """

    # Check for underflow
    if s2 != 0 and d1 != 0 and s2 * d1 == 0:
        raise ValueError("Underflow detected in equation_23: s2 or d1 is too small.")

    m2 = m2_prev + s2 * d1

    return m2


def equation_32(kappa: float, m3: float, omega: float, s2_prev: float) -> float:
    """Equation 32: Helper equation w2

    Eq. 32: $w_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}$

    """
    if kappa != 0 and m3 != 0 and kappa * m3 == 0:
        raise ValueError("Underflow detected in equation_32: kappa or m3 is too small.")

    if abs(kappa * m3) < 1 and abs(omega) < 1 and abs(kappa * m3 + omega) == 0:
        raise ValueError("Underflow detected in equation_32: kappa * m3 + omega is too small.")

    numerator = np.exp(kappa * m3 + omega)
    denominator = numerator + s2_prev

    if denominator == 0:
        raise ValueError("Division by zero detected in equation_32: denominator is zero.")

    w2 = numerator / denominator

    return w2


def equation_33(kappa: float, m3: float, omega: float, s2_prev: float) -> float:
    """Equation 33: Helper equation r2

    Eq. 33: $r_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega} - s_2^{(k-1)}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}$

    """
    if kappa != 0 and m3 != 0 and kappa * m3 == 0:
        raise ValueError("Underflow detected in equation_33: kappa or m3 is too small.")

    if abs(kappa * m3) < 1 and abs(omega) < 1 and abs(kappa * m3 + omega) == 0:
        raise ValueError("Underflow detected in equation_33: kappa * m3 + omega is too small.")

    exp_term = np.exp(kappa * m3 + omega)
    denominator = exp_term + s2_prev

    if denominator == 0:
        raise ValueError("Division by zero detected in equation_33: denominator is zero.")

    r2 = (exp_term - s2_prev) / denominator

    return r2


def equation_34(
    s2: float, m2: float, m2_prev: float, s2_prev: float, kappa: float, m3: float, omega: float
) -> float:
    """Equation 34: Prediction error at Level 2

    Eq. 34: $\delta_{2}^{(k)} = \frac{\sigma_{2}^{(k)} + (\mu_{2}^{(k)} - \mu_{2}^{(k-1)})^2}{\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}} - 1$

    """
    if kappa != 0 and m3 != 0 and kappa * m3 == 0:
        raise ValueError("Underflow detected in equation_34: kappa or m3 is too small.")

    if abs(kappa * m3) < 1 and abs(omega) < 1 and abs(kappa * m3 + omega) == 0:
        raise ValueError("Underflow detected in equation_34: kappa * m3 + omega is too small.")

    exp_term = np.exp(kappa * m3 + omega)
    denominator = s2_prev + exp_term

    if denominator == 0:
        raise ValueError("Division by zero detected in equation_34: denominator is zero.")

    d2 = (s2 + (m2 - m2_prev) ** 2) / denominator - 1

    return d2


def equation_29(pi3_hat: float, kappa: float, w2: float, r2: float, d2: float) -> float:
    """Equation 29: Update of Level 3 precision

    Eq. 29: $\pi_{3} = \hat{\pi}_{3}^{(k)} + \frac{\kappa^2}{2} * w_{2}^{(k)} * (w_{2}^{(k)} + r_{2}^{(k)} * \delta_{2}^{(k)})$

    """
    if kappa != 0 and w2 != 0 and kappa**2 * w2 == 0:
        raise ValueError("Underflow detected in equation_29: kappa or w2 is too small.")

    if r2 != 0 and d2 != 0 and r2 * d2 == 0:
        raise ValueError("Underflow detected in equation_29: r2 or d2 is too small.")

    pi3 = pi3_hat + (kappa**2 / 2) * w2 * (w2 + r2 * d2)

    return pi3


def equation_30(m3_prev: float, s3: float, kappa: float, w2: float, d2: float) -> float:
    """Equation 30: Update of Level 3 mean

    Eq. 30: $\mu_{3}^{(k)} = \mu_{3}^{(k-1)} + \sigma_{3}^{(k)} * \frac{\kappa}{2} * w_{2}^{(k)} * \delta_{2}^{(k)}$

    """
    if s3 != 0 and (kappa / 2) != 0 and w2 != 0 and d2 != 0 and s3 * (kappa / 2) * w2 * d2 == 0:
        raise ValueError("Underflow detected in equation_30: s3, kappa, w2, or d2 is too small.")

    m3 = m3_prev + s3 * (kappa / 2) * w2 * d2

    return m3


def equation_27(s2_prev: float, kappa: float, m3: float, omega: float) -> float:
    """Equation 27: Prediction of Level 2 variance

    Eq. 27: $\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}$

    """
    if kappa != 0 and m3 != 0 and kappa * m3 == 0:
        raise ValueError("Underflow detected in equation_27: kappa or m3 is too small.")

    if abs(kappa * m3) < 1 and abs(omega) < 1 and abs(kappa * m3 + omega) == 0:
        raise ValueError("Underflow detected in equation_27: kappa * m3 + omega is too small.")

    s2_hat = s2_prev + np.exp(kappa * m3 + omega)

    return s2_hat
