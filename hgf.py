from matplotlib import pyplot as plt
import numpy as np
import warnings

from equations import (
    equation_21,
    equation_23,
    equation_25,
    equation_27,
    equation_29,
    equation_30,
    equation_32,
    equation_33,
    equation_34,
)

# At the top of your file, enable all numpy warnings
np.seterr(all="raise")  # or 'raise' to stop execution
warnings.filterwarnings("error")  # Convert warnings to exceptions


class HGF:
    """Hierarchical Gaussian Filter"""

    def __init__(self, kappa, omega, theta):
        # Store parameters
        self.kappa = kappa
        self.omega = omega
        self.theta = theta

        # Initialize states - see Simulation on page 11
        self._m2 = 0.0  # $\mu_{2}^{(n)}$, n = 0, ..., k - 1
        self.m3 = 0.0  # $\mu_{3}^{(n)}$, n = 0, ..., k - 1
        self.s2 = 1.0  # $\sigma_{2}^{(n)}$, n = 0, ..., k - 1
        self.s3 = 1.0  # $\sigma_{3}^{(n)}$, n = 0, ..., k - 1

        self.m1_hat_prev = 0
        self._counter = 0

        # Expanded history to track all calculations
        self.history = {
            "u": [],  # Input
            "m1": [],  # Level 1 posterior mean
            "m2": [],  # Level 2 posterior mean
            "m3": [],  # Level 3 posterior mean
            "s2": [],  # Level 2 posterior variance
            "s3": [],  # Level 3 posterior variance
            "m1_hat": [],  # Level 1 prediction
            "s1_hat": [],  # Level 1 predicted variance
            "s2_hat": [],  # Level 2 predicted variance
            "pi1_hat": [],  # Level 1 predicted precision
            "pi2_hat": [],  # Level 2 predicted precision
            "pi3_hat": [],  # Level 3 predicted precision
            "d1": [],  # Level 1 prediction error
            "d2": [],  # Level 2 prediction error
            "w2": [],  # Weight factor w2
            "r2": [],  # Weight factor r2
            "pi2": [],  # Level 2 updated precision
            "pi3": [],  # Level 3 updated precision
        }

    def _predictions(self) -> tuple:
        """Using the internal model from (k-1) to make prediction about (k) before observing the next sensory value (u^{(k)})."""
        # Eq. 24:$\hat{m})_{1}^{(k)} = sigmoid(\mu_{2}^{(k-1)})$
        m1_hat = sigmoid(
            self._m2
        )  # Expected posterior parameter mu (believe about the future state of the environment)

        # Eq 26: $\hat{\sigma}_{1}^{(k-1)} = \hat{\mu}_{1}^{(k-1)} * (1 - \hat{\mu}_{1}^{(k-1)})$
        s1_hat = (
            self.m1_hat_prev * (1 - self.m1_hat_prev)
        )  # Expected posterior parameter sigma (uncertainty about the future state of the environment)

        # Division by zero undetected?!
        pi1_hat = 1 / s1_hat if s1_hat != 0 else 1
        # prevent underflow with minimal numerical epsilon

        s2_hat = equation_27(self.s2, self.kappa, self.m3, self.omega)
        pi2_hat = 1 / s2_hat  # prevent underflow, small to big number

        # posterior variance s3_hat
        s3_hat = self.s3 + self.theta
        # Eq 31: $\hat{\pi}_{3}^{(k)} = 1 / (\sigma_{3}^{(k-1)} + \theta)$
        pi3_hat = 1 / s3_hat

        self.m1_hat_prev = m1_hat

        print(
            f"Predictions Step {self._counter}: m1_hat={m1_hat:.4f}, s1_hat={s1_hat:.4f}, s2_hat={s2_hat:.4f}, pi1_hat={pi1_hat:.4f}, pi2_hat={pi2_hat:.4f}, pi3_hat={pi3_hat:.4f}"
        )

        if m1_hat > 1 or m1_hat < 0:
            raise ValueError(f"m1_hat out of bounds: {m1_hat}")

        if np.sqrt(s1_hat) > 4:
            raise ValueError(f"s1_hat out of bounds: {s1_hat}")

        if np.sqrt(s2_hat) > 2.0:
            raise ValueError(f"s2_hat out of bounds: {s2_hat}")

        if np.sqrt(s3_hat) > 4:
            raise ValueError(f"s3_hat out of bounds: {s3_hat}")

        return pi1_hat, pi2_hat, pi3_hat, m1_hat, s1_hat, s2_hat

    def update(self, u: int):
        """Update beliefs given binary input u (0 or 1) for point in time k."""

        # STEP 1: Make predictions (using previous trial's values)
        pi1_hat, pi2_hat, pi3_hat, m1_hat, s1_hat, s2_hat = self._predictions()

        # STEP 2: Update LEVEL 1
        self.m1 = equation_21(u)

        # STEP 3: Compute prediction error at Level 1
        d1 = equation_25(self.m1, m1_hat)

        # STEP 4: Update Level 2 (m2 and s2)
        ## Eq. 22: $\frac{1}{\frac{1}{\hat{\sigma}_2^{(k)}} + \sigma_{1}^{(k)}}$
        # self.s2 = 1 / (1 / (s2_hat) + s1_hat) # Values get too little underflow!

        s2_prev = self.s2
        pi2 = pi2_hat + (1 / pi1_hat)

        self.s2 = 1 / pi2

        m2_prev = self._m2
        self._m2 = equation_23(m2_prev, self.s2, d1)

        # STEP 5: Compute prediction error at Level 2 and helper equations
        w2 = equation_32(self.kappa, self.m3, self.omega, s2_prev)

        r2 = equation_33(self.kappa, self.m3, self.omega, s2_prev)

        d2 = equation_34(self.s2, self._m2, m2_prev, s2_prev, self.kappa, self.m3, self.omega)

        # STEP 6: Update Level 3 (m3 and s3)
        pi3 = equation_29(pi3_hat, self.kappa, w2, r2, d2)
        self.s3 = 1 / pi3

        self.m3 = equation_30(self.m3, self.s3, self.kappa, w2, d2)

        print(
            f"Update Step {self._counter}: m1={self.m1:.4f}, m2={self._m2:.4f}, m3={self.m3:.4f}, "
            f"s2={self.s2:.4f}, s3={self.s3:.4f}, d1={d1:.4f}, d2={d2:.4f}, "
            f"w2={w2:.4f}, r2={r2:.4f}, pi2={pi2:.4f}, pi3={pi3:.4f}"
        )

        # Store all calculations in history
        self.history["u"].append(u)
        self.history["m1"].append(self.m1)
        self.history["m2"].append(self._m2)
        self.history["m3"].append(self.m3)
        self.history["s2"].append(self.s2)
        self.history["s3"].append(self.s3)
        self.history["m1_hat"].append(m1_hat)
        self.history["s1_hat"].append(s1_hat)
        self.history["s2_hat"].append(s2_hat)
        self.history["pi1_hat"].append(pi1_hat)
        self.history["pi2_hat"].append(pi2_hat)
        self.history["pi3_hat"].append(pi3_hat)
        self.history["d1"].append(d1)
        self.history["d2"].append(d2)
        self.history["w2"].append(w2)
        self.history["r2"].append(r2)
        self.history["pi2"].append(pi2)
        self.history["pi3"].append(pi3)

        self._counter += 1

    def run_simulation(self, inputs):
        for u in inputs:
            self.update(u)
        return self.history

    def plot_results(self, true_prob=None):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        trials = np.arange(len(self.history["u"]))

        # Level 3: Log-volatility
        ax = axes[0]
        ax.plot(trials, self.history["m3"], "b-", linewidth=2, label="Posterior E[x3] = m3")
        s3_array = np.array(self.history["s3"])
        s3_array[s3_array < 0] = 0
        m3_array = np.array(self.history["m3"])
        ax.fill_between(
            trials,
            m3_array - np.sqrt(s3_array),
            m3_array + np.sqrt(s3_array),
            alpha=0.2,
            color="blue",
        )
        ax.set_ylabel("Level 3: x3 (log-volatility)")
        ax.set_xlabel("Trial")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 2: Tendency
        ax = axes[1]
        ax.plot(trials, self.history["m2"], "r-", linewidth=2, label="Posterior E[x2] = m2")
        s2_array = np.array(self.history["s2"])
        m2_array = np.array(self.history["m2"])
        ax.fill_between(
            trials,
            m2_array - np.sqrt(s2_array),
            m2_array + np.sqrt(s2_array),
            alpha=0.2,
            color="red",
        )
        ax.set_ylabel("Level 2: x2 (tendency)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 1: Inputs and belief
        ax = axes[2]
        ax.scatter(trials, self.history["u"], c="green", s=10, label="Input u", alpha=0.6)

        m1_prob = [sigmoid(m2) for m2 in self.history["m2"]]

        ax.plot(trials, m1_prob, "r-", linewidth=2, label="Posterior E[x1=1] = s(m2)")
        if true_prob is not None:
            ax.plot(trials, true_prob, "k-", linewidth=1, label="True p(x1=1)")
        ax.set_ylabel("Level 1: x1")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_all_variables(self, true_prob=None):
        """Plot all HGF variables over time k."""

        true_prob = true_prob[-100:]
        for key in self.history.keys():
            self.history[key] = self.history[key][-100:]

        trials = np.arange(len(self.history["u"]))

        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(6, 3, figsize=(18, 20))
        fig.suptitle("HGF: All Variables Over Time k", fontsize=16)

        # Row 1: Main state variables
        # m1, m2, m3
        axes[0, 0].plot(trials, self.history["m1"], "g-", linewidth=2, label="m1 (μ₁)")
        axes[0, 0].scatter(trials, self.history["u"], c="green", s=10, alpha=0.6, label="Input u")
        axes[0, 0].set_title("Level 1: m1 (μ₁) - Posterior Mean")
        axes[0, 0].set_ylabel("m1")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(trials, self.history["m2"], "r-", linewidth=2, label="m2 (μ₂)")
        axes[0, 1].set_title("Level 2: m2 (μ₂) - Tendency")
        axes[0, 1].set_ylabel("m2")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(trials, self.history["m3"], "b-", linewidth=2, label="m3 (μ₃)")
        axes[0, 2].set_title("Level 3: m3 (μ₃) - Log-volatility")
        axes[0, 2].set_ylabel("m3")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Variance/Precision at different levels
        # s2, s3, and precision plots
        axes[1, 0].plot(trials, self.history["s2"], "r--", linewidth=2, label="s2 (σ₂²)")
        axes[1, 0].set_title("Level 2: s2 (σ₂²) - Posterior Variance")
        axes[1, 0].set_ylabel("s2")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(trials, self.history["s3"], "b--", linewidth=2, label="s3 (σ₃²)")
        axes[1, 1].set_title("Level 3: s3 (σ₃²) - Posterior Variance")
        axes[1, 1].set_ylabel("s3")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Convert s1_hat for plotting
        s1_hat_array = np.array(self.history["s1_hat"])
        axes[1, 2].plot(trials, s1_hat_array, "g--", linewidth=2, label="s1_hat (σ̂₁²)")
        axes[1, 2].set_title("Level 1: s1_hat (σ̂₁²) - Predicted Variance")
        axes[1, 2].set_ylabel("s1_hat")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Row 3: Predictions
        # m1_hat, s2_hat, predicted precisions
        axes[2, 0].plot(trials, self.history["m1_hat"], "g:", linewidth=2, label="m1_hat (μ̂₁)")
        if true_prob is not None:
            axes[2, 0].plot(trials, true_prob, "k-", linewidth=1, label="True p(x1=1)", alpha=0.7)
        axes[2, 0].set_title("Level 1: m1_hat (μ̂₁) - Predicted Mean")
        axes[2, 0].set_ylabel("m1_hat")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(trials, self.history["s2_hat"], "r:", linewidth=2, label="s2_hat (σ̂₂²)")
        axes[2, 1].set_title("Level 2: s2_hat (σ̂₂²) - Predicted Variance")
        axes[2, 1].set_ylabel("s2_hat")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # Plot precision predictions
        pi_hat_data = [self.history["pi1_hat"], self.history["pi2_hat"], self.history["pi3_hat"]]
        colors = ["g", "r", "b"]
        labels = ["π̂₁", "π̂₂", "π̂₃"]
        for i, (data, color, label) in enumerate(zip(pi_hat_data, colors, labels)):
            axes[2, 2].plot(
                trials, data, color=color, linestyle=":", linewidth=2, label=f"{label}_hat"
            )
        axes[2, 2].set_title("Predicted Precisions")
        axes[2, 2].set_ylabel("Precision")
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_yscale("log")  # Log scale for precision

        # Row 4: Prediction errors
        # d1, d2
        axes[3, 0].plot(trials, self.history["d1"], "g-", linewidth=2, label="δ₁")
        axes[3, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[3, 0].set_title("Level 1: δ₁ - Prediction Error")
        axes[3, 0].set_ylabel("δ₁")
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)

        axes[3, 1].plot(trials, self.history["d2"], "r-", linewidth=2, label="δ₂")
        axes[3, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[3, 1].set_title("Level 2: δ₂ - Prediction Error")
        axes[3, 1].set_ylabel("δ₂")
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)

        # Plot both prediction errors together for comparison
        axes[3, 2].plot(trials, self.history["d1"], "g-", linewidth=2, label="δ₁", alpha=0.7)
        axes[3, 2].plot(trials, self.history["d2"], "r-", linewidth=2, label="δ₂", alpha=0.7)
        axes[3, 2].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[3, 2].set_title("Prediction Errors Comparison")
        axes[3, 2].set_ylabel("Prediction Error")
        axes[3, 2].legend()
        axes[3, 2].grid(True, alpha=0.3)

        # Row 5: Weight factors and updated precisions
        # w2, r2
        axes[4, 0].plot(trials, self.history["w2"], "purple", linewidth=2, label="w₂")
        axes[4, 0].set_title("Weight Factor: w₂")
        axes[4, 0].set_ylabel("w₂")
        axes[4, 0].legend()
        axes[4, 0].grid(True, alpha=0.3)

        axes[4, 1].plot(trials, self.history["r2"], "orange", linewidth=2, label="r₂")
        axes[4, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[4, 1].set_title("Weight Factor: r₂")
        axes[4, 1].set_ylabel("r₂")
        axes[4, 1].legend()
        axes[4, 1].grid(True, alpha=0.3)

        # Updated precisions
        pi_data = [self.history["pi2"], self.history["pi3"]]
        colors = ["r", "b"]
        labels = ["π₂", "π₃"]
        for i, (data, color, label) in enumerate(zip(pi_data, colors, labels)):
            axes[4, 2].plot(trials, data, color=color, linewidth=2, label=label)
        axes[4, 2].set_title("Updated Precisions")
        axes[4, 2].set_ylabel("Precision")
        axes[4, 2].legend()
        axes[4, 2].grid(True, alpha=0.3)
        axes[4, 2].set_yscale("log")  # Log scale for precision

        # Row 6: Summary plots
        # Input sequence, belief vs truth, volatility measure
        axes[5, 0].plot(trials, self.history["u"], "g-", linewidth=1, alpha=0.7, label="Input u")
        axes[5, 0].scatter(
            trials[::10], np.array(self.history["u"])[::10], c="green", s=20, alpha=0.8
        )
        axes[5, 0].set_title("Input Sequence u")
        axes[5, 0].set_ylabel("Input u")
        axes[5, 0].set_xlabel("Trial k")
        axes[5, 0].legend()
        axes[5, 0].grid(True, alpha=0.3)

        # Belief about probability
        m1_prob = [sigmoid(m2) for m2 in self.history["m2"]]
        axes[5, 1].plot(trials, m1_prob, "r-", linewidth=2, label="Belief: sigmoid(m2)")
        if true_prob is not None:
            axes[5, 1].plot(trials, true_prob, "k-", linewidth=1, label="True p(x1=1)", alpha=0.7)
        axes[5, 1].set_title("Belief vs Truth")
        axes[5, 1].set_ylabel("Probability")
        axes[5, 1].set_xlabel("Trial k")
        axes[5, 1].set_ylim(0, 1)
        axes[5, 1].legend()
        axes[5, 1].grid(True, alpha=0.3)

        # Volatility measure (exp of m3)
        volatility = [np.exp(m3) for m3 in self.history["m3"]]
        axes[5, 2].plot(trials, volatility, "b-", linewidth=2, label="Volatility: exp(m3)")
        axes[5, 2].set_title("Volatility Measure")
        axes[5, 2].set_ylabel("exp(m3)")
        axes[5, 2].set_xlabel("Trial k")
        axes[5, 2].legend()
        axes[5, 2].grid(True, alpha=0.3)
        axes[5, 2].set_yscale("log")

        plt.tight_layout()
        return fig


def sigmoid(x):
    sigmoid_val = 1 / (1 + np.exp(-x))
    # The output of the sigmoid is clipped to [1e-10, 1-1e-10] to prevent numerical issues in 1 / s1_hat!

    return np.clip(sigmoid_val, 1e-10, 1 - 1e-10)


def generate_reference_scenario():
    """Generate input sequence matching Figure 5."""
    np.random.seed(42)

    # Stage 1: 100 trials with p=0.5
    p1 = 0.5
    stage1 = np.random.rand(100) < p1
    true_prob1 = np.full(100, p1)

    # Stage 2: 120 trials with alternating high/low probability
    stage2 = []
    true_prob2 = []
    for i in range(6):
        p = 0.9 if i % 2 == 0 else 0.1
        stage2.extend(np.random.rand(20) < p)
        true_prob2.extend([p] * 20)

    # Stage 3: Repeat stage 1
    stage3 = np.random.rand(100) < p1
    true_prob3 = np.full(100, p1)

    inputs = np.concatenate([stage1, stage2, stage3]).astype(int)
    true_prob = np.concatenate([true_prob1, true_prob2, true_prob3])

    return inputs, true_prob


if __name__ == "__main__":
    # Create filter with reference parameters
    hgf = HGF(kappa=1.4, omega=-2.2, theta=0.5)

    # Generate inputs
    inputs, true_prob = generate_reference_scenario()

    hgf.run_simulation(inputs)

    # Plot original results
    # fig1 = hgf.plot_results(true_prob=true_prob)
    # plt.savefig("hgf_reference_scenario.png", dpi=150, bbox_inches="tight")

    # Plot all variables
    fig2 = hgf.plot_all_variables(true_prob=true_prob)
    plt.savefig("hgf_all_variables.png", dpi=150, bbox_inches="tight")

    plt.show()

    sum = 41
