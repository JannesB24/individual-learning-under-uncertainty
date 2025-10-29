import numpy as np


class HGF:
    """Hierarchical Gaussian Filter"""

    def __init__(self, kappa, omega, theta):
        # Store parameters
        self.kappa = kappa
        self.omega = omega
        self.theta = theta

        # Initialize states - see Simulation on page 11
        self.m2 = 0.0
        self.m3 = 0.0
        self.s2 = 1.0
        self.s3 = 1.0

    def update(self, u):
        """Update beliefs given binary input u (0 or 1)"""

        # STEP 1: Make predictions (using previous trial's values)
        # TODO: Equations 24, 26, 27, 31

        # STEP 2: Update Level 1
        mu_1 = float(u)

        # STEP 3: Compute prediction error at Level 1
        # TODO: Equation 25

        # STEP 4: Update Level 2 (m2 and s2)
        # TODO: Equations 22, 23

        # STEP 5: Compute prediction error at Level 2
        # TODO: Equation 34 (and helpers 32, 33)

        # STEP 6: Update Level 3 (m3 and s3)
        # TODO: Equations 29, 30

        return self.m2, self.m3
