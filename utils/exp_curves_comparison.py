import numpy as np


def fit_exp(x, y):
    """
    Fit y ≈ A * r^x via log-space linear regression.
    Returns A, r.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError("All y values must be positive to take log.")

    z = np.log(y)
    X = np.vstack([np.ones_like(x), x]).T
    b0, b1 = np.linalg.lstsq(X, z, rcond=None)[0]

    A = np.exp(b0)
    r = np.exp(b1)
    return A, r


def prob_mass_center_from_params(x, A, r):
    """
    Compute the probability mass center (discrete center of mass in x)
    for the fitted exponential y_hat(x) = A * r^x over the given x-grid.

    The y_hat values are normalized to sum to 1 and treated as a PMF.
    """
    x = np.asarray(x, dtype=float)

    y_hat = A * (r**x)  # fitted curve
    y_hat = np.clip(y_hat, a_min=0, a_max=None)

    total = y_hat.sum()
    if total == 0:
        raise ValueError("Total mass is zero; cannot compute center.")

    p = y_hat / total  # normalize to a probability mass
    x_center = np.sum(x * p)  # expectation of x

    return x_center


# Example with your two datasets
if __name__ == "__main__":
    x1 = np.array([1, 2, 3])
    y1 = np.array([32, 16, 8])

    x2 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([32, 16, 8, 4, 2])

    A1, r1 = fit_exp(x1, y1)
    A2, r2 = fit_exp(x2, y2)

    cm1 = prob_mass_center_from_params(x1, A1, r1)
    cm2 = prob_mass_center_from_params(x2, A2, r2)

    print("Set 1: A =", A1, " r =", r1, " center =", cm1)
    print("Set 2: A =", A2, " r =", r2, " center =", cm2)
