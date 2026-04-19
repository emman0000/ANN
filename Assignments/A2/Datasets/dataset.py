import numpy as np

# ----------------------------
# Function from paper
# ----------------------------
def target_function(x, y, t, z):
    return (
        np.exp(-0.5 * x)
        + np.log(1 + np.exp(0.4 * y))
        + np.tanh(t)
        + np.sin(z)
        - 0.4
    )


# ----------------------------
# Dataset generator
# ----------------------------
def generate_dataset(n_samples, low, high):
    # Sample inputs
    x = np.random.uniform(low, high, n_samples)
    y = np.random.uniform(low, high, n_samples)
    t = np.random.uniform(low, high, n_samples)
    z = np.random.uniform(low, high, n_samples)

    # Stack into input matrix
    X = np.stack([x, y, t, z], axis=1)

    # Compute outputs
    Y = target_function(x, y, t, z)

    return X, Y


# ----------------------------
# Generate datasets
# ----------------------------
def main():
    # Training data
    X_train, y_train = generate_dataset(
        n_samples=500,
        low=0.0,
        high=4.0
    )

    # Testing data
    X_test, y_test = generate_dataset(
        n_samples=5000,
        low=0.0,
        high=6.0
    )

    # Save datasets
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print("Datasets generated and saved!")

if __name__ == "__main__":
    main()
