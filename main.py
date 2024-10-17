import numpy as np
import matplotlib.pyplot as plt


def lms_filter(desired, input_signal, mu=0.01, filter_order=32):
    """
    Implements an LMS (Least Mean Squares) adaptive filter to minimize the difference
    between a desired signal and an input signal with noise. This filter adjusts its
    weights iteratively to adaptively filter out noise from the input signal.

    Args:
        desired (numpy.ndarray): The desired signal, typically the original clean signal,
            used as the reference for the adaptive filter.
        input_signal (numpy.ndarray): The input signal, which is a noisy version of
            the desired signal that needs to be filtered.
        mu (float, optional): The step size (learning rate) that controls the speed of
            convergence of the filter. A small mu results in slow convergence, while
            a large mu can lead to instability. Defaults to 0.01.
        filter_order (int, optional): The number of filter taps (coefficients),
            which determines how many past input samples are used to update the filter.
            Defaults to 32.

    Returns:
        output (numpy.ndarray): The filtered signal output, which should closely
            approximate the desired signal if the filter adapts successfully.
        error (numpy.ndarray): The error signal at each iteration, representing
            the difference between the desired signal and the filtered output.
    """

    n_iterations = len(input_signal)  # Total number of samples in the input signal
    weights = np.zeros(filter_order)  # Array of filter coefficients adapted during LMS
    output = np.zeros(n_iterations)  # Filtered output signal
    error = np.zeros(n_iterations)  # Difference between desired and output

    # It takes at least filter_order samples to perform filtering
    for n in range(filter_order, n_iterations):
        # Get most recent filter_order samples
        x = input_signal[n - filter_order : n][::-1]
        y = np.dot(weights, x)  # Output calculation
        error[n] = desired[n] - y  # Error calculation

        # w(n+1) = w(n) + * mu * e(n) * x(n)
        weights = weights + mu * error[n] * x  # Update weights
        output[n] = y  # Save output

    return output, error


def nlms_filter(desired, input_signal, mu=0.01, filter_order=32, epsilon=1e-6):
    """
    Implements a Normalized LMS (NLMS) adaptive filter.

    Args:
        desired (numpy.ndarray): The desired signal, typically the original clean signal,
            used as the reference for the adaptive filter.
        input_signal (numpy.ndarray): The input signal, which is a noisy version of
            the desired signal that needs to be filtered.
        mu (float, optional): The step size or learning rate that controls the speed of
            convergence of the filter. Defaults to 0.01.
        filter_order (int, optional): The number of filter taps (coefficients),
            which determines how many past input samples are used to update the filter.
            Defaults to 32.
        epsilon (float, optional): A small constant added to avoid division by zero in the
            normalization term. Defaults to 1e-6.

    Returns:
        output (numpy.ndarray): The filtered signal output, which should closely
            approximate the desired signal if the filter adapts successfully.
        error (numpy.ndarray): The error signal at each iteration, representing
            the difference between the desired signal and the filtered output.
    """
    n_iterations = len(input_signal)
    weights = np.zeros(filter_order)
    output = np.zeros(n_iterations)
    error = np.zeros(n_iterations)

    for n in range(filter_order, n_iterations):
        x = input_signal[n : n - filter_order : -1]
        y = np.dot(weights, x)
        error[n] = desired[n] - y

        # Normalization term: power of the input vector
        norm_factor = np.dot(x, x) + epsilon
        # Update the filter weights using the normalized LMS rule
        weights = weights + (mu / norm_factor) * error[n] * x

        output[n] = y

    return output, error


if __name__ == "__main__":
    # Simulating the desired signal and input (signal + noise)
    np.random.seed(0)
    n_samples = 500
    signal = np.sin(0.05 * np.arange(n_samples))  # Desired clean signal
    noise = 0.5 * np.random.randn(n_samples)  # Noise
    input_signal = signal + noise  # Noisy signal

    # Apply LMS filter
    lms_output, lms_error = lms_filter(signal, input_signal)
    nlms_output, nlms_error = nlms_filter(signal, input_signal)

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="Desired Signal")
    plt.plot(input_signal, label="Noisy Signal")
    plt.plot(lms_output, label="LMS Filtered Output")
    plt.plot(nlms_output, label="NLMS Filtered Output")
    plt.legend()
    plt.show()


def rls_filter(desired, input_signal, filter_order=32, lambda_factor=0.99, delta=1e-3):
    """
    Implements a Recursive Least Squares (RLS) adaptive filter.

    Args:
        desired (numpy.ndarray): The desired signal, typically the clean signal,
            used as the reference for the adaptive filter.
        input_signal (numpy.ndarray): The input signal, which is a noisy version
            of the desired signal that needs to be filtered.
        filter_order (int, optional): The number of filter taps (coefficients).
            Defaults to 32.
        lambda_factor (float, optional): The forgetting factor, which controls
            how much weight past data carries. A value close to 1 means older data
            still has significant influence. Defaults to 0.99.
        delta (float, optional): Initialization value for the inverse of the
            autocorrelation matrix. Defaults to 1e-3.

    Returns:
        output (numpy.ndarray): The filtered signal output, which should closely
            approximate the desired signal if the filter adapts successfully.
        error (numpy.ndarray): The error signal at each iteration, representing
            the difference between the desired signal and the filtered output.
    """
    n_iterations = len(input_signal)
    weights = np.zeros(filter_order)  # Initialize filter weights
    P = (
        np.eye(filter_order) / delta
    )  # Inverse autocorrelation matrix, initialized with delta
    output = np.zeros(n_iterations)  # Filtered output
    error = np.zeros(n_iterations)  # Error signal

    for n in range(filter_order, n_iterations):
        # Input vector (slice of input signal)
        x = input_signal[n : n - filter_order : -1]

        # Gain vector
        pi = P @ x
        gain_vector = pi / (lambda_factor + x.T @ pi)

        # Filter output
        y = np.dot(weights, x)

        # Error signal (difference between desired and filter output)
        error[n] = desired[n] - y

        # Update weights
        weights = weights + gain_vector * error[n]

        # Update the inverse autocorrelation matrix
        P = (P - np.outer(gain_vector, pi)) / lambda_factor

        # Store the filter output
        output[n] = y

    return output, error
