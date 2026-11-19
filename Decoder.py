import matplotlib.pyplot as plt
import numpy as np

#config
FS = 4_000_000
FC = 2_402_000_000
BIT_RATE = 1_000_000
SAMPLES_PER_BIT = FS // BIT_RATE
FREQ_DEVIATION = 250_000

def costas_loop_bpsk(iq, fs, fc, loop_bandwidth=0.1, damping=0.707):
    """
    BPSK Costas loop.

    Parameters:
        iq: Complex IQ input samples
        fs: Sample rate (Hz)
        fc: Expected carrier frequency (Hz), relative to IQ baseband
        loop_bandwidth: Normalized loop bandwidth
        damping: Loop damping factor

    Returns:
        recovered: Carrier-corrected complex IQ
        phase: Estimated carrier phase
        frequency: Estimated carrier frequency
        error: Costas loop phase error
    """

    # Second-order loop filter coefficients
    theta = loop_bandwidth / (
        damping + 1 / (4 * damping)
    )

    d = 1 + 2 * damping * theta + theta**2

    alpha = (4 * damping * theta) / d
    beta = (4 * theta**2) / d

    # Loop state
    phase = 0.0
    freq = 2 * np.pi * fc / fs

    recovered = np.zeros(len(iq), dtype=np.complex128)
    phase_history = np.zeros(len(iq))
    freq_history = np.zeros(len(iq))
    error_history = np.zeros(len(iq))

    for i, sample in enumerate(iq):

        # Remove estimated carrier
        corrected = sample * np.exp(-1j * phase)

        I = np.real(corrected)
        Q = np.imag(corrected)

        # BPSK Costas phase detector
        error = np.sign(I) * Q - np.sign(Q) * I

        # Update loop filter
        freq += beta * error
        phase += freq + alpha * error

        # Keep phase bounded
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

        # Store results
        recovered[i] = corrected
        phase_history[i] = phase
        freq_history[i] = freq * fs / (2 * np.pi)
        error_history[i] = error

    #convert to np array (low key should have done before but I dont want to rewrite)
    recovered = np.array(recovered)
    phase_history = np.array(phase_history)
    freq_history = np.array(freq_history)
    error_history = np.array(error_history)

    return {"recoverd":recovered, "phase_history":phase_history, "freq_history":freq_history, "error_history":error_history}

def plot_results(iq, result):

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.scatter(np.real(iq), np.imag(iq))
    plt.title("input IQ signal")
    plt.subplot(2,1,2)
    plt.scatter(np.real(result["recoverd"]), np.imag(result["recoverd"]))
    plt.title("costas loop prossesed IQ data")
    plt.show()


if __name__ == "__main__":

    #iq = np.load("ble_gfsk_iq.npy")
    iq = np.load("qpsk_iq.npy")

    result = costas_loop_bpsk(iq, FS, FC)

    if result is not None:
        plot_results(iq, result)