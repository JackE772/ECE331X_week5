
import numpy as np
import matplotlib.pyplot as plt


# Pluto SDR Configuration

FS = 4_000_000
CENTER_FREQ = 2_402_000_000
RX_BANDWIDTH = 2_000_000
BUFFER_SIZE = 8192

# QPSK Parameters

SYMBOL_RATE = 1_000_000
BITS_PER_SYMBOL = 2
SAMPLES_PER_SYMBOL = FS // SYMBOL_RATE

AMPLITUDE = 0.5
NOISE_LEVEL = 0.0

# Carrier frequency offset (baseband)
FREQ_OFFSET = 5_000#25_000

rng = np.random.default_rng(420)

# Generate Random Data

# Generate enough bits to fill the buffer
num_symbols = BUFFER_SIZE // SAMPLES_PER_SYMBOL
num_bits = num_symbols * BITS_PER_SYMBOL

bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

# Group bits into pairs
bit_pairs = bits.reshape(-1, 2)

# QPSK Modulation

# Gray-coded QPSK mapping:
#
# 00 ->  45 degrees
# 01 -> 135 degrees
# 11 -> 225 degrees
# 10 -> 315 degrees

qpsk_symbols = (
    (1 - 2 * bit_pairs[:, 1].astype(float))
    + 1j * (1 - 2 * bit_pairs[:, 0].astype(float))
) / np.sqrt(2)

# Upsample: hold each symbol for 4 samples
iq = np.repeat(qpsk_symbols, SAMPLES_PER_SYMBOL)

# Scale amplitude
iq *= AMPLITUDE

# Add Carrier Frequency Offset

t = np.arange(len(iq)) / FS

iq *= np.exp(
    1j * 2 * np.pi * FREQ_OFFSET * t
)

# Add AWGN Noise

noise = NOISE_LEVEL * (
    rng.normal(size=len(iq))
    + 1j * rng.normal(size=len(iq))
) / np.sqrt(2)

iq += noise

# Convert to Pluto Format

iq = iq.astype(np.complex64)

# Keep exactly one Pluto buffer
iq = iq[:BUFFER_SIZE]

# Plot Results

fig, ax = plt.subplots(4, 1, figsize=(12, 12))

# I and Q versus time
ax[0].plot(iq.real[:500], label="I")
ax[0].plot(iq.imag[:500], label="Q")

ax[0].set_title("QPSK IQ Samples")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid()

# Unwrapped phase
ax[1].plot(np.unwrap(np.angle(iq[:500])))

ax[1].set_title("Unwrapped IQ Phase")
ax[1].set_ylabel("Phase (rad)")
ax[1].grid()

# Constellation sampled once per symbol
symbol_samples = iq[::SAMPLES_PER_SYMBOL]

ax[2].scatter(
    symbol_samples.real,
    symbol_samples.imag,
    s=10
)

ax[2].set_title("QPSK Constellation")
ax[2].set_xlabel("I")
ax[2].set_ylabel("Q")
ax[2].axis("equal")
ax[2].grid()

# Original transmitted bits
ax[3].step(
    np.arange(40),
    bits[:40],
    where="post"
)

ax[3].set_title("Transmitted Bits")
ax[3].set_xlabel("Bit Index")
ax[3].set_ylabel("Bit")
ax[3].set_yticks([0, 1])
ax[3].grid()

plt.tight_layout()
plt.show()

# Save IQ Data

np.save("qpsk_iq.npy", iq)

print("Saved qpsk_iq.npy")
print("Sample rate:", FS)
print("Symbol rate:", SYMBOL_RATE)
print("Samples per symbol:", SAMPLES_PER_SYMBOL)
print("Carrier frequency offset:", FREQ_OFFSET, "Hz")
print("Number of bits:", len(bits))
print("Number of IQ samples:", len(iq))
