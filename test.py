import numpy as np
# array2freqs = np.arange(1, 10)
# repeat_array = np.repeat(array2freqs, 2)
# array11 = np.concatenate(([1, 2], repeat_array), 0)
# print(array2freqs)
# print(repeat_array)
# print(array11)

# base_array = np.repeat(np.arange(1, 10), 2)
# base_freqs = np.concatenate(([1, 2], base_array), 0)
# freq = np.concatenate((base_freqs, 2*base_freqs, 4*base_freqs, 8*base_freqs), axis=0)
# print(freq)

base_freqs = np.arange(1, 11)
base_freqs = np.concatenate(([1, 2, 3, 4, 5], base_freqs), 0)
high_freqs = np.arange(91, 100)
freq = np.concatenate(([1], base_freqs, 2*base_freqs, 4*base_freqs, 6*base_freqs, 8*base_freqs, 10*base_freqs, high_freqs), axis=0)
print(freq)
print(len(freq))