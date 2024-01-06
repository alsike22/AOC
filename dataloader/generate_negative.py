import numpy as np
import random

def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


# Add collective global outliers to original data
def point_global_outliers(train_x):
    for i, x_i in enumerate(train_x):
        position = int(np.random.rand() * train_x.shape[1])
        local_std = x_i[:, 0].std()
        scale = local_std * np.random.choice((-1, 1)) * 3 * (np.random.rand()+3)
        train_x[i, position, 0] = x_i[position, 0] + scale
    return train_x


# Add collective trend outliers to original data
def collective_trend_outliers(train_x, trend_rate):
    for i, x_i in enumerate(train_x):
        radius = max(10, int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand() * trend_rate
        position = int(np.random.rand() * (train_x.shape[1] - radius))
        start, end = position, position + radius
        slope = np.random.choice([-1, 1]) * factor * 0.1 * np.arange(end - start)
        train_x[i, start:end, 0] = x_i[start:end, 0] + slope
    return train_x


# Add collective trend outliers to original data
def collective_seasonal_outliers(train_x):
    seasonal_config = {'length': 400, 'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05}
    for i, x_i in enumerate(train_x):
        radius = max(10, int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand()
        seasonal_config['freq'] = factor * seasonal_config['freq']
        position = int(np.random.rand() * (train_x.shape[1] - radius))
        start, end = position, position + radius
        train_x[i, start:end, 0] = sine(**seasonal_config)[start:end]
    return train_x


# Add outliers to original data via the CutPaste method
# see details: Paper "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
def cut_paste_outliers(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(10, int(np.random.rand() * (train_x.shape[1]-2)))
        cut_data = x_i
        position = random.sample(range(0, train_x.shape[1] - radius + 1), 2)
        from_position = position[0]
        to_position = position[1]
        cut_data = cut_data[from_position:from_position + radius, 0]
        train_x[i, to_position:to_position + radius, 0] = cut_data[:]
    return train_x


# Add outliers to original data via the CutPaste method (Variant)
# see details: Paper "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
def cut_paste_outlier1(train_x):
    for i, x_i in enumerate(train_x):
        factor = np.random.rand()
        radius = max(10, int(np.random.rand() * train_x.shape[1] * factor))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(random.uniform(0, train_x.shape[1] - radius))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[from_position:from_position + radius, 0]
    return train_x


# Add outliers to original data via our CutAddPaste method
def cut_add_paste_outlier(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(10, int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand()
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
        cut_data = cut_data[from_position:from_position + radius, 0] + slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[:]
    return train_x


# our CutAddPaste method (Variant)
def cut_add_paste_outlier1(train_x):
    for i, x_i in enumerate(train_x):
        factor = np.random.rand()
        radius = max(10, int(np.random.rand() * train_x.shape[1] * factor))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        cut_data = cut_data[::-1, :]
        from_position = int(random.uniform(0, train_x.shape[1] - radius))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[from_position:from_position + radius, 0]
    return train_x


# our CutAddPaste method (Variant)
def cut_add_paste_outlier2(train_x):
    for i, x_i in enumerate(train_x):
        factor = np.random.rand()
        radius = max(10, int(np.random.rand() * train_x.shape[1] * factor))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        #  Outputs numbers in reverse order
        cut_data = cut_data[::-1, :]
        sample_list = [j for j in range(train_x.shape[1])]
        # down sampling
        sample_list = random.sample(sample_list, radius)
        cut_data = cut_data[sample_list, :]
        # from_position = int(random.uniform(0, train_x.shape[1] - radius))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x


# our CutAddPaste method (Variant)
def cut_add_paste_outlier3(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(10, int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand()
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        cut_data = cut_data[::-1, :]
        from_position = int(random.uniform(0, train_x.shape[1] - radius))
        slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
        cut_data = cut_data[from_position:from_position + radius, 0] + slope
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[:]
    return train_x


def cut_add_paste_outlier4(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = min(int(3 * train_x.shape[1]/4), radius)
        factor = np.random.rand()
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
        cut_data = cut_data[from_position:from_position + radius, 0] + slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[:]
    return train_x


def cut_add_paste_outlier5(train_x, trend_rate):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand() * trend_rate
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        slope = np.random.choice([-1, 1]) * factor * 0.1 * np.arange(radius)
        cut_data = cut_data[from_position:from_position + radius, 0] + slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, 0] = cut_data[:]
    return train_x
