from math import e as _e


def sigmoid(value):
    return 1/(1+_e**-value)


def scale_values(values):
    max_value = max(values)
    return [v/max_value for v in values]
