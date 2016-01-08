from math import e as _e


def sigmoid(value):
    try:
        return 1/(1+_e**-value)
    except OverflowError:
        return 0


def scale_values(values):
    max_value = float(max(values))
    return [v/max_value for v in values]
