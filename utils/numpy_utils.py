import numpy as np
import sys


def add_at_center(source, kernel, center_point):
    # get dimensions
    w, h, _ = source.shape
    w, h = w - 1, h - 1
    kernel_size = kernel.shape[0]
    half_size = kernel_size // 2

    # pre-calculate common values
    x_minus = center_point[0] - half_size
    y_minus = center_point[1] - half_size
    x_plus = center_point[0] + half_size
    y_plus = center_point[1] + half_size

    # calculate filter edge leaks
    xsd = -min(x_minus, 0)
    ysd = -min(y_minus, 0)
    xfd = -min(w - x_plus, 0)
    yfd = -min(h - y_plus, 0)

    check = np.array([xsd, ysd, xfd, yfd])
    if sum(check > half_size) > 0:
        return

    # calculate map start & end positions
    xmsp = x_minus + xsd
    ymsp = y_minus + ysd
    xmfp = x_plus - xfd + 1
    ymfp = y_plus - yfd + 1

    # kernel end points
    xkfp = kernel_size - xfd
    ykfp = kernel_size - yfd

    # set to max
    source[ymsp:ymfp, xmsp:xmfp, 0] = np.maximum(kernel[ysd:ykfp, xsd:xkfp], source[ymsp:ymfp, xmsp:xmfp, 0])


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.round(np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2), 2)


def debug(*variables):
    for variable in variables:
        print(variable, '=', repr(eval(variable)))
