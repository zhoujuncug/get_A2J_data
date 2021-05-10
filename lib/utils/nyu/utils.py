def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x