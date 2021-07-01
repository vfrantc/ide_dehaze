import cv2
import scipy
import numpy as np
import scipy.ndimage.filters as nd_filters
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

import cProfile

def local_filter(x, order):
    x.sort()
    return x[order]

def ordfilt2(A, order, mask_size):
    return nd_filters.generic_filter(A, lambda x, ord=order: local_filter(x, ord), size=(mask_size, mask_size))


def GS(t_low, A_, Ave_a, uu, p1, p2, sourcePic, A_channel, m, n):
    a = np.log(t_low) * A_
    b = np.log(t_low) * Ave_a - A_ * uu * p1 + np.log(t_low) * A_ * p2 - np.log(t_low) * A_
    c = p2 * np.log(t_low) * Ave_a - np.log(t_low) * A_ * p2
    t = np.abs((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
    ILCC = (p1 / (t + p2)) / (p1 / (t_low + p2))

    J = np.zeros((sourcePic.shape[0], sourcePic.shape[1], 3), dtype=np.float64)
    J[:, :, 0] = (sourcePic[:, :, 0] - A_channel[0] * (1 - t)) / (t * A_channel[0] * ILCC)
    J[:, :, 1] = (sourcePic[:, :, 1] - A_channel[1] * (1 - t)) / (t * A_channel[1] * ILCC)
    J[:, :, 2] = (sourcePic[:, :, 2] - A_channel[2] * (1 - t)) / (t * A_channel[2] * ILCC)

    J = np.minimum(np.maximum(J, 0), 1)
    SA = (np.sum(J.ravel() == 0) + np.sum(J.ravel() == 1)) / (m * n * 3)
    return SA

def airlight_our(image, wsz):
  A = []
  for i in range(3):
      min_image = ordfilt2(image[:, :, i], 1, wsz)
      A.append(min_image.max())
  return np.array(A, dtype=np.float64)


def dehaze(sourcePic):
    m, n, o = sourcePic.shape[:3]

    Graymean = 0.5
    p1 = -0.397
    p2 = 0.07774
    Patch_size_GWA = 75
    T = 0.02

    A_channel = airlight_our(sourcePic, 10)
    A_ = np.mean(A_channel)

    Kernel = np.ones((Patch_size_GWA, Patch_size_GWA), dtype=np.float64)
    I_mean = sourcePic[:, :, 0] + sourcePic[:, :, 1] + sourcePic[:,:,2]
    I_locolmean = scipy.ndimage.correlate(I_mean, Kernel, mode='reflect') / (Kernel.sum() * 3)

    a = 0.001
    b = 1.0
    Theta_error1 = 0.001
    Theta_error2 = 0.001
    N = 1
    r = 0.618
    a1 = b - r * (b - a)
    a2 = a + r * (b - a)
    stepNum = 0

    I_locolmean_downsample = I_locolmean.copy()
    sourcePic_downsample = sourcePic.copy()
    m_downsample, n_downsample = I_locolmean_downsample.shape

    while abs(b - a) > Theta_error1:
        stepNum = stepNum + 1
        f1 = GS(a1, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
                n_downsample)
        f2 = GS(a2, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
                n_downsample)
        if f1 > f2:
            a = a1
            f1 = f2
            a1 = a2
            a2 = a + r * (b - a)
        else:
            b = a2
            a2 = a1
            f2 = f1
            a1 = b - r * (b - a)
        t_low_MIN = (a1 + a2) / 2

    Z = GS(t_low_MIN, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample, n_downsample)
    Bound_low = 0.01
    Bound_high = t_low_MIN
    interval = Bound_high - Bound_low
    StepNUM2 = 1

    while interval > Theta_error2:
        cen = Bound_high / 2 + Bound_low / 2
        if (GS(Bound_low, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
               n) - Z - T) * (
                GS(cen, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
                   n_downsample) - Z - T) < 0:
            Bound_high = cen
        elif (GS(Bound_low, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
                 n_downsample) - Z - T) * (
                GS(cen, A_, I_locolmean_downsample, Graymean, p1, p2, sourcePic_downsample, A_channel, m_downsample,
                   n_downsample) - Z - T) > 0:
            Bound_low = cen
        else:
            Bound_high = cen
            Bound_low = cen

        interval = interval / 2
        StepNUM2 = StepNUM2 + 1

    t_low_determined = Bound_high / 2 + Bound_low / 2 + 0.02

    A = np.log(t_low_determined) * A_
    B = np.log(t_low_determined) * I_locolmean - A_ * Graymean * p1 + np.log(t_low_determined) * A_ * p2 - np.log(t_low_determined) * A_
    C = p2 * np.log(t_low_determined) * I_locolmean - np.log(t_low_determined) * A_ * p2
    t = abs((-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A))

    src = (sourcePic[:, :, 2]*255).astype(np.uint8)
    t_uint8 = (t*255).astype(np.uint8)
    t_refine = guidedFilter(src, t_uint8, 50, 0.02).astype(np.float64) / 255
    t_refine = 1.0 * np.maximum(t_refine, t_low_determined)

    ILCC = 1 * (p1 / (t_refine + p2)) / (p1 / (t_low_determined + p2))

    J = np.zeros((sourcePic.shape[0], sourcePic.shape[1], 3), np.float64)
    J[:, :, 0] = ((sourcePic[:, :, 0] - A_channel[0] * (1 - t_refine)) / t_refine / A_channel[0] / ILCC)
    J[:, :, 1] = ((sourcePic[:, :, 1] - A_channel[1] * (1 - t_refine)) / t_refine / A_channel[1] / ILCC)
    J[:, :, 2] = ((sourcePic[:, :, 2] - A_channel[2] * (1 - t_refine)) / t_refine / A_channel[2] / ILCC)
    return J


if __name__ == '__main__':
    sourcePic = cv2.imread('aerial_input.png', 1)
    sourcePic = cv2.cvtColor(sourcePic, cv2.COLOR_BGR2RGB)
    sourcePic = sourcePic.astype(np.float64) / 256
    #cProfile.run('dehaze(sourcePic)')
    J = dehaze(sourcePic)
    plt.figure()
    plt.imshow(sourcePic)
    plt.figure()
    plt.imshow(J)
    plt.show()