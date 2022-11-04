import numpy as np


def frac_BDF_coeff(p, k, al):
    #  compute the fractional BDF coefficients
    #  p: approximation order, p=1,2,...,6
    #  k: compute coeff up to terms k (i.e. k+1 coeffs)
    #  al: fractional order
    if p > 6:
        print(f'BDF is unstable for p>6')

    c = np.zeros(k+1)
    if p == 1:
        c[0] = 1
        for j in range(k):
            c[j+1] = -c[j]*(al-j)/(j+1)
    # if p == 2:
    #     c[0] = (3/2)**al
    #     c[1] = -4/3*al*c[0]
    #     for j in range(start=1,stop=k):
    #         c[j+1] = 2*(-2*(al-j+1)*c[j]+(2*al-j+2)*c[j-1]/2)/j/3
    # if p == 3:
    #     c[0] = (11/6)**al
    #     c[1] = -18/11*al*c[0]
    #     c[2] = 3/11*(-3*(al-1)*c[1]+3*al*c[0])
    #     for j in range(start=2,stop=k):
    #         c[j+1] = 6/11/j*(-3*(al-j+1)*c[j]+3/2*(2*al-j+2)*c[j-1]-1/3*(3*al-j+3)*c[j-2])
    return c




