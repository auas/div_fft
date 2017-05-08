import numpy as np
from fft import *
a = [1.,2.,3.,4.]

# test_fft(a)
# b = [20.0,-6.30844,0.0,0.4483]
# test_fft(a)
# exit(0)
# print(a[:5])
# exit(0)

# a1 = dct(a[:4])
# for it in a1:
#     print(it)
# print ("***********************")


a2 = cdt_1d(a[:4],4)
for it in a2:
    print(it)
print ("***********************")
a3 = re_cdt_1d(a2,4)
for it in a3:
    print(it)
# exit(0)
# a2 = idct(a1)
# print(a2)
#
# im = cdt_1d(a,len(a))
# for it in im:
#     print(it)

#
# re = re_cdt_1d(a1,4)
# for it in re:
#     print(it)

# t