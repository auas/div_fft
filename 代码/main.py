# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from fft import *
import datetime,time
import numpy as np
import scipy.misc
from hist import histeq,show_hist


#读取图片并转为数组

#test_fft_1d()


# print(pow(2.02,3.01))

# exit(0)

now = time.strftime("%Y-%m-%d %H:%M:%S")
print(now)
print("get started~ \n")
im = array(Image.open("C://alexandria_city.jpg").convert('L'))

#
# img_p = Image.open("C://alexandria_city.jpg", 'r')
# img_p = np.array(img_p.convert('L'))
# scipy.misc.imsave(im1, f_img, )
# exit(0)
# im = array(Image.open("C://hidden.jpg").convert('L'))
l,w = im.shape

im1 = padding(im,l,w)



data_dir = 'C:/Users/auas/result_jf/'
end = '.jpg'



for it in range(21):
    print("iter: {0}".format(it))
    path = data_dir + str(it)+ end
    # f_img = np.asarray(im1, dtype='float32')
    # scipy.misc.imsave(path, im1, )
    # exit(0)
    test_norm_div(im1,-it/5.0,path)

# test_norm_cdt(im1)
# hid_img = test_hidden(im1,1)
#test_re_hidden(hid_img,1)
#test_eq(im1)
#test_shift(im1)

