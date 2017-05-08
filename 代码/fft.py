import math
import numpy as np
import sys
from PIL import Image
from pylab import *
from hist import histeq,show_hist
import datetime,time
import scipy.misc
#glob :
anas_len = 256
anas_wid = 256
PI = 3.1415926
class complex:
    def __init__(self):
        self.real = 0.0
        self.image = 0.0

def copy_mat(src):
    ret = []
    w,l = src.shape
    for i in range(l):
        ret.append(src[i,:])
    ret = np.array(ret)
    return ret


def mul_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real * complex1.real - complex0.image * complex1.image
    complex_ret.image = complex0.real * complex1.image + complex0.image * complex1.real
    return complex_ret

def check(arr):
    for it in arr:
        print(it)
def add_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real + complex1.real
    complex_ret.image = complex0.image + complex1.image
    return complex_ret


def sub_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real - complex1.real
    complex_ret.image = complex0.image - complex1.image
    return complex_ret

def mul_en(complex0, num):
    complex_ret = complex()
    complex_ret.real = complex0.real*num
    complex_ret.image = complex0.image*num
    return complex_ret

def div_en(complex0, num):
    complex_ret = complex()
    complex_ret.real = complex0.real/num
    complex_ret.image = complex0.image/num
    #print("complex0.real = {0}   complex_ret.real = {1}".format(complex0.real,complex_ret.real))#
    return complex_ret

def forward_input_data(input_data, num):
    j = num / 2
    for i in range(1, num - 2):
        if (i < j):
            complex_tmp = input_data[i]
            #print("err")
            #print(num)
            input_data[i] = input_data[j]
            input_data[j] = complex_tmp
            #print("forward x[%d] <==> x[%d]" % (i, j))
        k = num / 2
        while (j >= k):
            j = j - k
            k = k / 2
        j = j + k


def fft_1d(in_data, num):
    PI = 3.1415926
    forward_input_data(in_data, num)

    M = 1  # num = 2^m
    tmp = num / 2
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2

    complex_ret = complex()
    for L in range(1, M + 1):
        B = int(math.pow(2, L - 1))
        for J in range(0, B):
            P = math.pow(2, M - L) * J
            for K in range(J, num, int(math.pow(2, L))):
                #print "L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P)
                complex_ret.real = math.cos((2 * PI / num) * P)
                complex_ret.image = -math.sin((2 * PI / num) * P)
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub

def re_fft_1d(in_data, num):
    tot = num
    PI = 3.1415926
    forward_input_data(in_data, num)

    M = 1  # num = 2^m
    tmp = num / 2
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2

    complex_ret = complex()
    for L in range(1, M + 1):
        B = int(math.pow(2, L - 1))
        for J in range(0, B):
            P = math.pow(2, M - L) * J
            for K in range(J, num, int(math.pow(2, L))):
                #print "L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P)
                complex_ret.real = math.cos((-2 * PI / num) * P)
                complex_ret.image = -math.sin((-2 * PI / num) * P)
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub
    max = len(in_data)
    for it in range(max):
        in_data[it] =div_en(in_data[it],num)
                #print("num={0}".format(num))

def padding(data,l,w):
    pad = data[0][0]
    if l>anas_len:
        print("data.len too large {0}".format(l))
    if w>anas_wid:
        print("data.wid too large {0}".format(w))
    fl = (int)(anas_len-l)/2
    fw = (int)(anas_wid-w)/2
    ret = np.ones([anas_len,anas_wid])*pad
    for i in range(l):
        for j in range(w):
            ret[fl+i,fw+j] = data[i,j]
    return ret

def shift(data,perc):

    l,w = data.shape
    ret = data[0, 0] * np.ones([l,w])
    x = (int)(l*perc)
    y = (int)(w*perc)

    for i in range(l-x):
        for j in range(w-y):
            ret[i,j] = data[i+x,j+y]
    return ret

def shift_x(data,perc):

    l,w = data.shape
    ret = data[0, 0] * np.ones([l,w])
    x = (int)(l*perc)
    y = (int)(w*perc)

    for i in range(l-x):
        for j in range(w):
            ret[i,j] = data[i+x,j]
    return ret

def relocate(x,y,theta):
    return [x*math.sin(theta)-y*math.cos(theta),y*math.sin(theta)+x*math.cos(theta)]

def rotate(data,theta):
    l,w = data.shape
    pl = data[0,0]
    ret = data[0,0]*np.ones([l,w])
    ml = (int)(l/2)
    mw = (int)(w/2)
    for i in range(l):
        for j in range(w):
            if data[i,j]==pl:
                continue
            [x,y] = relocate(i-ml,j-mw,theta)
            xx = (int)(x+ml)
            yy = (int)(y+mw)
            if xx>0 and xx<l and yy>0 and yy<w:
                ret[xx,yy] = data[i,j]
    return ret


def real2img(data):
    ret = []
    l,w = data.shape
    for i in range(l):
        row = []
        for j in range(w):
            c = complex()
            c.real = data[i,j]
            row.append(c)
        ret.append(row)
    return np.array(ret)

def getReal(data):
    l, w = data.shape
    ret = np.zeros([l,w])
    for i in range(l):
        for j in range(w):
            ret[i,j]=data[i,j].real

    return ret

def getImg(data):
    l, w = data.shape
    ret = np.zeros([l,w])
    for i in range(l):
        for j in range(w):
            ret[i,j]=data[i,j].image
    return ret

def e_getReal(data):
    l, w = data.shape
    ret = np.zeros([l,w])
    for i in range(l):
        for j in range(w):
            ret[i,j]=data[i,j].real
    print(get_mm(ret))
    return map_255(ret)

def e_getImg(data):
    l, w = data.shape
    ret = np.zeros([l,w])
    for i in range(l):
        for j in range(w):
            ret[i,j]=data[i,j].image
    return map_255(ret)


def fft_2d(data):
    '''
    :param data: real mat
    :return:
    '''
    l,w=data.shape
    #print("in")
    #print(data.shape)
    img = real2img(data)
    for i in range(l):
        row = img[i,:]
        fft_1d(row,w)
        img[i,:] = row
    for j in range(w):
        col = img[:,j]
        fft_1d(col,l)
        img[:,j] = col
    return img

def refft_2d(data):
    '''
    :param data: complex mat
    :return:
    '''
    l,w=data.shape
    ret = copy_mat(data)
    for j in range(l):
        col = ret[:,j]
        re_fft_1d(col,w)
        ret[:,j]=col
    for i in range(l):
        row = ret[i,:]
        re_fft_1d(row,w)
        ret[i,:]=row


    return ret

def test_fft_1d():
    in_data = [2, 3, 4, 5, 7, 9, 10, 11]
    data = [(complex()) for i in range(len(in_data))]
    for i in range(len(in_data)):
        data[i].real = in_data[i]
        data[i].image = 0.0
    for it in data:
        print("before::  real: {0} ; image: {1}".format(it.real,it.image))
    fft_1d(data, 8)
    for it in data:
        print("after::  real: {0} ; image: {1}".format(it.real,it.image))
    re_fft_1d(data, 8)
    for it in data:
        print("return::  real: {0} ; image: {1}".format(it.real, it.image))

def LB(data,type,prec,isCDT=0):
    '''
    :param data:  mat of complex
    :param type:  (0: high 1: low) to pass
    :param prec:  chose the area
    :return:
    '''
    ret = copy_mat(data)
    l,w = data.shape
    ll = l*prec
    ww = w*prec
    zer = complex()
    for i in range(l):
        for j in range(w):
            if i>ll and i<l-ll and j>ww and j<w-ww:
                if type==1:
                    if isCDT==0:
                        ret[i,j]=zer
                    else:
                        ret[i, j] = 0
            else:
                if type==0:
                    if isCDT==0:
                        ret[i,j]=zer
                    else:
                        ret[i, j] = 0
    return ret
def LB_side(data,type,prec,side):
    '''
    :param data:  mat of complex
    :param type:  (0: high 1: low) to pass
    :param prec:  chose the area
    :return:
    '''
    ret = copy_mat(data)
    l,w = data.shape
    ll = l*prec
    ww = w*prec
    zer = complex()
    if side == 'up_left':
        for i in range(l):

            for j in range(w):
                if i > ll and j > ww :
                    if type == 1:
                        ret[i, j] = zer
                else:
                    if type == 0:
                        ret[i, j] = zer
    else:
        for i in range(l):
            for j in range(w):
                if  i < l - ll and j < w - ww:
                    if type == 1:
                        ret[i, j] = zer
                else:
                    if type == 0:
                        ret[i, j] = zer
    return ret

def map_255(data):

    l,w=data.shape
    ret = np.zeros([l,w])

    max = -sys.maxint
    min = -max
    for i in range(l):
        for j in range(w):
            if data[i,j]>max:
                max = data[i,j]
            if data[i,j]<min:
                min = data[i,j]
    k = 255./(max-min)
    for i in range(l):
        for j in range(w):
            ret[i,j] = k*(data[i,j]-min)+min
    print([max,min])
    return ret

def com_mat(re,img):
    ret = []
    l,w = re.shape
    for i in range(l):
        row = []
        for j in range(w):
            c = complex()
            c.real = re[i,j]
            c.img = img[i,j]
            row.append(c)
        ret.append(row)
    return np.array(ret)

def test_norm(data):
    im_frq = fft_2d(data)

    # lb_img = LB(im_frq,0,0.01)
    lb_img = im_frq
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    f_img = refft_2d(lb_img)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish re~ \n")
    im_final = getReal(f_img)
    gray()
    imshow(im_final)
    show()
    return

def large_pad(data,prec): #data complex mat
    ll,ww = data.shape
    pll = (int)(ll*prec)
    pww = (int)(ww*prec)
    hf_l = (int)(pll/2)
    hf_w = (int)(pww/2)
    tot_l = pll+ll
    tot_w = pww+ww
    pre_ret = []
    for i in range(tot_l):
        row = []
        for j in range(tot_w):
            c = complex()
            row.append(c)
        pre_ret.append(row)
    ret = np.array(pre_ret)

    for i in range(ll):
        for j in range(ww):
            ret[i+hf_l,j+hf_w] = data[i,j]
    return ret

def small_pad(data,prec): # data : complex mat
    ll,ww = data.shape
    pll = (int)(ll/(1+prec))
    pww = (int)(ww/(1+prec))
    hf_l = (int)((ll-pll)/2)
    hf_w = (int)((ww-pww)/2)
    ret = []
    for i in range(ll):
        row = []
        if i>hf_l-1 and i<ll-hf_l:
            for j in range(ww):
                if j>hf_w-1 and j<ww-hf_w:
                    row.append(data[i,j])
                else:
                    continue
            ret.append(row)
        else:
            continue

    return np.array(ret)


def test_hidden(data,perc):
    im_frq = fft_2d(data)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    large = large_pad(im_frq,1)
    re_hid = refft_2d(large)
    print(now)
    print("finish hid re~ \n")
    re_hid_final = getReal(re_hid)
    gray()
    imshow(re_hid_final)
    show()
    return re_hid_final

def test_re_hidden(data,perc):
    im_frq = fft_2d(data)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    small = small_pad(im_frq,1)
    re_hid = refft_2d(small)
    print(now)
    print("finish hid re~ \n")
    re_hid_final = getReal(re_hid)
    gray()
    imshow(re_hid_final)
    show()
    return re_hid

def re_hidden(data,prec):
    return



def test_eq(data):
    im_frq = fft_2d(data)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    re_im_frq = getReal(im_frq)
    img_im_frq = getImg(im_frq)

    re, cdf_re = histeq(re_im_frq)
    img, cdf_img = histeq(img_im_frq)
    e_frq = com_mat(re, img)

    f_img = refft_2d(e_frq)
    im_final = e_getReal(f_img)

    gray()
    imshow(im_final)
    show()
    return
def get_mm(data):
    l, w = data.shape
    max = -sys.maxint
    min = -max
    for i in range(l):
        for j in range(w):
            if data[i, j] > max:
                max = data[i, j]
            if data[i, j] < min:
                min = data[i, j]
    return [max,min]

def dist(a,b):
    return (a-b)*(a-b)

def cout_diff(data1,data2):
    l,w = data1.shape
    ret = 0.
    e = data1-data2
    for i in range(l):
        for j in range(w):
            ret = ret+e[i,j]*e[i,j]
    return ret/(l*w)
def test_shift(data):
    e1 = 0
    e2 = 0



    data_shift = shift_x(data,0.1)
    e1 = cout_diff(data, data_shift)

    im_frq = fft_2d(data)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    frq_re = getReal(im_frq)
    frq_Img = getImg(im_frq)
    print(now)
    print("finish frq~ \n")

    im_frq_shift = fft_2d(data_shift)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    frq_re_shift = getReal(im_frq_shift)
    frq_Img_shift = getImg(im_frq_shift)
    print(now)
    print("finish frq~ \n")

    e2 = cout_diff(frq_re,frq_re_shift)
    e3 = cout_diff(frq_Img,frq_Img_shift)
    max1,min1 = get_mm(data)
    max2,min2 = get_mm(frq_re)
    max3,min3 = get_mm(frq_Img)
    print("e1={0}  e2={1}  e3={2}".format(e1/max1,e2/max2,e3/max3))

def show_I(a):
    print("real = {0}".format(a.real))
    print("image = {0}".format(a.image))
    print("\n")


def auas_fft_1d(in_data, num):
    PI = 3.1415926
    forward_input_data(in_data, num)

    M = 1  # num = 2^m
    tmp = num / 2
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2

    complex_ret = complex()
    for L in range(1, M + 1):
        B = int(math.pow(2, L - 1))
        for J in range(0, B):
            P = math.pow(2, M - L) * J
            for K in range(J, num, int(math.pow(2, L))):
                # print "L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P)
                complex_ret.real = math.cos((2 * PI / (num)) * (P))
                complex_ret.image = -math.sin((2 * PI / num) * (P))
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub


def auas_re_fft_1d(in_data, num):
    tot = num
    PI = 3.1415926
    forward_input_data(in_data, num)

    M = 1  # num = 2^m
    tmp = num / 2
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2

    complex_ret = complex()
    for L in range(1, M + 1):
        B = int(math.pow(2, L - 1))
        for J in range(0, B):
            P = math.pow(2, M - L) * J
            for K in range(J, num, int(math.pow(2, L))):
                # print "L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P)
                complex_ret.real = math.cos((2 * PI / num) * (P))
                complex_ret.image = -math.sin((2 * PI / num) * (P))
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub
    max = len(in_data)
    for it in range(max):
        in_data[it] = div_en(in_data[it], num)
        # print("num={0}".format(num))

def arr_real2mig(data):
    ret = []
    for it in data:
        c = complex()
        c.real = it
        c.image = 0
        ret.append(c)
    return ret


def cdt_1d(data,num):  #data real arry ret real arry
    d_data = []
    for i in range(num):
        c = complex()
        c.real = data[i]
        d_data.append(c)
    for i in range(num):
        c = complex()
        c.real  = data[num-i-1]
        d_data.append(c)
    fft_1d(d_data,num*2)
    # print("&&&&&&&&&&&&&&")
    # for it in d_data:
    #     show_I(it)
    # print("&&&&&&&&&&&&&&")
    change_theta(d_data,-1)
    ret = []

    for i in range(num):
        ret.append(d_data[i].real)
    return ret

def re_cdt_1d(data,num):  #data real arry ret real arry
    d_data = []
    for i in range(num):
        c = complex()
        c.real = data[i]
        d_data.append(c)

    for i in range(num):
        c = complex()
        if i>0:
            c.real = data[num-i]
        d_data.append(c)


    # exit(0)
    change_theta(d_data, 1)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # for it in d_data:
    #     show_I(it)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for i in range(num,2*num):
        c = complex()
        c.image = -d_data[i].image
        c.real = -d_data[i].real
        d_data[i] = c
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # show_I(c)
    # # # for i from num to
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # for it in d_data:
    #     show_I(it)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    re_fft_1d(d_data,num*2)

    # print("&&&&&&&&&&&&&&")
    # for it in d_data:
    #     show_I(it)
    # print("&&&&&&&&&&&&&&")
    ret = []
    for i in range(num):
        ret.append(d_data[i].real)
    return ret

def test_cdt(data,num): # 1d data
   a = 0

def change_theta(arr,type):
    '''

    :param arr:
    :param type: fft: 1   refft -1
    :return:
    '''
    # print("here")
    PI = 3.1415926
    num = len(arr)
    for i in range(len(arr)):
        theta = type*pi*(i)/(num)
        c = complex()
        c.real = math.cos(theta)
        c.image = math.sin(theta)
        temp = arr[i]
        arr[i] = mul_ee(temp,c)
    return

def cdt_2d(data):
    '''
    :param data: real mat
    :return:
    '''
    l,w=data.shape
    #print("in")
    for i in range(l):
        row = data[i,:]
        data[i,:] = cdt_1d(row,w)
    for j in range(w):
        col = data[:,j]
        data[:,j] = cdt_1d(col,l)
    return data

def recdt_2d(data):
    '''
    :param data: real mat
    :return:
    '''
    l,w=data.shape
    ret = copy_mat(data)
    for j in range(l):
        col = ret[:,j]

        ret[:,j]=re_cdt_1d(col,w)
    for i in range(l):
        row = ret[i,:]
        ret[i, :] = re_cdt_1d(row,w)


    return ret

def test_norm_cdt(data):
    im_frq = cdt_2d(data)
    # gray()
    # imshow(im_frq)
    # show()
    # exit(0)
    lb_img = LB(im_frq,0,0.01,1)
    lb_img_2 = LB(im_frq, 1, 0.01, 1)
    # lb_img = im_frq
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    f_img = recdt_2d(lb_img)
    f_img_2 = recdt_2d(lb_img_2)
    f = f_img+f_img_2
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish re~ \n")
    gray()
    imshow(f)
    show()
    return


def div_pic(data,a):
    '''
    :a: 0-1 double
    :param data: image data figure
    :return: image data figure
    '''
    global PI
    theta = -a*PI/2
    # print("div ~")
    e = complex()
    e.image = math.sin(theta)
    e.real = math.cos(theta)
    L,W = data.shape
    # print(e)
    # show_I(e)
    # exit(0)
    for i in range(L):
        for j in range(W):
            pre = complex()
            if a<0:
                mi = -a
                if not i==0:
                    x_dir = 1/pow((i), mi)
                else:
                    x_dir = 1
                if not j==0:
                    y_dir = 1/pow((j), mi)
                else:
                    y_dir = 1
            else:
                x_dir = pow((i),a)
                y_dir = pow((j),a)

            pre.real = mul_ee(data[i, j], e).real *x_dir*y_dir
            pre.image = mul_ee(data[i, j], e).image *x_dir*y_dir

            data[i,j] = pre
def test_norm_div(data,div,path):
    im_frq = fft_2d(data)
    div_pic(im_frq,div)
    # lb_img = LB(im_frq,0,0.01)
    lb_img = im_frq
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish frq~ \n")
    f_img = refft_2d(lb_img)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("finish re~ \n")
    im_final = getReal(f_img)
    gray()
    # f_img = np.asarray(f_img, dtype='float32')
    scipy.misc.imsave(path, im_final, )
    # imshow(im_final)
    # show()
    return