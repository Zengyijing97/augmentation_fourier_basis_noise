import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from numpy import asarray
import torch
import random
from abc import ABC, abstractmethod
from PIL import Image




# create Fourier Basis noise

def generate_fourier_base(h: int, w: int, h_index: int, w_index: int):
    assert h >= 1 and w >= 1
    assert abs(h_index) <= np.floor(h / 2) and abs(w_index) <= np.floor(w / 2)

    h_center_index = int(np.floor(h / 2))
    w_center_index = int(np.floor(w / 2))

    spectrum_matrix = torch.zeros(h, w)
    spectrum_matrix[h_center_index + h_index, w_center_index + w_index] = 1.0
    if (h_center_index - h_index) < h and (w_center_index - w_index) < w:
        spectrum_matrix[h_center_index - h_index, w_center_index - w_index] = 1.0

    spectrum_matrix = spectrum_matrix.numpy()
    spectrum_matrix = np.fft.ifftshift(spectrum_matrix)  # swap qadrant (low-freq centered to high-freq centered)

    fourier_base = torch.from_numpy(np.fft.ifft2(spectrum_matrix).real).float()
    fourier_base /= fourier_base.norm()

    return fourier_base

class AddFourierNoise(object):
    def __init__(self, h_index: int, w_index: int, eps: float, norm_type: str = 'l2'):
        """
        Add Fourier noise to RGB channels respectively.
        This class is able to use as same as the functions in torchvision.transforms.
        Args
            h_index: index of fourier basis about hight direction
            w_index: index of fourier basis about width direction
            eps: size of noise(perturbation)
        """
        assert eps >= 0.0
        self.h_index = h_index
        self.w_index = w_index
        self.eps = eps
        self.norm_type = norm_type

    def __call__(self, x):
        h = 32
        w = 32
        c = 3
        fourier_base = generate_fourier_base(h, w, self.h_index, self.w_index)  # l2 normalized fourier base

        fourier_base /= fourier_base.norm()
        fourier_base *= self.eps  # this looks might be same as original implementation.

        fourier_noise = fourier_base.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # multiple random noise form [-1, 1]
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)
        # print(fourier_noise.shape)
        convert_tensor = transforms.ToTensor()
        # x = torch.from_numpy(x)
        #x:IMG
        # print(type(x))
        # print(x.size)
        x = np.array(x)
        # print("x.shape after np.array:",x.shape)
        # x = convert_tensor(x)
        x = x.reshape(3,32,32)
        x = torch.from_numpy(x)
        # x = x.permute(2,0,1)
        
        # print("x.reshape:",x.shape)
        # with open("fastautoaugment/shape.txt", 'a') as f:
        #     f.write("fourier_noise: " + str(fourier_noise.shape) + "\n")
        #     f.write("x: " + str(x.shape) + "\n")
        img_new = torch.clamp(x + fourier_noise, min=0.0, max=1.0)
        img_new = img_new.numpy()
        img_new = img_new.reshape(3,32,32)
        return img_new

    @classmethod
    def calc_fourier_noise(cls, c: int, h: int, w: int, h_index: int, w_index: int, eps: float):
        assert c > 0 and h > 0 and w > 0
        assert abs(h_index) <= np.floor(h / 2) and abs(w_index) <= np.floor(w / 2)

        fourier_base = generate_fourier_base(h, w, h_index, w_index)  # l2 normalized fourier base
        fourier_base /= fourier_base.norm()
        fourier_base *= eps  # this looks might be same as original implementation.

        fourier_noise = fourier_base.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # multiple random noise form [-1, 1]
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return fourier_noise
    
def distance(i, j, imageSize, r):
    dis = np.sqrt((i - (imageSize-1)/2) ** 2 + (j - (imageSize-1)/2) ** 2)
    if dis < r and dis >= r-1:
        return 1.0
    else:
        return 0

def mask_radial(rows, cols, r):
    mask = np.zeros((rows, cols))
    res = []
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
            if mask[i,j] == 1:
                res.append((i,j))
    return res

def group_frequency(eps, group):
    transformations = []
    groups = mask_radial(31,31,group)
    tem = []
    p = []
    for group in groups:
        tem.append(group)
        for (h, w) in tem:
            transformations.append(AddFourierNoise(h-15, w-15, eps))
            p.append(1)
            # print((h-15,w-15))

    return transforms.RandomChoice(transformations, p)

class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass
    
class FourierBasisNoise_1(BaseTransform):
    def transform(self, img):
        eps = self.mag
        # print(type(img))
        # print(img.size)
        transform = group_frequency(eps,1)
        # img = np.array(img)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_2(BaseTransform):
    def transform(self, img):
        eps = self.mag
        # print(type(img))
        # print(img.size)
        transform = group_frequency(eps,2)
        # img = np.array(img)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_3(BaseTransform):
    def transform(self, img):
        eps = self.mag
        # print(type(img))
        # print(img.size)
        transform = group_frequency(eps,3)
        # img = np.array(img)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_4(BaseTransform):
    def transform(self, img):
        eps = self.mag
        # print(type(img))
        # print(img.size)
        transform = group_frequency(eps,4)
        # img = np.array(img)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_5(BaseTransform):
    def transform(self, img):
        eps = self.mag
        # print(type(img))
        # print(img.size)
        transform = group_frequency(eps,5)
        # img = np.array(img)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_6(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,6)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_7(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,7)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_8(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,8)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_9(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,9) 
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_10(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,10)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_11(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,11)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_12(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,12)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_13(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,13)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_14(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,14)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_15(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,15)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_16(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,16)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_17(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,17)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_18(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,18)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_19(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,19)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_20(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,20)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img
    
class FourierBasisNoise_21(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,21)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

class FourierBasisNoise_22(BaseTransform):
    def transform(self, img):
        eps = self.mag
        transform = group_frequency(eps,22)
        img = transform(img)
        img = Image.fromarray(img[0,:,:])
        img = img.convert("RGB")
        return img

