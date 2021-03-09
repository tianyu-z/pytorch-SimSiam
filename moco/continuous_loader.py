from PIL import Image
from torchvision import transforms
from moco.transforms import RandomResizedCrop, RandomGrayscale, ColorJitter, GaussianBlur
# from moco.loader import GaussianBlur
import random
import numpy as np


def IOU(para_1, para_2):
    bbox1 = [para_1[0], para_1[1], para_1[0] + para_1[2], para_1[1] + para_1[3]]
    bbox2 = [para_2[0], para_2[1], para_2[0] + para_2[2], para_2[1] + para_2[3]]

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if np.maximum(bbox1[0], bbox2[0]) < np.minimum(bbox1[2], bbox2[2]) and np.maximum(bbox1[1],
                                                                                      bbox2[1]) < np.minimum(
        bbox1[3], bbox2[3]):
        I = (np.minimum(bbox1[2], bbox2[2]) - np.maximum(bbox1[0], bbox2[0])) * (
                np.minimum(bbox1[3], bbox2[3]) - np.maximum(bbox1[1], bbox2[1]))
        U = area1 + area2 - I
        return I * 1. / U
    else:
        return 0


class TwoTransformPara:
    """Take two transforms of one image as the query and key with parameters returned."""

    def __init__(self, base_transform=None):
        if base_transform is None:
            base_transform = AugmentationSet()
        self.base_transform = base_transform

    def __call__(self, x):
        [q, para_q] = self.base_transform(x)
        [k, para_k] = self.base_transform(x)
        score = self.score(para_q,para_k)
        # print(para_k,para_q,score)
        # raise Exception
        return [q, k, score]

    def score(self, para_q, para_k):
        # return IOU(para_k['crop'], para_q['crop']) # IOU
        score = np.ones(4)
        rate = 0.2
        score[0] = IOU(para_k['crop'], para_q['crop']) * rate + (1. - rate)
        score[1] = np.mean(np.abs(para_q['color'] - para_k['color'])/np.array([0.4, 0.4, 0.4, 0.1]))/2. * rate + (1. - rate)
        score[2] = np.abs(para_q['gray'] - para_k['gray']) * rate + (1. - rate)
        score[3] = np.abs(para_k['blur'] - para_q['blur'])/2. * rate + (1. - rate)
        # print(score, np.mean(score), np.abs(para_q['color'] - para_k['color']))
        # raise Exception
        return np.mean(score)

    # def geometric_score(self, para_q, para_k):
    #     score = np.ones(4)
    #     rate = 0.2
    #     score[0] = IOU(para_k['crop'], para_q['crop']) * rate + (1. - rate)
    #     score[1] = self.geometric_discrepancy(para_q['color'], para_k['color'], np.array([0.4, 0.4, 0.4, 0.1])) * rate + (1. - rate)
    #     score[2] = np.abs(para_q['gray'] - para_k['gray']) * rate + (1. - rate)
    #     score[3] = np.abs(para_k['blur'] - para_q['blur'])/2. * rate + (1. - rate)
    #     # print(score, np.mean(score))
    #     # raise Exception
    #     return self.geometric_mean(score)
    #
    # def geometric_mean(self, score):
    #     return np.exp(np.mean(np.log(score)))
    #
    # def geometric_discrepancy(self, para_q, para_k, max_d):
    #     ratio = np.maximum(para_q, para_k)
    #     return 0

class AugmentationSet(object):
    def __init__(self, debug = False):
        self.debug = debug

    def __call__(self, img, debug = False):
        para = {}
        img, para['crop'] = RandomResizedCrop(224, scale=(0.2, 1.))(img)
        if 0.8 > random.random():
            img, para['color'] = ColorJitter(0.4, 0.4, 0.4, 0.1)(img)
        else:
            para['color'] = [1.,1.,1.,0.]
        para['color'] = np.array(para['color'])
        img, para['gray'] = RandomGrayscale(p=0.2)(img)
        if 0.5 > random.random():
            img, para['blur'] = GaussianBlur([.1, 2.])(img)
        else:
            para['blur'] = 0.

        img = transforms.RandomHorizontalFlip()(img)
        image = img.copy()

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.debug:
            return [image, para]
        else:
            return [img, para]