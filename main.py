import torch.nn as nn
import torch, os
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from crnn import CRNN
import config
from mydataset import resizeNormalize
from utils import strLabelConverter

class PytorchOcr():
    def __init__(self, model_path=''):
        self.alphabet = config.alphabet_v2
        self.nclass = len(self.alphabet) + 1
        self.model = CRNN(config.imgH, 1, self.nclass, 256)
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            self.model.cuda()
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.converter = strLabelConverter(self.alphabet)

    def recognize(self, img): # input: numpy, uint8, gray image

        h, w = img.shape

        image = Image.fromarray(img)
        transformer = resizeNormalize((int(w/h*32), 32))
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)

        if self.cuda:
            image = image.cuda()

        preds = self.model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        txt = self.converter.decode(preds.data, preds_size.data, raw=False).strip()

        return txt

if __name__ == '__main__':
    model_path = os.path.join(os.path.abspath('crnn_saved_models'), 'CRNN-CAPSTONE_51_980.pth')
    recognizer = PytorchOcr(model_path) # cost 4 seconds
    img_name = os.path.join(os.path.abspath('data_set'), 'test_images', 'splited_image_11.jpg')
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = recognizer.recognize(img) # cost 1 seconds a piece
    print(result)