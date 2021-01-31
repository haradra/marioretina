import cv2
import numpy as np
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
from os.path import join
import os
from matplotlib import pyplot as plt

class CorticalMapping:
    def __init__(self):
        self.R = None
        self.C = None
        self.fixation = None

    def setup_cortex(self, cortex_left, cortex_right):
        #Create and prepare cortex
        self.C = Cortex()
        lp = join(datadir, "cortices", "{0}loc.pkl".format(cortex_left))
        rp = join(datadir, "cortices", "{0}loc.pkl".format(cortex_right))
        self.C.loadLocs(lp, rp)
        self.C.loadCoeffs(join(datadir, "cortices", "{0}coeff.pkl".format(cortex_left)), join(datadir, "cortices", "{0}coeff.pkl".format(cortex_right)))

    def setup_retina(self, retina_resolution):
        #Create and load retina
        self.R = Retina()
        self.R.info()
        self.R.loadLoc(join(datadir, "retinas", "{0}_loc.pkl".format(retina_resolution)))
        self.R.loadCoeff(join(datadir, "retinas", "{0}_coeff.pkl".format(retina_resolution)))

        # img = cv2.imread("{}/examples/mario.png".format(os.getcwd()), cv2.IMREAD_COLOR)
        # print(os.getcwd())
        # print(type(img))

        #Prepare retina
        # x = img.shape[1]/2
        # y = img.shape[0]/2
        # print("X: ", x)
        # print("Y: ", y)
        # print("Retina shape: ", img.shape)
        x = 120.0
        y = 112.0
        self.fixation = (y,x)
        self.R.prepare((224, 240, 3), self.fixation)


    def cortical_transform(self, im_array):
        V = self.R.sample(im_array, self.fixation)
        cimg = self.C.cort_img(V)
        # print("Cimg type: ", type(cimg))
        return cimg

    def backproject_transform(self, im_array):
        V = self.R.sample(im_array, self.fixation)
        tight = self.R.backproject_tight_last()
        # print("Cimg type: ", type(cimg))
        return tight

    def arr_to_img(self, img_arr, index):
        cv2.imwrite("{}/test_images/test_image_{}.jpg".format(os.getcwd(), index), img_arr)

        # cv2.imwrite("{}/examples/mario_backprojection.png".format(os.getcwd()), tight)
    # cv2.namedWindow("inverted", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("inverted", tight) 
            
    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input", img) 
            
    # cv2.namedWindow("cortical", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("cortical", cimg)
            
    # key = cv2.waitKey(10)

