import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse
import time
import sys
from argparse import ArgumentParser
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceLandmarksDetection:

    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU', extensions=None, async_mode = True):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.output_shape = None
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.extensions = extensions
        self.initial_w = None
        self.initial_h = None
        self.async_mode = async_mode


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize the plugin

        self.plugin = IEPlugin(device=self.device)
        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.plugin.add_cpu_extension(self.extensions)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        self.check_plugin(self.plugin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load(self.network)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_blob].shape
        print("Face landmark Detection Model output shape : ", self.output_shape)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        count = 0
        coords = None
        self.initial_w = image.shape[1]
        self.initial_h = image.shape[0]
        frame = self.preprocess_input(image)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(inputs={self.input_blob: frame})
        else:
            self.exec_network.requests[0].infer(inputs={self.input_blob: frame})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            frame,coords = self.preprocess_output(image, outputs)
            return coords, frame

    def check_plugin(self, plugin):
        '''
        TODO: You will need to complete this method as a part of the
        standout suggestions
        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        unsupported_layers = [l for l in self.network.layers.keys() if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        # Reshape the image to input size of land_mark detection model
        # it should be h*w = 48x48 - as described in openVino doc
        frame = cv2.resize(image, (w, h))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape((n, c, h, w))
        return frame

    def preprocess_output(self, frame, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        current_count = 0
        coords = []
        outputs= outputs[0]
        
        xl,yl = outputs[0][0]*self.initial_w,outputs[1][0]*self.initial_h
        xr,yr = outputs[2][0]*self.initial_w,outputs[3][0]*self.initial_h

        # make box for left eye 
        xlmin = xl-20
        ylmin = yl-20
        xlmax = xl+20
        ylmax = yl+20
        
        # make box for right eye 
        xrmin = xr-20
        yrmin = yr-20
        xrmax = xr+20
        yrmax = yr+20
        
        cv2.rectangle(frame, (xlmin, ylmin), (xlmax, ylmax), (0, 55, 255), 1)
        cv2.rectangle(frame, (xrmin, yrmin), (xrmax, yrmax), (0, 55, 255), 1)
        coords = [[int(xlmin),int(ylmin),int(xlmax),int(ylmax)],[int(xrmin),
                    int(yrmin),int(xrmax),int(yrmax)]]
        return frame, coords

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network