'''
  ==================================================================
  Copyright (c) 2021, Tsinghua University.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the
  distribution.
  3. All advertising materials mentioning features or use of this software
  must display the following acknowledgement:
  This product includes software developed by the xxx Group. and
  its contributors.
  4. Neither the name of the Group nor the names of its contributors may
  be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY PI-CS Tsinghua University
  ===================================================================
  Author: Yibo Shen(EricSam413@outlook.com)
'''

import cv2
import math
import os
import sys

import cv2.aruco as aruco
import numpy as np

axis = np.float32([[0, 0, 0], [0, 0.01, 0], [0.01, 0, 0]])
mtx = np.array([[629.61554535, 0., 333.7279485], [
               0., 631.61712266, 229.33660831], [0., 0., 1.]])
dist = np.array(
    ([[0.03109901, -0.0100412, -0.00944869, 0.00123176, 0.31024847]]))
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 43
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.01
parameters.minCornerDistanceRate = 0.01
parameters.minOtsuStdDev = 10
parameters.polygonalApproxAccuracyRate = 0.09
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2
parameters.maxErroneousBitsInBorderRate = 0.5
parameters.errorCorrectionRate = 0.8

def _update_traits(id, x, y, occured_ids, xy_values):
    '''
        Calculate the min/max xy slopes
    '''
    if occured_ids[id] == 0:
        xy_values[id, 0:3, 0] = x
        xy_values[id, 0:3, 1] = y
        occured_ids[id] = 1
    else:
        if(x < xy_values[id][1][0]):
            xy_values[id][1][0] = x
        if(x > xy_values[id][2][0]):
            xy_values[id][2][0] = x
        if(y < xy_values[id][1][1]):
            xy_values[id][1][1] = y
        if(y > xy_values[id][2][1]):
            xy_values[id][2][1] = y
        xy_values[id][3][0] += (x-xy_values[id][0][0])
        xy_values[id][3][1] += (y-xy_values[id][0][1])


def detect_and_calculate(file_name):
    '''
        1. Detect every ARUCO marker and their pose
        2. If detected, calculate the pivotal xy slope value of each marker
    '''
    cap = cv2.VideoCapture(file_name)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    ret, frame = cap.read()
    if not ret:
        print("can't open video!")
        sys.exit(-1)
    occured_ids = np.zeros(5)
    xy_values = np.zeros((5, 4, 2))
    while ret:
        '''Detect the markers using algorithm in cv2.aruco'''
        corners, ids, _ = aruco.detectMarkers(
            frame, aruco_dict, parameters=parameters)
        if corners:
            '''Estimate the xyz pose of each marker under camera'''
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.01, mtx, dist)
            for i in range(rvec.shape[0]):
                if ids[i][0] < 5:
                    '''Calculate the xy-slope and update'''
                    imgpts, _ = cv2.projectPoints(
                        axis, rvec[i, :, :], tvec[i, :, :], mtx, dist)
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    vec1 = tuple(imgpts[1]-imgpts[0])
                    vec2 = tuple(imgpts[2]-imgpts[0])
                    slope1 = math.atan2(vec1[1], vec1[0])
                    slope2 = math.atan2(vec2[1], vec2[0])
                    _update_traits(ids[i][0], slope1, slope2,
                                   occured_ids, xy_values)
        ret, frame = cap.read()
    
    '''Set the outputs'''
    outputs = np.zeros(10)
    if occured_ids.sum() < 5:
        return False, outputs
    else:
        for i in range(5):
            if xy_values[i][3][0] > 0:
                outputs[2*i] = xy_values[i][2][0]-xy_values[i][0][0]
            else:
                outputs[2*i] = xy_values[i][1][0]-xy_values[i][0][0]
            if xy_values[i][3][1] > 0:
                outputs[2*i+1] = xy_values[i][2][1]-xy_values[i][0][1]
            else:
                outputs[2*i+1] = xy_values[i][1][1]-xy_values[i][0][1]

        return True, outputs
