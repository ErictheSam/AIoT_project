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

import os
import torch

import numpy as np
import sys

from src.aruco_detection import detect_and_calculate
from src.model.mlp_net import MLP


def main(argv):
    if(len(argv) != 2):
        print('Error: Params illegal!')
        return
    video_name = argv[0]
    param_dict = argv[1]

    root_dir = os.path.dirname(__file__)
    print(root_dir)
    detectable, traits = detect_and_calculate('videos/' + video_name)
    if detectable == False:
        print('Error: Video not detectable!')
        return
    param_dict = 'weights/'+param_dict
    model = MLP(param_dict)
    traits = torch.tensor(traits, dtype=torch.float32)
    traits = torch.unsqueeze(traits,0)
    outputs = model(traits)
    _, predicted = torch.max(outputs.data,1)
    if(predicted[0] == 1):
        print('Not grabbing!')
    else:
        print('Grabbing!')


if __name__ == '__main__':
    main(sys.argv[1:])
