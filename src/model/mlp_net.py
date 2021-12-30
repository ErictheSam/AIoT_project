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
import torch

from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class MLP(nn.Module):
    """My layers"""

    def __init__(self, dict_name=''):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
            nn.ReLU(inplace=True)
        )
        if(dict_name == ''):
            self.module.apply(init_weights)
        else:
            self.load_state_dict(torch.load(dict_name))

    def forward(self, x):
        x = self.module(x)
        return x
