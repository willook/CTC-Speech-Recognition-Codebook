# -*- coding: utf-8 -*-
#!/usr/bin/env python

import yaml
class default():
    def __init__(self):
        self.sr = 16000
        self.win_length = 160     #10ms
        self.hop_length = 80      # 5ms
        self.n_fft = 256
        self.preemphasis = 0.97
        self.n_mfcc = 13
        self.max_db = 35
        self.min_db = -55
        self.n_mels = 40
        
class hparam():
    def __init__(self):
        self.default = default()
    


