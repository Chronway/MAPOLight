# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:39:00 2022

@author: Wang Chong
"""

import os
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger

#本函数暂时用不上，要改文件夹改环境变量即可
def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator