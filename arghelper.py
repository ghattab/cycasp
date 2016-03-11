#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

def inputfile(path):
    if not path.endswith('.csv'):
        raise argparse.ArgumentTypeError('argument filename must be of type *.csv')
    return path

#def inputdirectory(arg): # (parser, arg)
#    if not os.path.isdir(arg):
#        parser.error('The directory {} does not exist!'.format(arg))
#    else:
#        # File exists so return the directory
#        return arg

def check_range(arg):
    try:
        value = int(arg)
    except ValueError as err:
       raise argparse.ArgumentTypeError(str(err))
    if value < 0 or value > 4:
        message = "Expected [1:4], got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)
    return value

def inputdir(parser, path):
  if not os.path.isdir(path):
    message = "directory:{0} is not a valid path".format(path)
    parser.error(message)
  elif os.access(path, os.R_OK):
    return path
  else:
    message = "directory:{0} is not a readable directory".format(path)
    raise argparse.ArgumentError(message)

