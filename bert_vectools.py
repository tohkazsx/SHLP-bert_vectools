import os
import math
import logging
import numpy as np
from transformers import BertTokenizer, BertModel, RoFormerTokenizer, RoFormerModel, AutoTokenizer, AutoModel
from typing import Union, List
from numpy import ndarray
import torch
import torch.nn as nn

from . import __version__
from .utility import snapshot_download

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)