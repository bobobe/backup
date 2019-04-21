import sys,os
sys.path.append("..")
#from models.GSN import GSN
#from configure import args,paths,config
from inputs.dataloader_test import load_data,load_glove_embed_matrix,load_random_embed_matrix
import tensorflow as tf
from models.model_helper import get_logger

logger = get_logger("log")
logger.info("3333333333333")