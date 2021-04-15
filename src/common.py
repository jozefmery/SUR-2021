import os
import logging as log

# call before importing tf or keras
def shut_up_tensorflow_thanks():
  set_tf_loglevel(log.FATAL)

def set_tf_loglevel(level):

  if level >= log.FATAL:
  
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  
  elif level >= log.ERROR:
      
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  
  elif level >= log.WARNING:
      
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  
  else:
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  log.getLogger("tensorflow").setLevel(level)