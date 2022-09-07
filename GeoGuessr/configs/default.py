## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

#########################################################

from PIL.Image import NONE
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

 # data  
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'country211' # (processed) data folder name 
  data.data_dir = '/content/' # you can change this, it decide where to put the downloaded data
  data.batch_size = 5
  data.gpus= None # use gpu or not, it will use default value if None
  data.num_workers = None # it will use default value if None
  data.extensions = None # the extentions of files to include in the datasets, it will use default if None
  
  #trainer
  config.trainer = trainer = ml_collections.ConfigDict()
  trainer.gpu = 1
  trainer.precision = 32
  trainer.limit_test_batches=1

  return config