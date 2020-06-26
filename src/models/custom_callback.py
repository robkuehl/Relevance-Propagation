#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:36:31 2020

@author: robin
"""


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.platform import tf_logging as logging
import six
import numpy as np


class CustomModelCheckpoint(ModelCheckpoint):
    
    def __init_(self,
               filepath,
               monitor=['val_precision', 'val_recall'],
               verbose=0,
               save_best_only=True,
               save_weights_only=False,
               save_freq='epoch',
               *args,
               **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        
        # Deprecated field `load_weights_on_restart` is for loading the checkpoint
        # file from `filepath` at the start of `model.fit()`
        # TODO(rchao): Remove the arg during next breaking release.
        if 'load_weights_on_restart' in kwargs:
          self.load_weights_on_restart = kwargs['load_weights_on_restart']
          logging.warning('`load_weights_on_restart` argument is deprecated. '
                          'Please use `model.load_weights()` for loading weights '
                          'before the start of `model.fit()`.')
        else:
          self.load_weights_on_restart = False
    
        # Deprecated field `period` is for the number of epochs between which
        # the model is saved.
        if 'period' in kwargs:
          self.period = kwargs['period']
          logging.warning('`period` argument is deprecated. Please use `save_freq` '
                          'to specify the frequency in number of batches seen.')
        else:
          self.period = 1
    
        self.monitor_op = np.greater
        self.best = (-np.Inf, -np.Inf)
    
        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
          raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))
    
        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False
        
        print(self.best)
    
          
        
        
    
    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
    
        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
          self.epochs_since_last_save = 0
          filepath = self._get_file_path(epoch, logs)
    
          try:
            current_precision = logs.get(self.monitor[0])
            current_recall = logs.get(self.monitor[1])
            current = (current_precision, current_recall)
            if current is None:
              logging.warning('Can save best model only with %s available, '
                              'skipping.', self.monitor)
            else:
              if self.monitor_op(current_precision, self.best[0]) and self.monitor_op(current_recall, self.best[1]):
                if self.verbose > 0:
                  print('\nEpoch %05d: (%s, %s) improved from (%0.5f, %0.5f) to (%0.5f, %0.5f),'
                        ' saving model to %s' % (epoch + 1, self.monitor[0],self.monitor[1],
                                                 self.best[0], self.best[1], current_precision, current_recall, filepath))
                self.best = current
                if self.save_weights_only:
                  self.model.save_weights(filepath, overwrite=True)
                else:
                  self.model.save(filepath, overwrite=True)
              else:
                if self.verbose > 0:
                  print('\nEpoch %05d: (%s. %s) did not improve' %
                        (epoch + 1, self.monitor[0], self.monitor[1]))
    
            self._maybe_remove_file()
          except IOError as e:
            # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            if 'is a directory' in six.ensure_str(e.args[0]):
              raise IOError('Please specify a non-directory filepath for '
                            'ModelCheckpoint. Filepath used is an existing '
                            'directory: {}'.format(filepath))