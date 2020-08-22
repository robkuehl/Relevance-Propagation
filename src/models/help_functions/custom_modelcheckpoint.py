from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""
Der Custom Model Checkpoint wird im Training des Multilabel Classifiers für den Pascal VOC Datensatz verwendet.
Der Code ist eine abgeänderte Variante des Source Codes der Keras Klasse. Er ist dahingehend verändert worden, dass er nun zwei Metriken berücksichtigt.  
Die Änderungen wurden in der Methode _save_model (ab Zeile 257) vorgenommen und sehen vor, dass der Checkpoint mit den Metriken "precision" und "recall" verwendet wird.
"""

import collections
import copy
import csv
import io
import json
import os
import re
import tempfile
import time

import numpy as np
import six

# from tensorflow.python.data.ops import iterator_ops
# from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.distribute import multi_worker_util
# from tensorflow.python.eager import context
# from tensorflow.python.framework import ops
# from tensorflow.python.keras import backend as K
from tensorflow.python.keras.distribute import multi_worker_training_state as training_state
# from tensorflow.python.keras.utils import generic_utils
# from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.keras.utils.data_utils import Sequence
# from tensorflow.python.keras.utils.generic_utils import Progbar
# from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import summary_ops_v2
# from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.training import checkpoint_management
# from tensorflow.python.util import nest
# from tensorflow.python.util.compat import collections_abc
# from tensorflow.python.util.tf_export import keras_export
# from tensorflow.tools.docs import doc_controls

from tensorflow.keras.callbacks import Callback

class ModelCheckpoint(Callback):
  """Callback to save the Keras model or model weights at some frequency.
  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.
  A few options this callback provides include:
  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.
  Example:
  ```python
  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_acc',
      mode='max',
      save_best_only=True)
  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```
  Arguments:
      filepath: string, path to save the model file. `filepath` can contain
        named formatting options, which will be filled the value of `epoch` and
        keys in `logs` (passed in `on_epoch_end`). For example: if `filepath` is
        `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints
        will be saved with the epoch number and the validation loss in the
        filename.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, the latest best model according
        to the quantity monitored will not be overwritten.
        If `filepath` doesn't contain formatting options like `{epoch}` then
        `filepath` will be overwritten by each new better model.
      mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        overwrite the current save file is made based on either the maximization
        or the minimization of the monitored quantity. For `val_acc`, this
        should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. Note that if the saving isn't aligned
        to epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor=['val_precision', 'val_recall'],
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               **kwargs):
    super(ModelCheckpoint, self).__init__()
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

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = (np.Inf, np.Inf)
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = (-np.Inf, -np.Inf)
    

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model
    # Use name matching rather than `isinstance` to avoid circular dependencies.
    if (not self.save_weights_only and
        not model._is_graph_network and  # pylint: disable=protected-access
        model.__class__.__name__ != 'Sequential'):
      self.save_weights_only = True

  def on_train_begin(self, logs=None):
    # pylint: disable=protected-access
    if self.model._in_multi_worker_mode():
      # MultiWorkerTrainingState is used to manage the training state needed
      # for preemption-recovery of a worker in multi-worker training.
      self.model._training_state = (
          training_state.MultiWorkerTrainingState(self.model, self.filepath))
      self._training_state = self.model._training_state
      if self._training_state.restore():
        # If the training state needs to be and is successfully restored,
        # it is recovering from a previous failure (or preemption). In such
        # case, do not load the weights from user specified file path.
        return

    # If this is not multi worker training, restoring is not needed, or
    # restoring failed, check if it should load weights on restart.
    if self.load_weights_on_restart:
      if (not self.model._in_multi_worker_mode() or
          multi_worker_util.should_load_checkpoint()):
        filepath_to_load = (
            self._get_most_recently_modified_file_matching_pattern(
                self.filepath))
        if (filepath_to_load is not None and
            training_state.checkpoint_exists(filepath_to_load)):
          try:
            # `filepath` may contain placeholders such as `{epoch:02d}`, and
            # thus it attempts to load the most recently modified file with file
            # name matching the pattern.
            self.model.load_weights(filepath_to_load)
          except (IOError, ValueError) as e:
            raise ValueError('Error loading file from {}. Reason: {}'.format(
                filepath_to_load, e))

  def on_train_end(self, logs=None):
    # pylint: disable=protected-access
    if self.model._in_multi_worker_mode():
      if self.model.stop_training or getattr(
          self.model, '_successful_loop_finish', False):
        # In multi-worker training, on successful exit of training, delete the
        # training state backup file that was saved for the purpose of worker
        # recovery.
        self._training_state.delete_backup()
        # Restore the training state so the model is ready for next (possible)
        # multi worker training.
        del self._training_state
        self.model._training_state = None

  def on_batch_end(self, batch, logs=None):
    if self._implements_train_batch_hooks():
      logs = logs or {}
      self._batches_seen_since_last_saving += 1
      if self._batches_seen_since_last_saving >= self.save_freq:
        self._save_model(epoch=self._current_epoch, logs=logs)
        self._batches_seen_since_last_saving = 0

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      if self.model._in_multi_worker_mode():
        # Exclude training state variables in user-requested checkpoint file.
        with self._training_state.untrack_vars():
          self._save_model(epoch=epoch, logs=logs)
      else:
        self._save_model(epoch=epoch, logs=logs)
    if self.model._in_multi_worker_mode():
      # For multi-worker training, back up the weights and current training
      # state for possible future recovery.
      # TODO(rchao): Call `back_up` at finer period such as N steps.
      self._training_state.back_up(epoch)

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
        if self.save_best_only:
          current_precision = logs.get(self.monitor[0])
          current_recall = logs.get(self.monitor[1])
          current = (current_precision, current_recall)
          if current is None:
            logging.warning('Can save best model only with {} available, skipping.'.format(self.monitor))
          else:
            # Beide Metriken haben sich verbessert
            both_improved = self.monitor_op(current[0], self.best[0]) and self.monitor_op(current[1], self.best[1])
            # eine Metrik hat sicher verbessert während die andere nicht schlechter geworden ist
            one_improved = (self.monitor_op(current[0], self.best[0]) and current[1]==self.best[1]) or (self.monitor_op(current[1], self.best[1]) and current[0]==self.best[0])
            # Die Metriken haben sich im Mittel verbessert und haben sich in ihren Werten einander genähert
            # Bsp: (0.8, 0.8) entspricht einer wesentlich besseren Klassifikation als (0.1, 0.9)
            mean_improved = sum(current)/2 > sum(self.best)/2 and abs(current[0]-current[1]) <= abs(self.best[0]-self.best[1])
            
            if both_improved or one_improved or mean_improved:
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from (%0.5f, %0.5f) to (%0.5f, %0.5f),'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best[0], self.best[1], current[0], current[1], filepath))
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
              else:
                self.model.save(filepath, overwrite=True)
            else:
              if self.verbose > 0:
                if not self.monitor_op(current[0], self.best[0]):
                    print('\nEpoch %05d: %s did not improve' %
                          (epoch + 1, self.monitor[0]))
                if not self.monitor_op(current[1], self.best[1]):
                    print('\nEpoch %05d: %s did not improve' %
                          (epoch + 1, self.monitor[1]))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
          else:
            self.model.save(filepath, overwrite=True)

        self._maybe_remove_file()
      except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in six.ensure_str(e.args[0]):
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))

  def _get_file_path(self, epoch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    if not self.model._in_multi_worker_mode(
    ) or multi_worker_util.should_save_checkpoint():
      try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
        return self.filepath.format(epoch=epoch + 1, **logs)
      except KeyError as e:
        raise KeyError('Failed to format this callback filepath: "{}". '
                       'Reason: {}'.format(self.filepath, e))
    else:
      # If this is multi-worker training, and this worker should not
      # save checkpoint, we use a temp filepath to store a dummy checkpoint, so
      # it writes to a file that will be removed at the end of `_save_model()`
      # call. This is because the SyncOnReadVariable needs to be synced across
      # all the workers in order to be read, and all workers need to initiate
      # that.
      self._temp_file_dir = tempfile.mkdtemp()
      extension = os.path.splitext(self.filepath)[1]
      return os.path.join(self._temp_file_dir, 'temp' + extension)

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.

    if (self.model._in_multi_worker_mode() and  # pylint: disable=protected-access
        not multi_worker_util.should_save_checkpoint()):
      file_io.delete_recursively(self._temp_file_dir)
      del self._temp_file_dir

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.
    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.
    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.
    Modified time of a file is obtained with `os.path.getmtime()`.
    This utility function is best demonstrated via an example:
    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```
    Arguments:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.
    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name

  def _implements_train_batch_hooks(self):
    # If save_freq="epoch", batch-level hooks don't need to be run.
    return isinstance(self.save_freq, int)