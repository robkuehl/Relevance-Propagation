#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""

from models.multilabel_cnn_classifier import cnn_classifier

classifier = cnn_classifier(model_type='complex_model', dataset='cifar10')
classifier.create_model()
classifier.run_model(batch_size=100, epochs=50)
classifier.pred(18)
evaluation = classifier.eval()
