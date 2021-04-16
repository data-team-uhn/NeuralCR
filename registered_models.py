#!/usr/bin/python

import ncrmodel

#Stored in a form of {"object": model object, "threshold": threshold value}
NCR_MODELS = {}

"""
Start with at least one hard-coded model, in future versions of this API, newer trained models
can be automatically added to the NCR_MODELS data structure. For now, simply modify the following
lines to select which trained models are to be available for use.
"""
NCR_MODELS['HPO'] = {}
NCR_MODELS['HPO']['object'] = ncrmodel.NCR.loadfromfile('checks', '../NeuralCR/data/model_pmc.bin')
NCR_MODELS['HPO']['threshold'] = 0.6
