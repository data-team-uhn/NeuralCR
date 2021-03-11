#!/usr/bin/python

import ncrmodel_flask_loader

"""
Register model loader methods so that HTTP PUT requests to /models/<selected_model> can
be used to instantiate a concept recognition model from local files
"""
MODEL_LOADERS = {}
MODEL_LOADERS['neural'] = ncrmodel_flask_loader.loadfromrequest
