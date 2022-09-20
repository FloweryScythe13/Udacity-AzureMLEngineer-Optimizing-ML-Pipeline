# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"age": pd.Series([0], dtype="int64"), "marital": pd.Series([0], dtype="int64"), "default": pd.Series([0], dtype="int64"), "housing": pd.Series([0], dtype="int64"), "loan": pd.Series([0], dtype="int64"), "month": pd.Series([0], dtype="int64"), "day_of_week": pd.Series([0], dtype="int64"), "duration": pd.Series([0], dtype="int64"), "campaign": pd.Series([0], dtype="int64"), "pdays": pd.Series([0], dtype="int64"), "previous": pd.Series([0], dtype="int64"), "poutcome": pd.Series([0], dtype="int64"), "emp_var_rate": pd.Series([0.0], dtype="float64"), "cons_price_idx": pd.Series([0.0], dtype="float64"), "cons_conf_idx": pd.Series([0.0], dtype="float64"), "euribor3m": pd.Series([0.0], dtype="float64"), "nr_employed": pd.Series([0.0], dtype="float64"), "job_admin_": pd.Series([0], dtype="int64"), "job_blue-collar": pd.Series([0], dtype="int64"), "job_entrepreneur": pd.Series([0], dtype="int64"), "job_housemaid": pd.Series([0], dtype="int64"), "job_management": pd.Series([0], dtype="int64"), "job_retired": pd.Series([0], dtype="int64"), "job_self-employed": pd.Series([0], dtype="int64"), "job_services": pd.Series([0], dtype="int64"), "job_student": pd.Series([0], dtype="int64"), "job_technician": pd.Series([0], dtype="int64"), "job_unemployed": pd.Series([0], dtype="int64"), "job_unknown": pd.Series([0], dtype="int64"), "contact_cellular": pd.Series([0], dtype="int64"), "contact_telephone": pd.Series([0], dtype="int64"), "education_basic_4y": pd.Series([0], dtype="int64"), "education_basic_6y": pd.Series([0], dtype="int64"), "education_basic_9y": pd.Series([0], dtype="int64"), "education_high_school": pd.Series([0], dtype="int64"), "education_illiterate": pd.Series([0], dtype="int64"), "education_professional_course": pd.Series([0], dtype="int64"), "education_university_degree": pd.Series([0], dtype="int64"), "education_unknown": pd.Series([0], dtype="int64")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")
sample_global_params = StandardPythonParameterType({"method": method_sample})

result_sample = NumpyParameterType(np.array([0]))
output_sample = StandardPythonParameterType({'Results':result_sample})

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, GlobalParameters={"method": "predict"}):
    data = Inputs['data']
    if GlobalParameters.get("method", None) == "predict_proba":
        result = model.predict_proba(data)
    elif GlobalParameters.get("method", None) == "predict":
        result = model.predict(data)
    else:
        raise Exception(f"Invalid predict method argument received. GlobalParameters: {GlobalParameters}")
    if isinstance(result, pd.DataFrame):
        result = result.values
    return {'Results':result.tolist()}
