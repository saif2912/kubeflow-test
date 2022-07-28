#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2020 The TensorFlow Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Create a TFX pipeline using templates

# Note: We recommend running this tutorial on Google Cloud Vertex AI Workbench. [Launch this notebook on Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?q=download_url%3Dhttps%253A%252F%252Fraw.githubusercontent.com%252Ftensorflow%252Ftfx%252Fmaster%252Fdocs%252Ftutorials%252Ftfx%252Ftemplate.ipynb).
# 
# 
# <div class="devsite-table-wrapper"><table class="tfo-notebook-buttons" align="left">
# <td><a target="_blank" href="https://www.tensorflow.org/tfx/tutorials/tfx/template">
# <img src="https://www.tensorflow.org/images/tf_logo_32px.png"/>View on TensorFlow.org</a></td>
# <td><a target="_blank" href="https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb">
# <img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a></td>
# <td><a target="_blank" href="https://github.com/tensorflow/tfx/tree/master/docs/tutorials/tfx/template.ipynb">
# <img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">View source on GitHub</a></td>
# <td><a href="https://storage.googleapis.com/tensorflow_docs/tfx/docs/tutorials/tfx/template.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a></td>
# </table></div>

# ## Introduction
# 
# This document will provide instructions to create a TensorFlow Extended (TFX) pipeline
# using *templates* which are provided with TFX Python package.
# Many of the instructions are Linux shell commands, which will run on an AI Platform Notebooks instance. Corresponding Jupyter Notebook code cells which invoke those commands using `!` are provided.
# 
# You will build a pipeline using [Taxi Trips dataset](
# https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
# released by the City of Chicago. We strongly encourage you to try building
# your own pipeline using your dataset by utilizing this pipeline as a baseline.
# 

# ## Step 1. Set up your environment.
# 
# AI Platform Pipelines will prepare a development environment to build a pipeline, and a Kubeflow Pipeline cluster to run the newly built pipeline.
# 
# **NOTE:** To select a particular TensorFlow version, or select a GPU instance, create a TensorFlow pre-installed instance in AI Platform Notebooks.
# 

# Install `tfx` python package with `kfp` extra requirement.

# In[1]:


#print("hi")


# In[2]:


get_ipython().system('pip uninstall tensorflow --yes')
get_ipython().system('pip uninstall tensorflow-io --yes')


# In[11]:


get_ipython().system('pip install tensorflow-gpu')
get_ipython().system('pip install --no-deps tensorflow-io')
# !pip install numpy


# In[17]:


import sys
# Use the latest version of pip.
get_ipython().system('pip install --upgrade pip')
# Install tfx and kfp Python packages.
get_ipython().system('pip install --upgrade "tfx[kfp]<2"')


# Let's check the versions of TFX.

# In[18]:


get_ipython().system('python3 -c "from tfx import version ; print(\'TFX version: {}\'.format(version.__version__))"')


# In AI Platform Pipelines, TFX is running in a hosted Kubernetes environment using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/).
# 
# Let's set some environment variables to use Kubeflow Pipelines.
# 
# First, get your GCP project ID.

# In[19]:


# Read GCP project id from env.
shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.project)' 2>/dev/null")
GOOGLE_CLOUD_PROJECT=shell_output[0]
get_ipython().run_line_magic('env', 'GOOGLE_CLOUD_PROJECT={GOOGLE_CLOUD_PROJECT}')
print("GCP project ID:" + GOOGLE_CLOUD_PROJECT)


# We also need to access your KFP cluster. You can access it in your Google Cloud Console under "AI Platform > Pipeline" menu. The "endpoint" of the KFP cluster can be found from the URL of the Pipelines dashboard, or you can get it from the URL of the Getting Started page where you launched this notebook. Let's create an `ENDPOINT` environment variable and set it to the KFP cluster endpoint. **ENDPOINT should contain only the hostname part of the URL.** For example, if the URL of the KFP dashboard is `https://1e9deb537390ca22-dot-asia-east1.pipelines.googleusercontent.com/#/start`, ENDPOINT value becomes `1e9deb537390ca22-dot-asia-east1.pipelines.googleusercontent.com`.
# 
# >**NOTE: You MUST set your ENDPOINT value below.**

# In[19]:


# This refers to the KFP cluster endpoint

ENDPOINT='https://167fbc89e032e2bb-dot-us-central1.pipelines.googleusercontent.com' # Enter your ENDPOINT here.
if not ENDPOINT:
    from absl import logging
    logging.error('Set your ENDPOINT in this cell.')


# Set the image name as `tfx-pipeline` under the current GCP project.

# In[20]:


# Docker image name for the pipeline image.
CUSTOM_TFX_IMAGE='gcr.io/' + GOOGLE_CLOUD_PROJECT + '/tfx-pipeline'


# In[22]:


#pip install -U tensorflow


# In[21]:


PATH = get_ipython().run_line_magic('env', 'PATH')
get_ipython().run_line_magic('env', 'PATH={PATH}:/home/jupyter/imported')


# And, it's done. We are ready to create a pipeline.

# ## Step 2. Copy the predefined template to your project directory.
# 
# In this step, we will create a working pipeline project directory and files by copying additional files from a predefined template.
# 
# You may give your pipeline a different name by changing the `PIPELINE_NAME` below. This will also become the name of the project directory where your files will be put.

# In[22]:


PIPELINE_NAME="pipelines"
import os
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
print(PROJECT_DIR)


# TFX includes the `taxi` template with the TFX python package. If you are planning to solve a point-wise prediction problem, including classification and regresssion, this template could be used as a starting point.
# 
# The `tfx template copy` CLI command copies predefined template files into your project directory.

# In[23]:


get_ipython().system('tfx template copy   --pipeline-name={PIPELINE_NAME}   --destination-path={PROJECT_DIR}   --model=taxi')


# Change the working directory context in this notebook to the project directory.

# In[24]:


get_ipython().run_line_magic('cd', '{PROJECT_DIR}')


# >NOTE: Don't forget to change directory in `File Browser` on the left by clicking into the project directory once it is created.

# ## Step 3. Browse your copied source files
# 
# The TFX template provides basic scaffold files to build a pipeline, including Python source code, sample data, and Jupyter Notebooks to analyse the output of the pipeline. The `taxi` template uses the same *Chicago Taxi* dataset and ML model as the [Airflow Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/airflow_workshop).
# 
# Here is brief introduction to each of the Python files.
# -   `pipeline` - This directory contains the definition of the pipeline
#     -   `configs.py` — defines common constants for pipeline runners
#     -   `pipeline.py` — defines TFX components and a pipeline
# -   `models` - This directory contains ML model definitions.
#     -   `features.py`, `features_test.py` — defines features for the model
#     -   `preprocessing.py`, `preprocessing_test.py` — defines preprocessing
#         jobs using `tf::Transform`
#     -   `estimator` - This directory contains an Estimator based model.
#         -   `constants.py` — defines constants of the model
#         -   `model.py`, `model_test.py` — defines DNN model using TF estimator
#     -   `keras` - This directory contains a Keras based model.
#         -   `constants.py` — defines constants of the model
#         -   `model.py`, `model_test.py` — defines DNN model using Keras
# -   `local_runner.py`, `kubeflow_runner.py` — define runners for each orchestration engine
# 

# You might notice that there are some files with `_test.py` in their name. These are unit tests of the pipeline and it is recommended to add more unit tests as you implement your own pipelines.
# You can run unit tests by supplying the module name of test files with `-m` flag. You can usually get a module name by deleting `.py` extension and replacing `/` with `.`.  For example:

# In[14]:


get_ipython().system('{sys.executable} -m models.features_test')
get_ipython().system('{sys.executable} -m models.keras.model_test')


# In[13]:


ls -l /usr/bin/python


# ## Step 4. Run your first TFX pipeline
# 
# Components in the TFX pipeline will generate outputs for each run as [ML Metadata Artifacts](https://www.tensorflow.org/tfx/guide/mlmd), and they need to be stored somewhere. You can use any storage which the KFP cluster can access, and for this example we will use Google Cloud Storage (GCS). A default GCS bucket should have been created automatically. Its name will be `<your-project-id>-kubeflowpipelines-default`.
# 

# Let's upload our sample data to GCS bucket so that we can use it in our pipeline later.

# In[11]:


get_ipython().system('gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv')


# Let's create a TFX pipeline using the `tfx pipeline create` command.
# 
# >Note: When creating a pipeline for KFP, we need a container image which will be used to run our pipeline. And `skaffold` will build the image for us. Because skaffold pulls base images from the docker hub, it will take 5~10 minutes when we build the image for the first time, but it will take much less time from the second build.

# In[16]:


get_ipython().system('pip install markupsafe==2.0.1')


# In[25]:


get_ipython().system('tfx pipeline create  --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT} --build-image')


# While creating a pipeline, `Dockerfile` will be generated to build a Docker image. Don't forget to add it to the source control system (for example, git) along with other source files.
# 
# NOTE: `kubeflow` will be automatically selected as an orchestration engine if `airflow` is not installed and `--engine` is not specified.
# 
# Now start an execution run with the newly created pipeline using the `tfx run create` command.

# In[26]:


get_ipython().system('tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}')


# Or, you can also run the pipeline in the KFP Dashboard.  The new execution run will be listed under Experiments in the KFP Dashboard.  Clicking into the experiment will allow you to monitor progress and visualize the artifacts created during the execution run.

# However, we recommend visiting the KFP Dashboard. You can access the KFP Dashboard from the Cloud AI Platform Pipelines menu in Google Cloud Console. Once you visit the dashboard, you will be able to find the pipeline, and access a wealth of information about the pipeline.
# For example, you can find your runs under the *Experiments* menu, and when you open your execution run under Experiments you can find all your artifacts from the pipeline under *Artifacts* menu.
# 
# >Note: If your pipeline run fails, you can see detailed logs for each TFX component in the Experiments tab in the KFP Dashboard.
#     
# One of the major sources of failure is permission related problems. Please make sure your KFP cluster has permissions to access Google Cloud APIs. This can be configured [when you create a KFP cluster in GCP](https://cloud.google.com/ai-platform/pipelines/docs/setting-up), or see [Troubleshooting document in GCP](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting).

# ## Step 5. Add components for data validation.
# 
# In this step, you will add components for data validation including `StatisticsGen`, `SchemaGen`, and `ExampleValidator`. If you are interested in data validation, please see [Get started with Tensorflow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started).
# 
# >**Double-click to change directory to `pipeline` and double-click again to open `pipeline.py`**. Find and uncomment the 3 lines which add `StatisticsGen`, `SchemaGen`, and `ExampleValidator` to the pipeline. (Tip: search for comments containing `TODO(step 5):`).  Make sure to save `pipeline.py` after you edit it.
# 
# You now need to update the existing pipeline with modified pipeline definition. Use the `tfx pipeline update` command to update your pipeline, followed by the `tfx run create` command to create a new execution run of your updated pipeline.
# 

# In[41]:


# Update the pipeline
get_ipython().system('tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT}')
# You can run the pipeline the same way.
get_ipython().system('tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}')


# ### Check pipeline outputs
# 
# Visit the KFP dashboard to find pipeline outputs in the page for your pipeline run. Click the *Experiments* tab on the left, and *All runs* in the Experiments page. You should be able to find the latest run under the name of your pipeline.

# ## Step 6. Add components for training.
# 
# In this step, you will add components for training and model validation including `Transform`, `Trainer`, `Resolver`, `Evaluator`, and `Pusher`.
# 
# >**Double-click to open `pipeline.py`**. Find and uncomment the 5 lines which add `Transform`, `Trainer`, `Resolver`, `Evaluator` and `Pusher` to the pipeline. (Tip: search for `TODO(step 6):`)
# 
# As you did before, you now need to update the existing pipeline with the modified pipeline definition. The instructions are the same as Step 5. Update the pipeline using `tfx pipeline update`, and create an execution run using `tfx run create`.
# 

# In[42]:


get_ipython().system('tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT}')
get_ipython().system('tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}')


# When this execution run finishes successfully, you have now created and run your first TFX pipeline in AI Platform Pipelines!
# 
# **NOTE:** If we changed anything in the model code, we have to rebuild the
# container image, too. We can trigger rebuild using `--build-image` flag in the
# `pipeline update` command.
# 
# **NOTE:** You might have noticed that every time we create a pipeline run, every component runs again and again even though the input and the parameters were not changed.
# It is waste of time and resources, and you can skip those executions with pipeline caching. You can enable caching by specifying `enable_cache=True` for the `Pipeline` object in `pipeline.py`.
# 

# ## Step 7. (*Optional*) Try BigQueryExampleGen
# 
# [BigQuery](https://cloud.google.com/bigquery) is a serverless, highly scalable, and cost-effective cloud data warehouse. BigQuery can be used as a source for training examples in TFX. In this step, we will add `BigQueryExampleGen` to the pipeline.
# 
# >**Double-click to open `pipeline.py`**. Comment out `CsvExampleGen` and uncomment the line which creates an instance of `BigQueryExampleGen`. You also need to uncomment the `query` argument of the `create_pipeline` function.
# 
# We need to specify which GCP project to use for BigQuery, and this is done by setting `--project` in `beam_pipeline_args` when creating a pipeline.
# 
# >**Double-click to open `configs.py`**. Uncomment the definition of `GOOGLE_CLOUD_REGION`, `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` and `BIG_QUERY_QUERY`. You should replace the region value in this file with the correct values for your GCP project.
# 
# >**Note: You MUST set your GCP region in the `configs.py` file before proceeding.**
# 
# >**Change directory one level up.** Click the name of the directory above the file list. The name of the directory is the name of the pipeline which is `my_pipeline` if you didn't change.
# 
# >**Double-click to open `kubeflow_runner.py`**. Uncomment two arguments, `query` and `beam_pipeline_args`, for the `create_pipeline` function.
# 
# Now the pipeline is ready to use BigQuery as an example source. Update the pipeline as before and create a new execution run as we did in step 5 and 6.

# In[ ]:


get_ipython().system('tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT}')
get_ipython().system('tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}')


# ## Step 8. (*Optional*) Try Dataflow with KFP
# 
# Several [TFX Components uses Apache Beam](https://www.tensorflow.org/tfx/guide/beam) to implement data-parallel pipelines, and it means that you can distribute data processing workloads using [Google Cloud Dataflow](https://cloud.google.com/dataflow/). In this step, we will set the Kubeflow orchestrator to use dataflow as the data processing back-end for Apache Beam.
# 
# >**Double-click `pipeline` to change directory, and double-click to open `configs.py`**. Uncomment the definition of `GOOGLE_CLOUD_REGION`, and `DATAFLOW_BEAM_PIPELINE_ARGS`.
# 
# >**Change directory one level up.** Click the name of the directory above the file list. The name of the directory is the name of the pipeline which is `my_pipeline` if you didn't change.
# 
# >**Double-click to open `kubeflow_runner.py`**. Uncomment `beam_pipeline_args`. (Also make sure to comment out current `beam_pipeline_args` that you added in Step 7.)
# 
# Now the pipeline is ready to use Dataflow. Update the pipeline and create an execution run as we did in step 5 and 6.

# In[ ]:


get_ipython().system('tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT}')
get_ipython().system('tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}')


# You can find your Dataflow jobs in [Dataflow in Cloud Console](http://console.cloud.google.com/dataflow).
# 

# ## Step 9. (*Optional*) Try Cloud AI Platform Training and Prediction with KFP
# 
# TFX interoperates with several managed GCP services, such as [Cloud AI Platform for Training and Prediction](https://cloud.google.com/ai-platform/). You can set your `Trainer` component to use Cloud AI Platform Training, a managed service for training ML models. Moreover, when your model is built and ready to be served, you can *push* your model to Cloud AI Platform Prediction for serving. In this step, we will set our `Trainer` and `Pusher` component to use Cloud AI Platform services.
# 
# >Before editing files, you might first have to enable *AI Platform Training & Prediction API*.
# 
# >**Double-click `pipeline` to change directory, and double-click to open `configs.py`**. Uncomment the definition of `GOOGLE_CLOUD_REGION`, `GCP_AI_PLATFORM_TRAINING_ARGS` and `GCP_AI_PLATFORM_SERVING_ARGS`. We will use our custom built container image to train a model in Cloud AI Platform Training, so we should set `masterConfig.imageUri` in `GCP_AI_PLATFORM_TRAINING_ARGS` to the same value as `CUSTOM_TFX_IMAGE` above.
# 
# >**Change directory one level up, and double-click to open `kubeflow_runner.py`**. Uncomment `ai_platform_training_args` and `ai_platform_serving_args`.
# 
# Update the pipeline and create an execution run as we did in step 5 and 6.

# In[ ]:


get_ipython().system('tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint={ENDPOINT}')
get_ipython().system('tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}')


# You can find your training jobs in [Cloud AI Platform Jobs](https://console.cloud.google.com/ai-platform/jobs). If your pipeline completed successfully, you can find your model in [Cloud AI Platform Models](https://console.cloud.google.com/ai-platform/models).

# ## Step 10. Ingest YOUR data to the pipeline
# 
# We made a pipeline for a model using the Chicago Taxi dataset. Now it's time to put your data into the pipeline.
# 
# Your data can be stored anywhere your pipeline can access, including GCS, or BigQuery. You will need to modify the pipeline definition to access your data.
# 
# 1. If your data is stored in files, modify the `DATA_PATH` in `kubeflow_runner.py` or `local_runner.py` and set it to the location of your files. If your data is stored in BigQuery, modify `BIG_QUERY_QUERY` in `pipeline/configs.py` to correctly query for your data.
# 1. Add features in `models/features.py`.
# 1. Modify `models/preprocessing.py` to [transform input data for training](https://www.tensorflow.org/tfx/guide/transform).
# 1. Modify `models/keras/model.py` and `models/keras/constants.py` to [describe your ML model](https://www.tensorflow.org/tfx/guide/trainer).
#   - You can use an estimator based model, too. Change `RUN_FN` constant to `models.estimator.model.run_fn` in `pipeline/configs.py`.
# 
# Please see [Trainer component guide](https://www.tensorflow.org/tfx/guide/trainer) for more introduction.

# ## Cleaning up
# 
# To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.
# 
# Alternatively, you can clean up individual resources by visiting each consoles:
# - [Google Cloud Storage](https://console.cloud.google.com/storage)
# - [Google Container Registry](https://console.cloud.google.com/gcr)
# - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
# 
