"""
Steps to run the End to End Workflow for Model Training

If Project/model name exist in MarkovML App,then they will be fetched.
Otherwise new Project and Model will be created
Steps to get the Project name
    * Go to URL:https://app.markovml.com
    * Login to your Account
    * Click on the Projects tab
    * Select the Project which you want to use for the Model Training
    * Copy the Project name which we will use in the below code

PROJECT_NAME = "YOUR Test Project"
Steps to get the Model name
    * Choose a model name for your model
    * A new model will be created with this name if it doesn't exist
    * If a model with the same name already exists, we will use that one

MODEL_NAME = "Pytorch News Classifier"
Once we successfully run the custom model, we can check it out on the UI
    * Go to URL:https://app.markovml.com
    * Login to your account if not already done
    * In the left nav bar, go to the "Models" tab
    * From the top menu, choose the "All Models" table
    * You can find your model with MODEL_NAME with a button to "Generate App"
    * Click on "Generate App"
    * Enter the name of the app, and provide the number of minutes for which you want to run the app
    * Wait for the app to start, it will take 5-7 minutes
    * You can start using the app by providing any input

"""
import os.path

import markov
import pandas as pd

# markov imports
from markov.api.models.artifacts.base import (
    MarkovPredictor,
    MarkovPyfunc,
    infer_schema_and_samples_from_dataframe,
)
from markov.api.models.artifacts.inference_pipeline import InferencePipeline
from markov.library.dependencies_helper import pytorch_pip_requirements
from markov.library.mlflow_helper import MarkovSupportedFlavours
from train_model import get_trained_model

# code to train the model. In the example we have sample code to train a model in `train_model.py`


# This is the code to train your custom-model. In this example we have trained a custom model on AG NEWS data .
# The code is in `train_model.py` file in the model_examples folder
model = get_trained_model()

# To create a model app you'll need to provide samples from your test/train set.
# You can sample some rows from your test/train dataframe to register with MarkovML.
samples = ["Generative AI has been impacting the industry trends at a very fast pace."]
# Note the content here is the feature column in AG News Dataset,
# you just need to provide a dataframe with a few examples.
sample_input = pd.DataFrame([{"content": samples}])

# You can use the utility `infer_schema_and_samples_from_dataframe` to convert your input dataframe
# into schema and samples in the format accepted by Markov backend
schema, samples = infer_schema_and_samples_from_dataframe(sample_input)

# This is your inference pipeline
my_inference_model = InferencePipeline(
    name="pytorch-text-classifier-demo",
    schema=schema,    # mandatory
    samples=samples,  # optional
)


# This is an `optional step`. This is required if you want to do any post-processing on top
# of your model inference. For example, your model is returning 0/1, and you want to map it to NEGATIVE/POSITIVE
def post_process(prediction):
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    prediction_int = prediction.argmax(1).item() + 1
    return ag_news_label[prediction_int]


def get_current_directory_path():
    from inspect import getsourcefile
    directory_path = os.path.basename(os.path.abspath(getsourcefile(lambda : 0)))
    return directory_path


# Add stages to the Inference Pipeline
my_inference_model.add_pipeline_stage(
    stage=MarkovPredictor(
        name="pytorch_predictor", model=model, flavour=MarkovSupportedFlavours.PYTORCH
    )
).add_pipeline_stage(
    stage=MarkovPyfunc(name="post_process", pyfunc=post_process)
)

# Optional
my_inference_model.add_pip_requirements(
    pytorch_pip_requirements()
).add_dependent_code(
    code_paths=[os.path.join(get_current_directory_path(), 'train_model.py')]
)


# MarkovML helps organize your models into projects. To do so, you can use an existing project or create new.
# This is a helper function to create a new project of fetch an existing one.
# More details are here: https://developer.markovml.com/docs/start-with-a-project
def get_or_create_project(project_name, description: str = ""):
    """
    This is a helper function to get the project by name and if the project does not exist, create one with that name.
    :param project_name: Project name (string)
    :param description: Friendly description of the Project and its purpose. This is helpful for other members to
    know the purpose and helps in easy discovery.
    :return: Project object
    """
    try:
        mkv_project = markov.Project.get_by_name(project_name=project_name)
    except markov.exceptions.ResourceNotFoundException:
        mkv_project = markov.Project(name=project_name, description=description)
        mkv_project.register()
    return mkv_project


def get_or_create_model(project_id, model_name, description: str = ""):
    """
    This is a helper function to get the project by name and if the project does not exist, create one with that name.
    :param project_id: The project in which the model needs be fetched / created
    :param model_name: Model name (string)
    :param description: Friendly description of the Model and its purpose. This is helpful for other members to
    know the purpose and helps in easy discovery.
    :return: Model object
    """
    try:
        mkv_project = markov.Model.get_by_name(project_id, model_name=model_name)
    except markov.exceptions.ResourceNotFoundException:
        mkv_project = markov.Model(name=model_name, project_id=project_id, description=description)
        mkv_project.register()
    return mkv_project


PROJECT_NAME = "demo_ag_news"


project = get_or_create_project(
    PROJECT_NAME, description="This is a demo project to work on AG Dataset"
)


MODEL_NAME = "News-Classifier-Pytorch"

# Create a Model Metadata on MarkovML
mkv_model = get_or_create_model(
    project_id=project.project_id,
    model_name=MODEL_NAME,
    description="This model using Pytorch Text classifier to train a model on AG dataset",
)
mkv_model.register()  # this registers the model metadata with MarkovML

# register the model artifact (YOUR TRAINED MODEL)
my_inference_model.register(model_id=mkv_model.model_id)
