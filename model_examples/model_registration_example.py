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

PROJECT_NAME = "Govini Test Project"
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
import markov
# code to train the model. In the example we have sample code to train a model in `train_model.py`

# markov imports
from markov.api.models.artifacts.base import (
    MarkovPredictor,
    MarkovPyfunc,
    infer_schema_from_dataframe,
)
from markov.api.models.artifacts.inference_pipeline import InferencePipeline
from markov.library.dependencies_helper import pytorch_pip_requirements
from markov.library.mlflow_helper import MarkovSupportedFlavours
import pandas as pd
from train_model import get_trained_model

# THis is the code to train your model. In this example we have trained a custom model on AG NEWS data .
# The code is in `train_model.py`
model = get_trained_model()

# To create a model app you'll need to provide samples from your test/train set.
samples = ["Generative AI has been impacting the industry trends at a very fast pace."]
sample_input = pd.DataFrame([{"content": samples}])

# This is your inference pipeline
my_inference_model = InferencePipeline(
    name="pytorch-text-classifier-demo",
    schema=infer_schema_from_dataframe(sample_input),
    samples=samples,
)


# This is an `optional step`. This is required if you want to do any type of post-processing on top
# of your model inference. For example your model is returning 0/1 and you want to map it to NEGATIVE/POSITIVE
def post_process(prediction):
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    prediction_int = prediction.argmax(1).item() + 1
    return ag_news_label[prediction_int]


# Add stages to the Inference Pipeline
my_inference_model.add_pipeline_stage(
    stage=MarkovPredictor(
        name="pytorch_predictor", model=model, flavour=MarkovSupportedFlavours.PYTORCH
    )
).add_pipeline_stage(
    stage=MarkovPyfunc(name="post_process", pyfunc=post_process)
).add_pip_requirements(
    pytorch_pip_requirements()
)


# MarkovML helps organize your models into projects. To do so, you can use an existing project or create new.

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


project = get_or_create_project('demo_ag_news', description='This is a demo project to work on AG Dataset')

# Create a Model Metadata on MarkovML
mkv_model = project.create_model(
    model_name='<YOUR_MODEL_NAME>',
    model_description='This model using Pytorch Text classifier to train a model on AG dataset'
)
mkv_model.register()  # this registers the model metadata with MarkovML

# register the model artifact (YOUR TRAINED MODEL)
my_inference_model.register(model_id=mkv_model.model_id)
