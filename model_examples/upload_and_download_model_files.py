from typing import List

try:
    import sklearn
except ImportError:
    print("Please make sure that scikit-learn is installed to run this example")
    import sys
    sys.exit(1)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    import joblib
except ImportError:
    print("Please make sure that joblib is installed to run this example")
    import sys
    sys.exit(1)

import os
import markov
from markov.data_models.model import ModelRegistryStageStates

# get_dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# save model locally
local_model_filename = 'rf_iris_model.joblib'
joblib.dump(clf, local_model_filename)


# Use markov to register to track the model files

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
        mkv_model = markov.Model.get_by_name(project_id, model_name=model_name)
    except markov.exceptions.ResourceNotFoundException:
        mkv_model = markov.Model(name=model_name, project_id=project_id, description=description)
        mkv_model.register()
    return mkv_model


PROJECT_NAME = "Demo RF Classifiers"
MODEL_NAME = "RF Classifier for Iris Dataset"

project = get_or_create_project(project_name=PROJECT_NAME)
mkv_model = get_or_create_model(project_id=project.project_id, model_name=MODEL_NAME)

mkv_model.upload_model_files(local_file_paths=local_model_filename)

# feedback: show the location of uploaded files on UI

# Download the files from markov
# You can run the following in a separate file
loaded_mkv_model = markov.Model.get_by_name(project_id=project.project_id, model_name=MODEL_NAME)


download_model_path = os.path.abspath('./downloads')
if not os.path.exists(download_model_path):
    os.mkdir(path=download_model_path)

loaded_mkv_model.download_model_files(local_destination_path=download_model_path)

# load your model using your own loaders
trained_model = joblib.load(os.path.join(download_model_path, local_model_filename))
y_pred = trained_model.predict(X_test)


# Model registry operations
if markov.__version__ <= '2.0.2':
    import logging
    import sys
    logging.warning("Model registry operations run only on markov version >= 2.0.3")
    sys.exit(0)

REGISTRY_NAME = "Test Registry"

# Note: Creation of registry is through UI
# Going forward: Creation of registry supported SDK as well

loaded_mkv_model.link_to_registry(registry_name=REGISTRY_NAME)

loaded_mkv_model.update_model_stage_in_registry(to_stage=ModelRegistryStageStates.DEV)

# other available commands:
# ModelRegistry.get_all()
# ModelRegistry.get_by_id(registry_id="")
# ModelRegistry.get_by_name(registry_name="")

loaded_mkv_model.add_metadata(
    metadata={
        "tag": "latest",
        "fix_version": "2024.02.12"
    }
)
retrieved_models: List[markov.Model] = markov.Model.get_models_by_metadata({"tag": "latest"})


