import urllib.request

import markov
from markov.api.data.data_family import DataFamily
from markov.api.data.data_set import DataSet, DataSetRegistrationResponse
from markov.api.mkv_constants import DataCategory

# Create dataset family to tag the dataset
data_family = DataFamily(
    notes="Example Data family for Markovml Datasets. You can describe your dataset here for future reference. ",
    name="MarkovMLExampleFamily",  # Give unique name to your dataset
)
try:
    df_response = data_family.register()
except:
    df_response = markov.data.DataFamily.get_by_name("MarkovMLExampleFamily")

# Create final dataset object formed from filepath as datasource to upload
# Select the x_col_names which are the features and the y_name as the target while registering the dataset
# here data is the folder that contains your dataset.
# This example is based on Twitter sentiment dataset available here
# "https://platform-assets.markovml.com/datasets/sample/twitter_sentiment.csv"

urllib.request.urlretrieve(
    "https://platform-assets.markovml.com/datasets/sample/twitter_sentiment.csv",
    "twitter_train.csv",
)
urllib.request.urlretrieve(
    "https://platform-assets.markovml.com/datasets/sample/twitter_sentiment.csv",
    "twitter_test.csv",
)

data_set = DataSet.from_filepath(
    df_id=df_response.df_id,  # data family id
    x_col_names=["tweet"],  # features column
    y_name="sentiment",  # target column
    delimiter=",",  # delimiter used in the dataset
    name="FilepathUploadSDK_01",  # dataset name which is being used for upload
    data_category=DataCategory.Text,  # dataset category (Text: If any of the feature column is text)
    train_source="twitter_train.csv",  # train dataset segment filepath
    test_source="twitter_test.csv",  # test dataset segment filepath
)

# Register Dataset
ds_response: DataSetRegistrationResponse = data_set.upload()
print(ds_response)
