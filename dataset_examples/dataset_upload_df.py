import pandas as pd
from sklearn.model_selection import train_test_split

from markov.api.data.data_family import DataFamily
from markov.api.data.data_set import DataSet, DataSetRegistrationResponse
from markov.api.mkv_constants import DataCategory

# STEP 1
# Datafamily is the logical collection of datasets that serve a single purpose.
# More details here: https://developer.markovml.com/docs/datasets-data-families
# NOTE: Only use this step if you want to create a new datafamily. If you are using an existing data-family, you can
# use the id directly as shown in step 3

# STEP 1
data_family = DataFamily(
    notes="Example Data family for Markovml Datasets. You can describe your dataset here for future reference. ",
    name="MarkovMLExampleFamily",  # Give unique name to your dataset
)
df_response = data_family.register()

# STEP 2
# Preparing dataframe to be uploaded
# You can also download from the below link or read your data into a dataframe to upload to MarkovML
df = pd.read_csv(
    "https://platform-assets.markovml.com/datasets/sample/twitter_sentiment.csv"
)
# We are splitting dataset into test and train to save as different segments. If you have already segmented your dataset
# you can use test / train dataframe
# If your dataset is not split, you can register that as unsplit segment
train_df, test_df = train_test_split(df, test_size=0.2)

# STEP 3
# Create final dataset object formed from dataframe as datasource to upload
# Select the x_col_names which are the features and the y_name as the target while registering the dataset
data_set = DataSet.from_dataframe(
    df_id=df_response.df_id,  # data family id
    x_col_names=["tweet"],  # features column
    y_name="sentiment",  # target column
    delimiter=",",  # delimiter used in the dataset
    name="DataframeUploadSDK_01",  # dataset name (should be unique)
    data_category=DataCategory.Text,
    # dataset category (Text, Numeric, Categorical) (
    # here if any of feature column is text we categorize it as Text data category)
    train_source=train_df,  # train dataset segment, set to None if your dataset is unsplit
    test_source=test_df,  # train dataset segment, set to None if your dataset is unsplit
    unsplit_source=None  # test dataset segment  # if your dataset is unsplit use unsplit_source = unsplit_df
)
# Register Dataset
ds_response: DataSetRegistrationResponse = data_set.upload()
print(ds_response)
