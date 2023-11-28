import markov
from markov.api.data.data_family import DataFamily
from markov.api.data.data_set import DataSet, DataSetRegistrationResponse
from markov.api.mkv_constants import DataCategory

"""
This example explains how to register a dataset in your S3 bucket safely with MarkovML. 
Your data stays where it is and is only processed in memory when an analysis is triggered. 
If you are an enterprise customer with Hybrid Deployment, your data is also analyzed within your VPC.
"""

# STEP 1
# Datafamily is the logical collection of datasets that serve a single purpose.
# More details here: https://developer.markovml.com/docs/datasets-data-families
# NOTE: Only use this step if you want to create a new datafamily. If you are using an existing data-family, you can
# use the id directly as shown in step 3
data_family = DataFamily(
    notes="Example Data family for Markovml Datasets. You can describe your dataset here for future reference. ",
    name="MarkovMLExampleFamily",  # Give unique name to your dataset
)
# This step actually registers a datafamily
df_response = data_family.register()

# STEP 2
# Register the creds to fetch dataset from s3 store. These credentials should have read access from your S3
# NOTE: If you have already registered credentials with MarkovML, you can ignore this step and move to STEP 3
cred_response = markov.credentials.register_s3_credentials(
    name="YOUR_CREDENTIAL_NAME",
    access_key="<ACCESS KEY>",
    access_secret="<ACCESS SECRET>",
    notes="Creds to access datasets for cloud upload",
)
# Use the newly registered credentials, or copy the credentials from your dataset details page.
cred_id = cred_response.credential_id

# STEP 3
# Create final dataset object formed from cloud upload path
# Select the x_col_names which are the features and the y_name as the target while registering the dataset
data_set = DataSet.from_cloud(
    df_id=df_response.df_id,  # data family id
    x_col_names=["tweet"],  # feature columns
    y_name="sentiment",  # target column
    delimiter=",",  # delimiter used for the input dataset segments
    name="CloudUploadSDK_01",  # dataset name (should be unique)
    data_category=DataCategory.Text,
    # supply dataset category (Text data category) (Text: If any of the feature column is text)
    credential_id=cred_id,  # pass cred id to access the s3
    # path to the dataset segment in s3 bucket, keep none if unsplit
    train_source="s3://path_to_dataset/twitter_train.csv",
    # path to the dataset segment in s3 bucket, keep none if unsplit
    test_source="s3://path_to_dataset/twitter_test.csv",  # path to the dataset segment in s3 bucket
    unsplit_source="s3://path_to_dataset/twitter_unsplit.csv"  # set path here if your dataset is unsplit
)

# Register Dataset. This is the actual call to upload and register dataset with MarkovML
ds_response: DataSetRegistrationResponse = data_set.upload()
print(ds_response)
