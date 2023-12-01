"""
This example has the code for the following

1. Accessing Your Registered Dataset with MarkovML using SDK

2. Extracting High Quality Points from the Dataset for Training
"""
import markov
from markov.api.data.data_set import DataSet, DatasetQuality

# The dataset that is registered with Markov has a dataset_id. You can use that to fetch the dataset
ds: DataSet = markov.dataset.get_by_id(dataset_id="<YOUR_DATA_SET_ID>")

# To get the dataframe # Return the dataframe from the dataset
# Fetch the given segment (train/test/validate/unknown) as dataframe
# for example to get the train dataframe you can do df_train = ds.train.as_df()
df_stored = (
    ds.unsplit.as_df()
)  # here we did not segment dataset as such the available segment is unslit

# You can also get train, test split using this code
train_df, test_df = ds.get_train_test_split()

# Filter the dataframe with only columns that were registered as features during registration
dataset_with_features = df_stored[ds.features]

# Extract the quality details from the dataset. These details are available when Markov DataQuality has been run
ds_quality: DatasetQuality = ds.quality
quality_df = ds_quality.df

# Get a rows in the original dataset that have labeling issue
mislabeled_df = quality_df[quality_df["is_label_issue"] is True]

# Get rows in the original dataset that are high quality
high_quality_df = quality_df[quality_df["is_label_issue"] is False]
