"""
This example has the code for the following

1. Accessing Your Registered Dataset with MarkovML using SDK

"""
import markov
from markov.api.data.data_set import DataSet

# The dataset that is registered with Markov has a dataset_id. You can use that to fetch the dataset
ds: DataSet = markov.dataset.get_by_id(dataset_id="<YOUR_DATA_SET_ID>")

# To get the dataframe # Return the dataframe from the dataset
# Fetch the given segment (train/test/validate/unknown) as dataframe
df_unsplit = (
    ds.unsplit.as_df()
)  # here we did not segment dataset as such the available segment is unknown

# filter the dataframe with only columns that were registered as features during registration
ds_features = df_unsplit[ds.features]

# you can do any further operation on the dataframe.
# fetch dataset url
print(ds.get_url())

# open dataset details page
ds.view_details()
