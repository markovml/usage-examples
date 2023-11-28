import random

import markov
from markov import EmbeddingRecorder

"""
This example demonstrates how to upload a custom embedding for a registered dataset with Markov.
To run this recorder you'll need your dataframe in this format 
_______________________
|FEATURES  | EMBEDDING|
|_____________________|
|__________|__________|
|__________|__________|
"""

# STEP 1 -> GET DATASET
# this is the dataset for which you want to upload custom embedding that you have generated
dataset = markov.dataset.get_by_id("<YOUR_DATASET_ID>")
# Get the dataframe you want to upload embedding for
df = dataset.unsplit.as_df()

# STEP 2 -> CREATE EMBEDDING RECORDER to upload embedding to MarkovML
embedding_recorder = EmbeddingRecorder(
    name="<Add notes on this embedding, for example how it was generated etc.>", dataset_id=" <YOUR_DATASET_ID>"
)
embedding_recorder.register()

# STEP 3 -> REGISTER EMBEDDING WITH RECORDER
# Note we are simulating embedding generation,You can pass the embeddings as part of your embedding generation pipeline
# all we need is  "your original features", "embedding"
# Note that we do not store your original features, we hash them to generate an identifier to align your
# embeddings against a specific point in your dataset registered with MarkovML
for i in range(2000):
    # read each row
    dataset_record = df[embedding_recorder.ds_columns].iloc[i].tolist()
    # pass the original features & corresponding embedding to recorder
    embedding_recorder.add_embedding_record(
        dataset_record, [random.uniform(-1, 1) for _ in range(10)]
    )
# Finish should be called to close the recorder. If recorder is not closed, embedding would not be displayed on UI
embedding_recorder.finish()
