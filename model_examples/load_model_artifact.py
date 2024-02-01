from model_registration_example import MODEL_NAME, PROJECT_NAME
import pandas

from markov.api.models.artifacts.inference_pipeline import InferencePipeline
import markov

markov_project = markov.Project.get_by_name(PROJECT_NAME)

markov_model = markov.Model.get_by_name(project_id=markov_project.project_id, model_name=MODEL_NAME)

loaded_inference_pipeline = InferencePipeline.load_inference_pipeline(
    model_id=markov_model.model_id
)

print(loaded_inference_pipeline._samples)
loaded_inference_pipeline.predict_samples()

new_input = "Microsoft on Thursday briefly unseated Apple as the world’s most valuable publicly traded company in early trading."

df = pandas.DataFrame(
    [{
        "content": new_input
    }]
)

loaded_inference_pipeline.predict(df)
