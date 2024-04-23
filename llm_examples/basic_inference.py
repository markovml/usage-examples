import markov
from markov.api.models.llm.llm_model import LLMModel, LLMLoader, ModelSource


def generate_text():
    markov.init(api_token="**hBsHNDRrGzGD9foYf99A**")
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    new_llm = LLMModel(model_id=repo_id, model_source=ModelSource.HUGGING_FACE)
    new_llm.download_model_completion()
    new_llm.wait_till_model_download()
    new_llm.load_model_completion(loader=LLMLoader.TRANSFORMERS)
    new_llm.wait_till_model_load()
    text = new_llm.generate_text_completion(prompt="Write a 200 word paragraph on India's diversity",
                                            max_tokens=200)
    print(text)


if __name__ == "__main__":
    generate_text()
