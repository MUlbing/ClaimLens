from sentence_transformers import SentenceTransformer

# The pretrained model, downloaded from Hugging Face by name (same model every time)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    """Turn a list of texts into meaning vectors."""
    texts_list = list(texts)
    return embedding_model.encode(texts_list, show_progress_bar=False)
