import streamlit as st
from transformers import AutoTokenizer, pipeline

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Loads the fine-tuned Pegasus model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    summarization_pipe = pipeline("summarization", model=model_path, tokenizer=tokenizer)
    return summarization_pipe

# Load the model and tokenizer
MODEL_PATH = "pegasus-samsum-model"
TOKENIZER_PATH = "tokenizer"

gen_kwargs = {
    "length_penalty": 0.8,
    "num_beams": 8,
    "max_length": 128
}

# Streamlit App
st.title("Dialogue Summarizer")

st.write("Enter a dialogue conversation below, and the fine-tuned Pegasus model will generate a summary for you.")

# Input Text
user_input = st.text_area("Enter the dialogue conversation:", height=200)

# Summarization
if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Generating summary..."):
            pipe = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
            summary = pipe(user_input, **gen_kwargs)[0]["summary_text"]
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter a dialogue conversation before summarizing.")
