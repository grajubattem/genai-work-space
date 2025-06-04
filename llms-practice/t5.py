'''lib required
pip install streamlit transformers torch
pip install sentencepiece'''
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"  # Can be changed to "t5-base" or "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("GenAI Content Generator using T5")
st.subheader("Generate high-quality content based on your topic")

# User Input
topic = st.text_input("Enter your topic:", placeholder="E.g., The impact of AI in Healthcare")
max_length = st.slider("Select max output length", min_value=50, max_value=500, value=200)

if st.button("Generate Content"):
    if topic:
        with st.spinner("Generating content... Please wait ⏳"):
            # Prepare the input
            input_text = f"generate a detailed article about: {topic}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt")

            # Generate output
            output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Display Result
            st.success("Content Generated Successfully!")
            st.write(generated_text)

            # Add Copy Button
            st.download_button(" Download Content", generated_text, file_name="generated_content.txt")
    else:
        st.error(" Please enter a topic before generating content.")

# run this file
#streamlit run t5.py        