import streamlit as st
import torch

class NERApp:
    def __init__(self, model, tokenizer, id2label, threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.threshold = threshold

    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).logits
        probs = torch.softmax(output, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label[p.item()] for p in predictions[0]]

        return [(tok, lab) for tok, lab in zip(tokens, labels) if lab != "O"]

    def run(self):
        st.title("Genia NER Biomedical App")
        sentence = st.text_input("Enter a sentence:")
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        self.threshold = threshold

        if st.button("Extract Entities"):
            if sentence:
                results = self.predict(sentence)
                st.write("Entities Found:")
                for tok, label in results:
                    st.markdown(f"`{tok}`: **{label}**")
            else:
                st.warning("Please enter a sentence.")
