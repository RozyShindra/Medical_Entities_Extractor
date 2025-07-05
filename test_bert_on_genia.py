import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the trained model and tokenizer
model_path = "bert-base-uncased"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Streamlit UI
st.title("🔬 Biomedical NER (GENIA)")

sentence = st.text_area("Enter biomedical sentence:")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
hide_o_labels = st.checkbox("Hide 'O' label entities", value=True)

if st.button("Extract Entities"):
    if sentence:
        results = ner_pipeline(sentence)

        filtered = []
        for r in results:
            if r['score'] >= conf_threshold:
                if hide_o_labels and r['entity_group'] == 'O':
                    continue
                filtered.append(r)

        if filtered:
            st.markdown("### 🧠 Extracted Entities")
            for ent in filtered:
                st.write(f"• **{ent['word']}** → `{ent['entity_group']}` (confidence: {ent['score']:.2f})")
        else:
            st.info("No entities above the threshold.")
    else:
        st.warning("Please enter a sentence.")
