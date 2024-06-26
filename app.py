# minimum requirements:
# tensorflow==2.15.0
# torch==2.0.1
# sentence_transformers==2.2.2
# streamlit

# import libraries
import streamlit as st # type: ignore
import pickle
from tensorflow.keras.layers import TextVectorization # type: ignore
from tensorflow import keras
from utils.commons import recommendation,predict_category,invert_multi_hot

# load save recommendation model
embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

# load save prediction models
loaded_model = keras.models.load_model("models/model.h5")

# Load the configuration of the text vectorizer
with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)

# Create a new TextVectorization layer with the saved configuration
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)

# Load the saved weights into the new TextVectorization layer
with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)




# create application interface#######

st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Base App")

input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Past paper abstract....")
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation(input_paper,rec_model,embeddings,sentences)
    st.subheader("Recommended Papers")
    st.write(recommend_papers)

    #prediction part
    st.write("===================================================================")
    predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
    st.subheader("Predicted Subject area")
    st.write(predicted_categories)