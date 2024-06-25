from utils.logging import logger
from GetData.download import DataIngestion
import pandas as pd # type: ignore
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization # type: ignore
import pickle

from src.preprocessing import preprocess
from src.vectorization import vectorization
from src.model_trainer import train_model,plot_loss
from src.model_evaluation import evaluate_model,predict_category,invert_multi_hot
from src.recommendation import sentence_tnfs,print_embeddings,recommendation

# Data ingestion stage:

data = DataIngestion()
file_path = data.download_datasets()
data.extract_datasets(file_path)

# Preprocessing stage:
arxiv_data = pd.read_csv("dataset/arxiv_data_210930-054931.csv")
train, validation, test, vocab_size, lookup = preprocess(arxiv_data = arxiv_data)

# Vectorization stage:
train,validation,test = vectorization(train,test,validation,vocabulary_size=vocab_size)

# Model training stage:
history = train_model(train_dataset=train,validation_dataset=validation,test_dataset=test,lookup=lookup)
plot_loss(history=history,item="loss",save_path="models")
plot_loss(history=history,item="binary_accuracy",save_path="models")

# Model evaluation:
model = keras.models.load_model("models/model.h5")
with open("models/text_vectorizer_config.pkl", "rb") as f: # Load the configuration of the text vectorizer
    saved_text_vectorizer_config = pickle.load(f)

text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config) # Create a new TextVectorization layer with the saved configuration
result = evaluate_model(model=model,test_dataset=test,validation_dataset=validation)

# Example usage
new_abstract = """Graph neural networks (GNNs) have been widely used to learn 
vector\nrepresentation of graph-structured data and achieved better task performance\nthan conventional methods. 
The foundation of GNNs is the message passing\nprocedure, which propagates the information in a node to its neighbors.
Since\nthis procedure proceeds one step per layer, the range of the information\npropagation among nodes is small in 
the lower layers, and it expands toward the\nhigher layers. Therefore, a GNN model has to be deep enough to capture 
global\nstructural information in a graph. On the other hand, it is known that deep GNN\nmodels suffer from performance
degradation because they lose nodes' local\ninformation, which would be essential for good model performance, through 
many\nmessage passing steps. In this study, we propose multi-level attention pooling\n(MLAP) for graph-level classification
tasks, which can adapt to both local and\nglobal structural information in a graph. It has an attention pooling layer 
for\neach message passing step and computes the final graph representation by\nunifying the layer-wise graph representations.
The MLAP architecture allows\nmodels to utilize the structural information of graphs with multiple levels of\nlocalities because
it preserves layer-wise information before losing them due\nto oversmoothing. Results of our experiments show that the MLAP
architecture\nimproves the graph classification performance compared to the baseline\narchitectures. In addition, analyses
on the layer-wise graph representations\nsuggest that aggregating information from multiple levels of localities indeed\nhas
the potential to improve the discriminability of learned graph\nrepresentations."""

predicted_categories = predict_category(new_abstract,model, text_vectorizer, invert_multi_hot)
print("Predicted Categories:", predicted_categories)

# Example usage
new_abstract = '''Deep networks and decision forests (such as random forests and gradient\nboosted trees)
are the leading machine learning methods for structured and\ntabular data, respectively. Many papers have empirically 
compared large numbers\nof classifiers on one or two different domains (e.g., on 100 different tabular\ndata settings).
However, a careful conceptual and empirical comparison of these\ntwo strategies using the most contemporary best practices 
has yet to be\nperformed. Conceptually, we illustrate that both can be profitably viewed as\n"partition and vote" schemes. 
Specifically, the representation space that they\nboth learn is a partitioning of feature space into a union of convex 
polytopes.\nFor inference, each decides on the basis of votes from the activated nodes.\nThis formulation allows for a 
unified basic understanding of the relationship\nbetween these methods. Empirically, we compare these two strategies on 
hundreds\nof tabular data settings, as well as several vision and auditory settings. Our\nfocus is on datasets with at most 
10,000 samples, which represent a large\nfraction of scientific and biomedical datasets. In general, we found forests 
to\nexcel at tabular and structured data (vision and audition) with small sample\nsizes, whereas deep nets performed better 
on structured data with larger sample\nsizes. This suggests that further gains in both scenarios may be realized via\nfurther 
combining aspects of forests and networks. We will continue revising\nthis technical report in the coming months with 
updated results.'''

predicted_categories = predict_category(new_abstract,model,text_vectorizer, invert_multi_hot)
print("Predicted Categories:", predicted_categories)


# recommendations

arxiv_data_copy = arxiv_data.copy()
arxiv_data_copy.drop(columns = ["terms","abstracts"], inplace = True)

arxiv_data_copy.drop_duplicates(inplace= True)
arxiv_data_copy.reset_index(drop= True,inplace = True)

pd.set_option('display.max_colwidth', None)

rec_model, embeddings, sentences = sentence_tnfs(arxiv_data= arxiv_data_copy)
print_embeddings(sentences=sentences,embeddings= embeddings)

# exampel usage 1: (use this paper as input (Attention is All you Need))
input_paper = input("Enter the title of any paper you like")
recommend_papers = recommendation(
    input_paper=input_paper,
    rec_model=rec_model,
    embeddings=embeddings,
    sentences=sentences
    )

print("We recommend to read this paper............")
print("=============================================")
for paper in recommend_papers:
    print(paper)

# exampel usage 2: (use this paper as input (BERT: Pre-training of Deep Bidirectional 
# Transformers for Language Understanding))
input_paper = input("Enter the title of any paper you like")
recommend_papers = recommendation(
    input_paper=input_paper,
    rec_model=rec_model,
    embeddings=embeddings,
    sentences=sentences
    )

print("We recommend to read this paper............")
print("=============================================")
for paper in recommend_papers:
    print(paper)

# exampel usage 3: (use this paper as input (Review of deep learning: concepts,
#  CNN architectures, challenges, applications, future directions))
input_paper = input("Enter the title of any paper you like")
recommend_papers = recommendation(
    input_paper=input_paper,
    rec_model=rec_model,
    embeddings=embeddings,
    sentences=sentences
    )

print("We recommend to read this paper............")
print("=============================================")
for paper in recommend_papers:
    print(paper)