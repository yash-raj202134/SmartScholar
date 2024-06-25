# recommendation

import pickle
import torch # type: ignore

# This imports the SentenceTransformer class from the Sentence Transformers library.
from sentence_transformers import SentenceTransformer, util # type: ignore

def sentence_tnfs(arxiv_data):
    # we load all-MiniLM-L6-v2, which is a MiniLM model fine tuned on a large dataset of over 
    # 1 billion training pairs.
    #This initializes the 'all-MiniLM-L6-v2' model from Sentence Transformers. 
    # This model is capable of encoding sentences into fixed-size vectors (embeddings).

    model = SentenceTransformer('all-MiniLM-L6-v2')
    #Our sentences we like to encode
    sentences = arxiv_data['titles']
    
    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Saving sentences and corresponding embeddings
    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    with open('models/sentences.pkl', 'wb') as f:
        pickle.dump(sentences, f)
        
    with open('models/rec_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, embeddings , sentences 

def print_embeddings(sentences,embeddings):

    c = 0
    #This loop iterates over pairs of sentences and their corresponding embeddings. 
    #zip is used to iterate over both lists simultaneously.
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding length:", len(embedding)) # list of floats
        print("")
        # Breaks out of the loop after printing information for the first 5 sentences.
        if c >=5:
            break
        c +=1 


def recommendation(input_paper,rec_model,embeddings,sentences):

    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    
    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
                                 
    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])
    
    return papers_list
