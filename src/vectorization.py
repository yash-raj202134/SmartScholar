# Text Vectorization
from tensorflow.keras import layers # type: ignore
import tensorflow as tf
import pickle 


def vectorization(train_dataset,validation_dataset,test_dataset,vocabulary_size):
    """
        Mapping Vectorization to Datasets: The code maps the text vectorization operation to 
        each element of the training, validation, and test datasets. This ensures that the text
        data in each dataset is transformed into numerical vectors using the adapted TextVectorization layer.
        The num_parallel_calls parameter is used to parallelize the mapping process, and prefetch is 
        applied to prefetch data batches 
        for better performance.
        """
    #auto = tf.data.AUTOTUNE: auto is assigned the value tf.data.AUTOTUNE,
    auto = tf.data.AUTOTUNE

    # Initializes a TextVectorization layer
    text_vectorizer = layers.TextVectorization(max_tokens=vocabulary_size,ngrams=2,output_mode="tf_idf")
    # `TextVectorization` layer needs to be adapted as per the vocabulary from our
    # training set.
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

    train_dataset = train_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)
    validation_dataset = validation_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)
    test_dataset = test_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(auto)

    # Save the configuration of the text vectorizer
    saved_text_vectorizer_config = text_vectorizer.get_config()
    with open("models/text_vectorizer_config.pkl", "wb") as f:
        pickle.dump(saved_text_vectorizer_config, f)

    # save the weights of the text vectorizer
    text_vectorizer_weights = text_vectorizer.get_weights()
    with open("models/text_vectorizer_weights.pkl", "wb") as f:
        pickle.dump(text_vectorizer_weights, f)


    return train_dataset,validation_dataset,test_dataset