
import tensorflow as tf

from sklearn.model_selection import train_test_split # type: ignore
import pickle
from ast import literal_eval
# is used for safely evaluating strings containing Python literals or container displays
# (e.g., lists, dictionaries) to their corresponding Python objects.


def make_dataset(dataframe,train_df,batch_size, is_train=True):
        
    terms = tf.ragged.constant(train_df['terms'].values)
        # This line creates a StringLookup layer in TensorFlow. The purpose of this layer is to map strings to integer indices and vice versa. The output_mode="multi_hot" indicates that the layer will output a multi-hot encoded representation of the input strings.
    lookup = tf.keras.layers.StringLookup(output_mode='multi_hot')
        # This step adapts the StringLookup layer to the unique values in the "terms" column, building the vocabulary.
    lookup.adapt(terms)


    # creating sequences of labesls
    labels = tf.ragged.constant(dataframe["terms"].values)
    # This line uses the previously defined lookup layer to convert the ragged tensor of labels into a binarized representation. The resulting label_binarized is a NumPy array.
    label_binarized = lookup(labels).numpy()
    # creating sequences of text.
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["abstracts"].values, label_binarized))
    # shuffling data basis on condition
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset

    return dataset.batch(batch_size)


def preprocess(arxiv_data):
    
    # getting unique labels
    labels_column = arxiv_data['terms'].apply(literal_eval)
    labels = labels_column.explode().unique()


    # remove duplicate entries based on the "titles" (terms) column
    # This filters the DataFrame, keeping only the rows where the titles are not duplicated.
    arxiv_data = arxiv_data[~arxiv_data['titles'].duplicated()]

    arxiv_data_filtered = arxiv_data.groupby('terms').filter(lambda x: len(x) > 1)

    # It evaluates the given string containing a Python literal or container display (e.g., a list or dictionary) and returns the corresponding Python object.
    arxiv_data_filtered['terms'] = arxiv_data_filtered['terms'].apply(lambda x: literal_eval(x))
    arxiv_data_filtered['terms'].values[:3]


    # train and test split.

    test_split = 0.1

    # Initial train and test split.
    # The stratify parameter ensures that the splitting is done in a way that preserves the same distribution of labels (terms) in both the training and test sets.
    train_df, test_df = train_test_split(arxiv_data_filtered,test_size=test_split,stratify=arxiv_data_filtered["terms"].values,)

    # Splitting the test set further into validation
    # and new test sets.
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)

    # print(f"Number of rows in training set: {len(train_df)}")
    # print(f"Number of rows in validation set: {len(val_df)}")
    # print(f"Number of rows in test set: {len(test_df)}")

    # creates a TensorFlow RaggedTensor (terms) from the values in the "terms" column of the train_df DataFrame. A RaggedTensor is a tensor with non-uniform shapes
    terms = tf.ragged.constant(train_df['terms'].values)
    # This line creates a StringLookup layer in TensorFlow. The purpose of this layer is to map strings to integer indices and vice versa. The output_mode="multi_hot" indicates that the layer will output a multi-hot encoded representation of the input strings.
    lookup = tf.keras.layers.StringLookup(output_mode='multi_hot')
    # This step adapts the StringLookup layer to the unique values in the "terms" column, building the vocabulary.
    lookup.adapt(terms)
    # retrieve vocabulary
    vocab = lookup.get_vocabulary()

    # print("Vocabulary:\n")
    # print(vocab)

    sample_label = train_df["terms"].iloc[0]
    # print(f"Original label: {sample_label}")

    label_binarized = lookup([sample_label])
    # print(f"Label-binarized representation: {label_binarized}")


    # following lines::
    # which is used for automatic adjustment of resource usage by TensorFlow's data loading pipeline.

    #max_seqlen: Maximum sequence length. It indicates the maximum length allowed for sequences.
    max_seqlen = 150
    #batch_size: Batch size. It specifies the number of samples to use in each iteration.
    batch_size = 128
    #padding_token: A token used for padding sequences.
    padding_token = "<pad>"
  


    train_dataset = make_dataset(train_df,train_df,batch_size, is_train=True)
    validation_dataset = make_dataset(val_df,train_df,batch_size, is_train=False)
    test_dataset = make_dataset(test_df,train_df,batch_size, is_train=False)


    # This code snippet is iterating through batches of the training dataset and printing the abstract text along with the corresponding labels.
    text_batch, label_batch = next(iter(train_dataset))
    for i, text in enumerate(text_batch[:5]):
        label = label_batch[i].numpy()[None, ...]
        # print(f"Abstract: {text}")
        # # print(f"Label(s): {invert_multi_hot(label[0])}")
        # print(" ")


    # This code calculates the size of the vocabulary in the "abstracts" column of the train_df DataFrame.

    # Creating vocabulary with uniques words
    vocabulary = set()
    train_df["abstracts"].str.lower().str.split().apply(vocabulary.update)
    vocabulary_size = len(vocabulary)
    # print(vocabulary_size)


    # Save the vocabulary
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    return train_dataset, validation_dataset,test_dataset,vocabulary_size,lookup
