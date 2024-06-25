# model evaluation
import numpy as np
import pickle

def evaluate_model(model,test_dataset,validation_dataset):
    
    # model evaltuation on test and val dataset
    model1 = model
    _, binary_acc1 = model1.evaluate(test_dataset)
    _, binary_acc2 = model1.evaluate(validation_dataset)

    print(f"Categorical accuracy on the test set: {round(binary_acc1 * 100, 2)}%.")
    print(f"Categorical accuracy on the validation set: {round(binary_acc2 * 100, 2)}%.")

    return True


def invert_multi_hot(encoded_labels):

    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""

    # Load the vocabulary
    with open("models/vocab.pkl", "rb") as f:
        loaded_vocab = pickle.load(f)
    
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)


def predict_category(abstract, model, vectorizer, label_lookup):
    # Preprocess the abstract using the loaded text vectorizer
    preprocessed_abstract = vectorizer([abstract])

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_abstract)

    # Convert predictions to human-readable labels
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels