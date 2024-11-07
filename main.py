import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from utilities import Utilities
from transformer import TransformerClassifier, TransformerLanguageModel

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    ####### Classification Task #########

    # Train Loader
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    # Test Loader
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    ####### Language Modeling #########

    # Train Loader
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Test Loaders
    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_obama = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_obama = DataLoader(test_LM_dataset_obama, batch_size=batch_size, shuffle=True)

    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_hbush = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_hbush = DataLoader(test_LM_dataset_hbush, batch_size=batch_size, shuffle=True)

    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_wbush = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_wbush = DataLoader(test_LM_dataset_wbush, batch_size=batch_size, shuffle=True)

    ###### Switch Case ########

    print("\nChoose Option from below: ")
    print("1. Classification Task")
    print("2. Language Modeling")
    print("3. ALiBi & RoPE PE Technique Exploration")
    print("4. Exit")
    part = input("Choose which part to execute : ")

    while True:
        if part == "1":
            print("\nPart 1: Classification Task (Only Encoder Model)")
            # Load the classification model
            model_classifier = TransformerClassifier(tokenizer.vocab_size, block_size, n_embd, n_head,
                                                    n_input, n_hidden, n_output, n_layer, device).to(device)

            optimizer = torch.optim.AdamW(model_classifier.parameters(), lr=learning_rate)\

            for epoch in range(epochs_CLS):
                model_classifier.train()  # Set the model to training mode
                for xb, yb in train_CLS_loader:
                    #print(xb[0])
                    xb, yb = xb.to(device), yb.to(device)
                    
                    # Forward pass
                    y_pred, loss, attention_maps = model_classifier((xb, None), yb)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # Calculate and print accuracy after each epoch
                train_accuracy = compute_classifier_accuracy(model_classifier, train_CLS_loader)
                test_accuracy = compute_classifier_accuracy(model_classifier, test_CLS_loader)
                print(f"Epoch {epoch+1}/{epochs_CLS}: Train Loss {loss.item():.3f}; Train Accuracy {train_accuracy:.3f}; Test Accuracy {test_accuracy:.3f}")

            print("\nPart 1 : Sanity check")

            # Define sample sentences
            sentence1 = "At the same time, peace brings real benefits to everyone."
            sentence2 = "And this cycle of suspicion and discord must end."
            
            # Initialize Utilities with tokenizer and classifier model
            utility_cls = Utilities(tokenizer, model_classifier)
            
            # Sanity check for Sentence 1
            print(f"Sentence 1: {sentence1}")
            utility_cls.sanity_check(sentence1, block_size, "Part1Sent1_AttnMap")
            
            # Sanity check for Sentence 2
            print(f"Sentence 2: {sentence2}")
            utility_cls.sanity_check(sentence2, block_size, "Part1Sent2_AttnMap")
            
            print("Sanity check completed!")


if __name__ == "__main__":
    main()
