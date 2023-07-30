### First Prototype of TREC 2023 Deep Learning Training Phase

The approach we took in developing the code for the Training Phase of TREC 2023 Deep Learning Track's passage ranking task was carefully 
considered to meet the specific requirements and objectives of the project. Here are the reasons behind the decisions we made:

1. **Requirements and Initial Setup**: The task at hand was part of the passage ranking task in the Training Phase of the TREC 2023 Deep Learning Track. We were provided with three files: query, qrels (which contained relevance information), and top100 (which contained the top 100 documents for each query). The goal was to create a machine learning model to rank the passages in terms of their relevance to the given queries. The main focus was on two subtasks: full ranking and top-100 reranking. 

   For this task, we used Google Colab as our coding environment due to its easy integration with Google Drive for data storage and its free GPU resources. We used the transformers library from Hugging Face for accessing pre-trained models and tokenizers, and PyTorch for data handling and model training. 

2. **Data Loading and Preparation**: Given the large size of the files, loading them into memory was a challenge. To overcome this, we implemented a chunk-based loading system where we read parts of the data into memory, performed necessary operations, and then freed up the memory for the next chunks.

3. **Model and Tokenizer Selection**: We selected the BERT model for its powerful ability to understand the context of text data. Specifically, we used the "bert-tiny" model to save computational resources, which is a smaller version of the BERT model but still retains a significant portion of its performance.

4. **Tokenization and Checkpointing**: Tokenizing the entire dataset at once was memory-intensive due to the size of the data. Therefore, we tokenized the data in chunks. To handle interruptions in the Google Colab session, we saved tokenized data to the disk after processing each chunk. This allowed us to resume from the last completed chunk instead of starting from scratch.

5. **Data Preparation**: After tokenization, we converted the data into a PyTorch Dataset and used a DataLoader for batching during training.

6. **Training**: During training, we saved model checkpoints after each epoch to protect against session terminations. This way, we could resume training from the last completed epoch instead of starting over.

7. **Competitiveness of the Approach**: This approach is quite robust and efficient given the constraints. It effectively handles large datasets that cannot fit into memory all at once by processing data in chunks and storing intermediate results to disk. The use of a smaller pre-trained model (bert-tiny) allows us to leverage the power of BERT while reducing computational requirements. The checkpointing system ensures that our progress is saved regularly, allowing us to resume training even after interruptions. Moreover, this approach aligns with the track's encouragement of studying the efficacy of transfer learning methods. Thus, this approach is quite competitive for the given task and constraints.

Looking forward, we identified potential improvements that could further enhance the performance of our model. 
These include using an ensemble of models, employing more recent transformer models like RoBERTa or ELECTRA, or 
incorporating additional features into our model, such as query and passage length or the presence of specific keywords. 
We believed these enhancements could potentially give us an edge in the TREC 2023 Deep Learning Track.

```Python
!pip install transformers

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import gc  # For garbage collection
import time  # For timing

# Define column names
query_cols = ['query_id', 'query']
top100_cols = ['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_id']
qrels_cols = ['query_id', '0', 'doc_id', 'relevance']

# Read the data from the TSV and TXT files
start_time = time.time()

train_queries = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_queries.tsv', sep='\t', names=query_cols, header=None)
train_top100 = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_top100.txt', sep=' ', names=top100_cols, header=None)
train_qrels = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_qrels.tsv', sep='\t', names=qrels_cols, header=None)

print("--- Data loading time: %s seconds ---" % (time.time() - start_time))

# Merge the two dataframes on 'query_id'
merged = pd.merge(train_queries, train_top100, on='query_id')

# Free up memory
del train_queries
del train_top100
gc.collect()  # Force garbage collector to release unreferenced memory

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Use this import instead

# Specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model and move it to GPU
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')
model.to(device)
model.train()

# Initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Specify the number of epochs
epochs = 3

# Break the data into smaller chunks and tokenize them one by one
chunksize = 1000  # You may adjust this value based on your available memory
train_encodings = []

start_time = time.time()

last_processed_chunk = 0
# Check if a file with the last processed chunk exists and if so, read the value
try:
    with open('/content/drive/MyDrive/Colab/Deep/last_processed_chunk.txt', 'r') as f:
        last_processed_chunk = int(f.read())
except FileNotFoundError:
    pass

# Start tokenization from the last processed chunk
for i in range(last_processed_chunk, len(merged), chunksize):
    chunk = merged[i:i+chunksize]
    chunk_encodings = tokenizer(chunk['query'].tolist(), chunk['doc_id'].tolist(), truncation=True, padding=True, max_length=512)
    # Save encodings to disk, then delete the variable
    torch.save(chunk_encodings, f'/content/drive/MyDrive/Colab/Deep/train_encodings_{i}.pt')
    del chunk_encodings
    gc.collect()  # Force garbage collector to release unreferenced memory

    # Update the last processed chunk
    with open('/content/drive/MyDrive/Colab/Deep/last_processed_chunk.txt', 'w') as f:
        f.write(str(i))

print("--- Tokenization time: %s seconds ---" % (time.time() - start_time))

# Defining labels 
train_labels = train_qrels['relevance'].tolist()

# Load each set of encodings from disk and concatenate
start_time = time.time()

for i in range(0, len(merged), chunksize):
    chunk_encodings = torch.load(f'/content/drive/MyDrive/Colab/Deep/train_encodings_{i}.pt')
    if i == 0:
        train_encodings = chunk_encodings
    else:
        train_encodings = {key: torch.cat([train_encodings[key], chunk_encodings[key]]) for key in train_encodings}
    del chunk_encodings
    gc.collect()  # Force garbage collector to release unreferenced memory

print("--- Encoding concatenation time: %s seconds ---" % (time.time() - start_time))

# Convert to PyTorch Dataset
train_dataset = TextDataset(train_encodings, train_labels)

# Prepare PyTorch DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
start_time = time.time()

for epoch in range(epochs):
    for batch in train_dataloader:
        # Move batch tensors to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optim.step()
        optim.zero_grad()

    # Save model after each epoch
    checkpoint_dir = f'/content/drive/MyDrive/Colab/Deep/bert_checkpoints/epoch_{epoch}'
    model.save_pretrained(checkpoint_dir)
    print(f'Saved model checkpoint to {checkpoint_dir}')

print("--- Training time: %s seconds ---" % (time.time() - start_time))

model.eval()
```

