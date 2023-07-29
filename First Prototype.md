### First Prototype of TREC 2023 Deep Learning Training Phase

The approach we took in developing the code for the Training Phase of TREC 2023 Deep Learning Track's passage ranking task was carefully 
considered to meet the specific requirements and objectives of the project. Here are the reasons behind the decisions we made:

1. **Task Compatibility**: Our primary goal was to create a system capable of ranking passages based on their relevance to a given query.
This is directly aligned with the objectives of both the full ranking and reranking subtasks for the passage ranking task of the 2023 Deep Learning track.
To achieve this, we designed our model to employ a transformer-based model, BERT, which is well-known for its excellent performance in
various natural language processing (NLP) tasks, including text classification and relevance ranking.

2. **Employing BERT**: We chose to use BERT because it has been pre-trained on a large corpus of text data and has a proven track record of
understanding the contextual meaning of words in a sentence. This made it a perfect fit for assessing the relevance of a passage to a
given query, which is the core requirement of our task. The line `model = BertForSequenceClassification.from_pretrained('bert-base-uncased')`
in our code is where we employed BERT. We used the `BertForSequenceClassification` model, a variant of BERT specifically designed for
classification tasks, and we loaded the pre-trained weights from `'bert-base-uncased'`.

3. **Leveraging Transfer Learning**: Our approach leverages the power of transfer learning by utilizing a pre-trained BERT model as a
starting point. This is a strategy encouraged in this TREC track description and it allowed us to save significant training time and
computational resources, while also taking advantage of the strong language understanding capabilities already built into BERT.

4. **Batch Training**: We realized that handling large amounts of data was a crucial aspect of this project. To effectively train our model on
potentially large training data without running into memory issues, we implemented batch training using PyTorch's DataLoader.
This ensured that our training process was both efficient and scalable. This is evident in the line
`train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)`.
Here, we specified a batch size of 16, meaning the model would be trained on 16 examples at a time.
This approach helped us manage memory usage efficiently and made the training process scalable.

5. **Data Preparation**: Understanding that our model needed the data in a specific format, we included steps in our code to process
the data into a usable format. This included tokenizing the queries and passages and encoding them into input vectors that our model could process.
The preparation of data involved several steps in our code. First, we read the data from the provided files using `pandas`' `read_csv` function.
Then, we tokenized and encoded the queries and passages using BERT's tokenizer (`tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`
and `train_encodings = tokenizer(train_data['query'].text, train_data['doc_id'].text, truncation=True, padding=True, max_length=512)`).

6. **Flexibility**: We designed our model to be adaptable to both the full ranking and reranking tasks. For the full ranking task,
we could feed the entire collection of Corpus passages into our model. For the reranking task, we tailored our model to accept only the
top 100 passages provided by top 100 training batch. This flexibility allowed us to cater to the specific requirements of each subtask.

Finally, the actual training of our model happens in the training loop: 

```
for epoch in range(epochs):
    for batch in train_dataloader:
        # Move batch tensors to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
```

In this loop, we iteratively update the weights of our model based on the training data. 
This process is repeated for a specified number of epochs to improve the model's performance. 

However, we recognized that the success of our approach wouldn't just depend on the model itself, but also on the quality of our training data 
and the computational resources available to us. We also knew that to achieve optimal results, we might need to fine-tune various hyperparameters 
such as the learning rate, batch size, and the number of training epochs.

Looking forward, we identified potential improvements that could further enhance the performance of our model. 
These include using an ensemble of models, employing more recent transformer models like RoBERTa or ELECTRA, or 
incorporating additional features into our model, such as query and passage length or the presence of specific keywords. 
We believed these enhancements could potentially give us an edge in the TREC 2023 Deep Learning Track.

```Python
# Step 1
!pip install transformers

# Step 2 - 1: Mount the Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Load the data
import pandas as pd

# Define column names
query_cols = ['query_id', 'query']
top100_cols = ['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_id']
qrels_cols = ['query_id', '0', 'doc_id', 'relevance']

# Read the data from the TSV and TXT files
train_queries = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_queries.tsv', sep='\t', names=query_cols, header=None)
train_top100 = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_top100.txt', sep=' ', names=top100_cols, header=None)
train_qrels = pd.read_csv('/content/drive/MyDrive/Colab/Deep/passv2_train_qrels.tsv', sep='\t', names=qrels_cols, header=None)

# Print the first few rows to check the data
print("Train queries data:\n", train_queries.head())
print("Train top100 data:\n", train_top100.head())
print("Train qrels data:\n", train_qrels.head())

# Step 4
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model and move it to GPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)
model.train()

# Initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Specify the number of epochs
epochs = 3

# Merge the two dataframes on 'query_id'
merged = pd.merge(train_queries, train_top100, on='query_id')

# Break the data into smaller chunks and tokenize them one by one
chunksize = 1000  # You may adjust this value based on your available memory
train_encodings = []

for i in range(0, len(merged), chunksize):
    chunk = merged[i:i+chunksize]
    chunk_encodings = tokenizer(chunk['query'].tolist(), chunk['doc_id'].tolist(), truncation=True, padding=True, max_length=512)
    train_encodings.append(chunk_encodings)
    del chunk  # Delete the chunk to save memory

# Concatenate all encodings
train_encodings = {key: torch.cat([enc[key] for enc in train_encodings]) for key in train_encodings[0]}

# Define your labels (this step may change based on your specific case)
train_labels = train_qrels['relevance'].tolist()

# Convert to PyTorch Dataset
train_dataset = TextDataset(train_encodings, train_labels)

# Prepare PyTorch DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch in train_dataloader:
        # Move batch tensors to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optim.step()
        optim.zero_grad()

model.eval()
```

