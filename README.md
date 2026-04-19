# Natural Language Processing - Assignment 2
**Student ID:** i22-1931  
**Course:** CS-4063

This repository contains my complete submission for Assignment 2 of the Natural Language Processing course. The objective of this assignment was to build a comprehensive neural NLP pipeline entirely from scratch in PyTorch, focusing on word embeddings, sequence labeling, and transformer-based topic classification using a BBC Urdu corpus.

As per the strict assignment requirements, no pre-trained models, Gensim, or HuggingFace libraries were used. Furthermore, built-in PyTorch transformer modules (`nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`) were completely avoided. All components, including the multi-head attention blocks and Skip-gram algorithms, are explicitly coded via native PyTorch tensor operations.

## Repository Contents

- `i22-1931_Assignment2_DS-X.ipynb`: The main, fully executed Jupyter Notebook containing all the code for the three phases of the NLP pipeline.
- `i22-1931_Assignment2_Report.pdf`: A detailed 3-page evaluation report discussing the implementation, results, ablation studies, and architectural decisions.
- `cleaned.txt` & `raw.txt`: The BBC Urdu text dataset utilized for training the pipeline.
- `Metadata.json`: The categorical metadata mapping used for the topic classifier.
- `/embeddings`: A subdirectory containing the generated TF-IDF matrices, PPMI matrices, Skip-gram W2V arrays (`.npy`), and exported t-SNE evaluation charts (`.png`).
- `/models`: A subdirectory containing the saved `.pt` weight files for the frozen and fine-tuned BiLSTM POS/NER sequence taggers, as well as the final Transformer model weights.

## How to Run

1. **Setup the Environment**  
   Ensure you have Python 3.10+ installed. Install the necessary prerequisites via pip:
   ```bash
   pip install torch numpy matplotlib scikit-learn scipy jupyterlab
   ```

2. **Launch the Notebook**  
   Open terminal in the project root directory and start the Jupyter environment:
   ```bash
   jupyter lab i22-1931_Assignment2_DS-X.ipynb
   ```
   
3. **Execution**  
   The notebook is grouped logically into the three distinct assignment parts. Execute the cells sequentially from top to bottom. It will automatically parse the local `.txt` and `.json` data files, construct the vocabulary indexes, train the models, and deploy the outputs dynamically into the respective `models/` and `embeddings/` folders.

## Implementation Overview

- **Part 1 (Word Embeddings):** Generates raw term-document arrays using TF-IDF and PPMI. It also implements a custom PyTorch Skip-gram neural network with negative sampling, evaluating the outputs via t-SNE dimensionality reduction and Cosine MRR testing.
- **Part 2 (Sequence Labeling):** Initially implements baseline rule-based parsers and gazetteers for Urdu tags, eventually transitioning to a 2-layer PyTorch BiLSTM network coupled with a Conditional Random Field (CRF) emission layer to accurately identify complex POS and NER tag sequences.
- **Part 3 (Transformer Classifier):** A fully functional Transformer Encoder classification head crafted explicitly using algebraic tensor matrix multiplications (Scaled Dot-Product Attention, custom Multi-Head Attention, Sinusoidal Positional Encoding, and Pre-LN blocks) to classify BBC articles accurately into 5 unique semantic themes.

Please refer to `i22-1931_Assignment2_Report.pdf` for extensive training statistics, detailed ablation comparisons, and test set accuracy assessments.
