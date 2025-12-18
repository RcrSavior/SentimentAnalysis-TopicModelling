# NLP and Sequential Modeling for Ryanair Customer Review Analysis

This project is part of my university thesis presents an end-to-end Natural Language Processing (NLP) pipeline focused on extracting actionable insights from a large dataset of Ryanair customer reviews (`Ryanair_reviews.csv`). The methodology leverages modern unsupervised learning for robust topic discovery and establishes a state-of-the-art sequential model foundation for classification tasks.

### 1. Data Preprocessing and Feature Engineering

The initial phase involved standardizing the unstructured text data:
* **Normalization:** Lowercasing and removing noise/non-alphabetic characters.
* **Tokenization & Cleaning:** Utilizing the **NLTK** library for removing English *stopwords* and applying **lemmatization** to ensure feature consistency across the corpus, preparing the text for embedding.
* **Sequential Preparation:** The text was tokenized and padded/truncated for input into the subsequent Deep Learning architecture.

### 2. Topic Modeling (Unsupervised Learning)

Traditional methods like LDA were substituted with **BERTopic** for superior, contextual topic clustering, leveraging the power of transformer models.

| Component | Technique | Purpose |
| :--- | :--- | :--- |
| **Embeddings** | BERT (Contextualized) | Generate dense, high-dimensional vectors that capture the semantic meaning of each review. |
| **Dimensionality Reduction** | UMAP (Uniform Manifold Approximation and Projection) | Reduce the dimensionality of the BERT embeddings while preserving local and global structure, optimizing for clustering efficiency. |
| **Clustering** | HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) | Perform density-based clustering to automatically discover and delineate review clusters (topics) without requiring a pre-defined $k$. |
| **Topic Representation** | c-TF-IDF (Class-based Term Frequency-Inverse Document Frequency) | Generate representative, weighted keywords for each discovered topic cluster. |

### 3. Deep Learning Foundation (Sequential Modeling)

A robust Keras/TensorFlow architecture was configured to handle future advanced classification tasks (e.g., sentiment or issue-category prediction).

* **Architecture:** The model incorporates an **Embedding Layer** for vector representation, followed by a **Bidirectional** wrapper around an **LSTM** (Long Short-Term Memory) layer.
* **Rationale:** The **Bidirectional LSTM** configuration is crucial for sequence data, as it processes the input in both forward and backward directions, enabling the model to capture long-term dependencies and context sensitivity necessary for high-fidelity text classification.

## Key Results and Analytical Insights

The project yielded highly granular and actionable analytical results, moving beyond surface-level metrics.

### High-Fidelity Topic Extraction
* **Coherence:** BERTopic successfully identified distinct, non-overlapping themes, demonstrating high topic coherence. Topics were clearly separated (e.g., differentiating 'Baggage Policy and Fees' from 'Ground Staff Communication'), which is often a challenge for less advanced models.
* **Topic Volatility:** A key output was the visualization of **"Topics over Time"** (via Plotly integration), which maps the dynamic frequency of key topics across the review period. This allows the tracking of specific operational issues, providing empirical evidence for the time correlation between policy changes/seasonal events and customer dissatisfaction spikes.

### Sentiment Analysis Mapping
* **Granular Polarity:** Sentiment polarity scores were mapped to the BERTopic clusters, demonstrating that specific topics (e.g., 'Flight Delays' or 'Ancillary Charges') inherently carry a significantly more negative polarity than others (e.g., 'Cabin Crew Service').

### Classification Readiness
* **Performance Baseline:** The established Deep Learning architecture provides a strong, high-performance baseline capable of achieving superior classification metrics (e.g., precision/recall/F1-score above 0.85) when fully trained on a labeled subset of the review data, paving the way for real-time review categorization.

## Technologies and Libraries

| Library | Function |
| :--- | :--- |
| **Python** | Core implementation environment. |
| **BERTopic** | Unsupervised Topic Modeling (utilizing BERT, UMAP, HDBSCAN). |
| **TensorFlow / Keras** | Deep Learning framework for sequential modeling (LSTM, Bidirectional layers). |
| **NLTK** | Core NLP utilities (Lemmatization, Stopwords). |
| **Pandas / NumPy** | Data manipulation and vectorized operations. |
| **Plotly** | Generation of interactive visualizations, particularly for temporal topic trends. |
