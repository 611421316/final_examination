# EDA Knowledge for RAG

## Key Insights
- Token/word length distribution is important for retrieval performance
- Very short documents may lack context → poor retrieval quality
- Very long documents may reduce efficiency and introduce noise

## Practical Implications
- Use descriptive statistics (mean, median, max) to understand dataset
- Detect outliers (e.g., extremely long documents > threshold)
- Normalize or chunk documents to improve retriever performance

## Impact on RAG
- Balanced document length improves retrieval accuracy
- Proper chunking leads to better context for generation
- EDA helps optimize indexing and retrieval pipeline