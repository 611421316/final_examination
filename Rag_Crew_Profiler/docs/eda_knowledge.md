# EDA Knowledge for RAG

## Key Insights

- Token/word length distribution significantly affects retrieval performance.
- Very short reviews often lack sufficient context, making rating prediction less reliable.
- Very long reviews may introduce noise and reduce retrieval efficiency.
- Rating distribution reveals dataset bias (e.g., dominance of 4–5 star reviews).
- Review text patterns strongly correlate with rating behavior:
  - 1–2 stars: complaints, poor service, long waiting time, bad food, high price.
  - 3 stars: mixed or neutral experience.
  - 4–5 stars: positive expressions such as "great", "friendly", "delicious".

## Practical Implications

- Apply descriptive statistics (mean, median, min, max, standard deviation) to understand data distribution.
- Detect and handle outliers (e.g., extremely long reviews).
- Normalize or chunk long documents to improve retrieval quality.
- Consider rating distribution to avoid prediction bias.
- Retrieve relevant user history and item history to provide better context.

## Impact on RAG System

- Improves retrieval relevance by controlling document length and noise.
- Provides structured knowledge to guide agent reasoning.
- Helps the Crew infer rating from text patterns instead of random guessing.
- Enhances consistency and stability of predictions.
- Reduces prediction error (e.g., lower MAE) by aligning with real user behavior.