# Twitter-Tweets-Clustering

- Twitter provides a service for posting short messages. In practice, many of the tweets are very similar to each other and can be clustered together. By clustering similar tweets together, we can generate a more concise and organized representation of the raw tweets, which will be very useful for many  Twitter-based applications (e.g., truth discovery, trend analysis, search ranking, etc.)

- Similarity between tweets is measured using the Jaccard Distance metric.
- Tweets are clustered together using the K-means clustering algorithm.

- The input to the program is a real world dataset sampled from Twitter during the Boston Marathon Bombing event in April 2013 that contains 251 tweets. The program also takes a list of initial centroids (as input) for creating different clusters. The input data is cleansed and K-means algorithm is applied to cluster identical tweets together.  


Introduction to Jaccard Distance:

The Jaccard distance, which measures dissimilarity between two sample sets (A and B). It is defined as the difference of the sizes of the union and the intersection of two sets divided by the size of the union of the sets.

For example, consider the following tweets:

Tweet A: the long march

Tweet B: ides of march

|A intersection B | = 1 and |A U B | = 5, therefore the distance is 1 – (1/5)