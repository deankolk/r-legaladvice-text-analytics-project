# r-legaladvice-text-analytics-project
web scraping using the reddit api and clustering 

GOAL:
To determine what are some common trends in law based on the r/legaladvice subreddit. The hot and trending posts are particular to each individual, but what are some overall
common topics or issues.

Used the reddit api to pull the title of the post, and the body. Did not consider comments. Looked at the top 1000 hot posts. Everytime the code is re run, it would pull the top 1000 hot posts, there fore
observing most recent and current topics, instead of all time highest posts. 

Process:

standardized data to remove punctuation and make all text lowercase etc.

applied a tfidf matrix to get word importance and transform data to numeric 

used a k-means clustering algorithm, experimented with using 3-7 clusters

used inital set of stopwards included in coding package (the, I, what, and, get etc), and removed additional stop words by hand after creating clusters (manager, management etc. 
redundant variations of words that have the same topic) 

determined the number of clusters by what makes most sense in context of the scope of the problem

used PCA to reduce dimensionality for the purpose of visualizing clusters 

ranked the final clusters using the average TFIDF score 


