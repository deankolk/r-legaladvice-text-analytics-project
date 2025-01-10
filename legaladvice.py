import praw
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# API CREDENTIALS
client_id = 'enter your client id here'
client_secret = 'enter your client secret here'
user_agent = 'legaladvice_clustering by dkolk'
# INTIALIZE REDDIT INSTANCE
reddit = praw.Reddit(client_secret=client_secret,
                     client_id=client_id,
                     user_agent=user_agent)
# SPECIFY SUBREDDIT
subreddit_name = 'legaladvice'
subreddit = reddit.subreddit(subreddit_name)
data_list = []
# LOOP THROUGH POSTS AND ADD TO LIST
for post in subreddit.hot(limit=1000):
    temp_list = [
        post.title,
        post.score,
        post.url,
        post.created,
        post.num_comments,
        post.selftext,
        post.upvote_ratio
    ]
    data_list.append(temp_list)
df = pd.DataFrame(data_list, columns=['Title', 'Score', 'URL', 'Created',
                                      'Num Comments', 'Text', 'Upvote Ratio'])
# COMBINE TITLE AND TEXT
df['Title'] = df['Title'].str.lower()
df['Text'] = df['Text'].str.lower()
df['combined_text'] = df['Title'] + " " + df['Text']
df['combined_text'] = df['combined_text'].str.lower()

# after creating clusters, as insignificant words are included in clusters
# continuously returning and adding those words to my stop words
additional_stop_words = ['would', 'get', 'year', 'name', 'court', 'know',
                         'like', 'said', 'back', 'time', 'day', 'support',
                         'want', 'told', 'dont', 'one', 'got', 'also', 'email',
                         'legal', 'case', 'lawyer', 'ticket', 'account', 'door',
                         'ive', 'need', 'take', 'company', 'water', 'going',
                         'state', 'since','someone','could','record',
                         'month','hour','week','didnt','pay','anything',
                         'notice','even','say','police','order','card',
                         'phone','help','fault','question',
                         'parking','address','went','still',
                         'never','advice','damage','fence','tree','camera','property',
                         'vehicle','parked','deposit','agreement','attorney','owner',
                         'see','right','free',
                         'judge','due','find','yard','side','front',
                         'cover','claim','road','received','letter',
                         'new','asked',
                         'use','service','small','move','law','illegal',
                         'closed','he','really','thing','called',
                         'lane','say','signed','issue',
                         'make','cant','call','bos',
                         'well','moving','leave','im','months',
                         'us','go','information','ago','years','district','shop',
                         'legally','lien','action','public','sale',
                         'give','two','something','days','first','moved','hearing',
                         'live','sent','shes','certificate','date','trying','long',
                         'hit','put','people','able','old','member','filed','amount',
                         'things','way','sure','made','hours','getting','recently','report',
                         'found','living','doctors','arrested','street',
                         'done','tried','let','saying','number','sue','guy','charges',
                         'charged','jail','evidence','claims','option','sell','thanks',
                         'next','nothing','gave','around','california','much','dog','working',
                         'care','place','sign','honey','keep','test','another','officer','text',
                         'charge','cop','pto']
# remove spaces just to be safe
additional_stop_words = [word.strip() for word in additional_stop_words]
stop_words = set(stopwords.words('english')).union(additional_stop_words)
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)       # tokenize
    tokens = [
        lemmatizer.lemmatize(word)  # lemmatize each token
        for word in tokens
        if word.lower() not in stop_words and len(word) >= 3  # Remove stop words and short words
    ]
    return ' '.join(tokens)  # Join tokens back into a string

# Apply preprocessing
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])


# k means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters,n_init=25, random_state=23)
kmeans.fit(tfidf_matrix)

# Step 3: Analyze the results
# Cluster assignments
df['cluster'] = kmeans.labels_  # Assign cluster labels to each document
# Centroids of the clusters
centroids = kmeans.cluster_centers_
terms = vectorizer.get_feature_names_out()
# Function to display top terms for each cluster
def get_top_terms(centroids, terms, n=5):
    top_terms = {}
    for i, centroid in enumerate(centroids):
        top_indices = centroid.argsort()[-n:][::-1]  # Get indices of top n terms
        top_terms[i] = [terms[j] for j in top_indices]
    return top_terms
# Display top terms for each cluster
top_terms = get_top_terms(centroids, terms)
print("Top terms per cluster:", top_terms)
# Get the top terms per cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
# Track words already assigned to a cluster

cluster_scores = {}

for i in range(num_clusters):
    cluster_terms = []
    cluster_score = 0
    for ind in order_centroids[i, :5]:  # Top 5 terms
        term = terms[ind]
        score = kmeans.cluster_centers_[i, ind]  # Corresponding score
        cluster_terms.append(term)
        cluster_score += score  # Sum scores for the cluster

    cluster_scores[i] = {
        'terms': cluster_terms,
        'score': cluster_score
    }

# Sort clusters by score
sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1]['score'], reverse=True)

# summing TF IDF scores of the top n terms to create a cluster importance score to rank the topics
# Print the top 5 clusters
print("Top Clusters Ranked:")
for cluster_id, info in sorted_clusters[:5]:  # Top 5
    print(f"Cluster {cluster_id}: Score = {info['score']:.4f}, Terms = {', '.join(info['terms'])}")
assigned_words = set()

# Assuming kmeans.labels_ contains your cluster labels
num_clusters = len(np.unique(kmeans.labels_))  # Get number of clusters

# Create a dictionary for cluster shapes
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'H']  # Add more shapes if you have more clusters
colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Generate colors for clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix)  # Use .toarray() if using a sparse matrix
# Plot the clusters with different markers
plt.figure(figsize=(10, 7))

for cluster in range(num_clusters):
    plt.scatter(pca_result[kmeans.labels_ == cluster, 0],
                pca_result[kmeans.labels_ == cluster, 1],
                label=f'Cluster {cluster}',
                marker=markers[cluster % len(markers)],
                color=colors[cluster])

plt.title("K-Means Clustering of Documents")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Clusters')
plt.show()
# Reduce dimensionality using PCA to visualize in 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering of Documents")
plt.show()

