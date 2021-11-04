import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import os
import glob
import multiprocessing as mp
import timeit
import gensim

from sklearn.metrics import pairwise_distances
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from tqdm import tqdm_notebook as tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML

pd.options.display.max_rows = 20
# %matplotlib inline

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)


InteractiveShell.ast_node_interactivity = "all"

print('Number of CPU cores:', mp.cpu_count())

DATA_PATH = './data/instacart-market-basket-analysis'
files_list = glob.glob(f'{DATA_PATH}/*.csv')

files_list

data_dict = {}

for file in files_list:
    print(f'\n\nReading: {file}')
    data = pd.read_csv(file)
    print(data.info(null_counts=True))
    data_dict[file.split('.')[1].split('/')[-1]] = data

print(f'Loaded data sets: {data_dict.keys()}')

train_orders = data_dict['order_products__train']
prior_orders = data_dict['order_products__prior']
products = data_dict['products'].set_index('product_id')

orders = data_dict['orders']
prior_orders = prior_orders.merge(right=orders[['user_id', 'order_id', 'order_number']], on='order_id', how='left')

prior_orders.head()

# Sample users to keep the problem computationaly tractable
USER_SUBSET = 50000
user_ids_sample = prior_orders['user_id'].sample(n=USER_SUBSET, replace=False)

prior_orders_details = prior_orders[prior_orders.user_id.isin(user_ids_sample)].copy()
prior_orders_details['product_id'] = prior_orders_details['product_id'].astype(int)

prior_orders_details = prior_orders_details.merge(data_dict['products'], on='product_id', how='left')
prior_orders_details = prior_orders_details.merge(data_dict['aisles'], on='aisle_id', how='left')
prior_orders_details = prior_orders_details.merge(data_dict['departments'], on='department_id', how='left')

prior_orders_details.head()

# Create basic user features: relative purchase frequences in each depertment/aisle

feature_department = pd.pivot_table(prior_orders_details, index=['user_id'], values=['product_id'],
                                    columns=['department'], aggfunc='count', fill_value=0)
feature_department = feature_department.div(feature_department.sum(axis=1), axis=0)
feature_department.columns = feature_department.columns.droplevel(0)
feature_department = feature_department.reset_index()

feature_aisle = pd.pivot_table(prior_orders_details, index=['user_id'], values=['product_id'], columns=['aisle'],
                               aggfunc='count', fill_value=0)
feature_aisle = feature_aisle.div(feature_aisle.sum(axis=1), axis=0)
feature_aisle.columns = feature_aisle.columns.droplevel(0)
feature_aisle = feature_aisle.reset_index()

feature_df = feature_department.merge(feature_aisle, how='left', on='user_id').set_index('user_id')

feature_df.iloc[:5, :10]  # show first 10 columns (departments) only

len(feature_df.columns)

mm_scale = MinMaxScaler()
feature_df_scale = pd.DataFrame(mm_scale.fit_transform(feature_df),
                                columns=feature_df.columns,
                                index=feature_df.index.values)

tsne_doc_features = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
tsne_features_doc = tsne_doc_features.fit_transform(feature_df_scale.values)

tsne_doc_features = pd.DataFrame({'user_id': feature_df.index.values})
tsne_doc_features['tsne-2d-one'] = tsne_features_doc[:, 0]
tsne_doc_features['tsne-2d-two'] = tsne_features_doc[:, 1]

plt.figure(figsize=(16, 16))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    data=tsne_doc_features,
    legend="full",
    alpha=0.3
)
plt.savefig('./scatterplot_1.png', dpi=200)
plt.close()

# Computing silhouette scores for different clustering option
silhouette_list = []
for k in tqdm(range(2, 12, 2)):
    clusters = KMeans(n_clusters=k).fit(feature_df_scale).labels_.astype(float)
    silhouette_avg = silhouette_score(feature_df_scale, clusters, metric="euclidean")
    silhouette_list.append(silhouette_avg)
    print(f'Silhouette score for {k} clusters is : {silhouette_avg:.4}')

plt.figure(figsize=(10, 6))
plt.plot(range(2, 12, 2), silhouette_list)
plt.savefig('./silhouette_list.png', dpi=200)
plt.close()

train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

# It is important to sort order and products chronologically
prior_orders.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'], inplace=True)

combined_orders_by_user_id = prior_orders.groupby("user_id").apply(lambda order: ' '.join(order['product_id'].tolist()))

combined_orders_by_user_id = pd.DataFrame(combined_orders_by_user_id, columns=['all_orders'])
print(f'Number of orders: {combined_orders_by_user_id.shape[0]}')
combined_orders_by_user_id.reset_index(inplace=True)
combined_orders_by_user_id.user_id = combined_orders_by_user_id.user_id.astype(str)

combined_orders_by_user_id.head()

TRAIN_USER_MODEL = False  # True - create a new model, False - load a previosuly created model
MODEL_DIR = 'models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

embeddings_dim = 200  # dimensionality of user representation

filename = f'models/customer2vec.{embeddings_dim}d.model'
if TRAIN_USER_MODEL:

    class TaggedDocumentIterator(object):
        def __init__(self, df):
            self.df = df

        def __iter__(self):
            for row in self.df.itertuples():
                yield TaggedDocument(words=dict(row._asdict())['all_orders'].split(),
                                     tags=[dict(row._asdict())['user_id']])


    it = TaggedDocumentIterator(combined_orders_by_user_id)

    doc_model = gensim.models.Doc2Vec(vector_size=embeddings_dim,
                                      window=5,
                                      min_count=10,
                                      workers=mp.cpu_count(),
                                      alpha=0.055,
                                      min_alpha=0.055,
                                      epochs=120)  # use fixed learning rate

    train_corpus = list(it)

    doc_model.build_vocab(train_corpus)

    for epoch in tqdm(range(10)):
        doc_model.alpha -= 0.005  # decrease the learning rate
        doc_model.min_alpha = doc_model.alpha  # fix the learning rate, no decay
        doc_model.train(train_corpus, total_examples=doc_model.corpus_count,
                        epochs=doc_model.epochs)  # epochs=doc_model.iter)
        print('Iteration:', epoch)

    doc_model.save(filename)
    print(f'Model saved to [{filename}]')

else:
    doc_model = gensim.models.Doc2Vec.load('./' + filename)
    print(f'Model loaded from [{filename}]')

vocab_doc = list(doc_model.docvecs.doctags.keys())
# vocab_doc = list(doc_model.dv.doctags.keys())
doc_vector_dict = {arg: doc_model.docvecs[arg] for arg in vocab_doc}
X_doc = pd.DataFrame(doc_vector_dict).T.values

X_doc.shape, len(vocab_doc), prior_orders["user_id"].nunique()

user_ids_sample_str = set([str(id) for id in user_ids_sample])
idx = []
for i, user_id in enumerate(doc_vector_dict):
    if user_id in user_ids_sample_str:
        idx.append(i)
X_doc_subset = X_doc[idx]  # only sampled user IDs
X_doc_subset.shape

doc_vec_subset = pd.DataFrame(doc_vector_dict).T.iloc[idx]
doc_vec_subset.shape

distance_matrix_doc = pairwise_distances(X_doc_subset, X_doc_subset, metric='cosine', n_jobs=-1)
tsne_doc = TSNE(metric="precomputed", n_components=2, verbose=1, perplexity=30, n_iter=500)
tsne_results_doc = tsne_doc.fit_transform(distance_matrix_doc)

tsne_doc = pd.DataFrame()
tsne_doc['tsne-2d-one'] = tsne_results_doc[:, 0]
tsne_doc['tsne-2d-two'] = tsne_results_doc[:, 1]

plt.figure(figsize=(16, 16))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    data=tsne_doc,
    legend="full",
    alpha=0.3
)
plt.savefig('./scatterplot_2.png', dpi=200)
plt.close()

def cluster_cosine(X, k):
    # normalization is equivalent to cosine distance
    return KMeans(n_clusters=k).fit(preprocessing.normalize(X_doc_subset)).labels_.astype(float)


silhouette_list = []
for k in tqdm(range(2, 22, 1)):
    latent_clusters = cluster_cosine(X_doc_subset, k)
    silhouette_avg = silhouette_score(X_doc_subset, latent_clusters, metric="cosine")
    silhouette_list.append(silhouette_avg)
    print(f'Silhouette score for {k} clusters is : {silhouette_avg:.4}')

plt.figure(figsize=(10, 6))
plt.plot(range(2, 22, 1), silhouette_list);
plt.savefig('./silhouette_2.png', dpi=200)
plt.close()

N_CLUSTER = 12

latent_clusters = cluster_cosine(X_doc_subset, N_CLUSTER)
doc_vec_end = doc_vec_subset.copy()
doc_vec_end['label'] = latent_clusters
tsne_doc['cluster'] = latent_clusters

doc_vec_end['label'].value_counts()

plt.figure(figsize=(16, 16))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue='cluster',
    palette=sns.color_palette("hls", tsne_doc['cluster'].nunique()),
    data=tsne_doc,
    legend="full",
    alpha=0.3
);
plt.savefig('./scatterplot_3.png', dpi=200)

feature_df['latent_cluster'] = latent_clusters

department_names = np.setdiff1d(prior_orders_details['department'].unique(), ['other', 'missing'])
interpetation_department = feature_df.groupby('latent_cluster')[department_names].mean()

interpetation_department.T.div(interpetation_department.sum(axis=1)).round(3)

interpetation_aisle = feature_df.groupby('latent_cluster')[feature_df.columns.values[16:-1]].mean()
interpetation_aisle.T.div(interpetation_aisle.sum(axis=1)).round(3).head(20)

prior_orders_details_clustered = prior_orders_details.copy()
prior_orders_details_clustered = prior_orders_details_clustered.merge(feature_df['latent_cluster'], on='user_id',
                                                                      how='left')

for cluster_id in [7.0, 3.0]:
    prior_orders_details_clustered[prior_orders_details_clustered['latent_cluster'] == cluster_id][
        ['user_id', 'product_name']].groupby("user_id").apply(
        lambda order: ' > '.join(order['product_name'])).reset_index().head(5)
