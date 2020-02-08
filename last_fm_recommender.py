import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# doc data tu file
user_data = pd.read_table('data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])

user_profiles = pd.read_table('data/lastfm-dataset-360K/usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])

if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['artist-name'])

artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )

user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')

popularity_threshold = 40000
user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')

combined = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
usa_data = combined.query('country == \'United States\'')

if not usa_data[usa_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = usa_data.shape[0]
    usa_data = usa_data.drop_duplicates(['users', 'artist-name'])

wide_artist_data = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_artist_data_sparse)

# save the model for future
from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr('data/lastfm_sparse_artist_matrix.npz', wide_artist_data_sparse)

# gợi ý  ca sĩ theo ngẫu nhiên thi query_index gia tri random ca si trong wide_artist_data
# query_index = np.random.choice(wide_artist_data.shape[0])
# distances, indices = model_knn.kneighbors(wide_artist_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         print('Recommendations for {0}:\n'.format(wide_artist_data.index[query_index]))
#     else:
#         print('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data.index[indices.flatten()[i]], distances.flatten()[i]))

wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
wide_artist_data_zero_one_sparse = csr_matrix(wide_artist_data_zero_one.values)

# save_sparse_csr('data/lastfm_sparse_artist_matrix_binary.npz', wide_artist_data_zero_one_sparse)

model_nn_binary = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn_binary.fit(wide_artist_data_zero_one_sparse)

# distances, indices = model_nn_binary.kneighbors(wide_artist_data_zero_one.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         print('Recommendations with binary play data for {0}:\n'.format(wide_artist_data_zero_one.index[query_index]))
#     else:
#         print('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data_zero_one.index[indices.flatten()[i]], distances.flatten()[i]))
# gợi ý ca sĩ tương tự theo ca sĩ cũ thể

from fuzzywuzzy import fuzz


def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    """
    Inputs:
    query_artist: query artist name
    artist_plays_matrix: artist play count dataframe (not the sparse one, the pandas dataframe)
    knn_model: our previously fitted sklearn knn model
    k: the number of nearest neighbors.

    Prints: Artist recommendations for the query artist
    Returns: None
    """
    query_index = None
    ratio_tuples = []

    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    print('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples]))

    try:
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]  # get the index of the best artist match in the data
    except:
        print('Your artist didn\'t match any artists in the data. Try again')
        return None

    distances, indices = knn_model.kneighbors(artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]],
                                                     distances.flatten()[i]))

    return None

print_artist_recommendations('arctic monkeys', wide_artist_data_zero_one, model_nn_binary, k = 10)
