"""
A simple working example of LightFM recommendation system using MovieLens Dataset

Ref: LightFM Website
"""

import numpy as np

from lightfm.datasets import fetch_movielens, fetch_stackexchange
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score


def sample_recommendation(model, data, user_ids):

    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


def collab_filtering():
    """
    implements collaborative filtering version
    by using only the rating data from movielens dataset
    :return:
    """
    data = fetch_movielens()

    for key, value in data.items():
        print(key, type(value), value.shape)

    train = data['train']
    test = data['test']
    print('The dataset has %s users and %s items, '
          'with %s interactions in the test and %s interactions in the training set.'
          % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

    model = LightFM(learning_rate=0.05, loss='bpr')
    model.fit(train, epochs=50, num_threads=5)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()
    train_recall = recall_at_k(model, test, k=10).mean()
    test_recall = recall_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    model = LightFM(learning_rate=0.05, loss='warp')

    #resume training from the model's previous state
    model.fit_partial(train, epochs=50, num_threads=5)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()
    train_recall = recall_at_k(model, test, k=10).mean()
    test_recall = recall_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print("*****************")
    print("After re-training")
    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    #check sample recommendation
    sample_recommendation(model, data, [3, 25, 450])

    #return model


def get_similar_tags(model, tag_id):
    # Define similarity as the cosine of the angle
    # between the tag latent vectors

    # Normalize the vectors to unit length
    tag_embeddings = (model.item_embeddings.T
                      / np.linalg.norm(model.item_embeddings, axis=1)).T

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar



def hybrid_model():
    """
    implements hybrid model using
    interaction data, as well as
    item features
    :return:
    """
    # Set the number of threads; you can increase this
    # if you have more physical cores available.
    NUM_THREADS = 2
    NUM_COMPONENTS = 30
    NUM_EPOCHS = 3
    ITEM_ALPHA = 1e-6
    data = fetch_stackexchange('crossvalidated',
                               test_set_fraction=0.1,
                               indicator_features=False,
                               tag_features=True)


    train = data['train'].tocsr().tocoo()
    test = data['test'].tocsr().tocoo()
    print('The dataset has %s users and %s items, '
          'with %s interactions in the test and %s interactions in the training set.'
          % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

    # Define a new model instance
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    #fit the collaborative filtering model
    model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    # Compute and print the AUC score
    train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
    print('Collaborative filtering train AUC: %s' % train_auc)
    # We pass in the train interactions to exclude them from predictions.
    # This is to simulate a recommender system where we do not
    # re-recommend things the user has already interacted with in the train
    # set.

    #suppress the error of train/test overlap
    LightFM._check_test_train_intersections = lambda x, y, z: True

    test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)

    """
    The fact that we score them lower than other items (AUC < 0.5) is due to estimated per-item biases, 
    which can be confirmed by setting them to zero and re-evaluating the model.
    """
    model.item_biases *= 0.0

    test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
    print('Collaborative filtering test AUC after using biases: %s' % test_auc)


    # Fit the hybrid model. Note that this time, we pass
    # in the item features matrix.
    item_features = data['item_features']
    tag_labels = data['item_feature_labels']

    print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)
    model = model.fit(train,
                      item_features=item_features,
                      epochs=NUM_EPOCHS,
                      num_threads=NUM_THREADS)

    # Don't forget the pass in the item features again!
    train_auc = auc_score(model,
                          train,
                          item_features=item_features,
                          num_threads=NUM_THREADS).mean()

    print('Hybrid training set AUC: %s' % train_auc)
    test_auc = auc_score(model,
                         test,
                         train_interactions=train,
                         item_features=item_features,
                         num_threads=NUM_THREADS).mean()
    print('Hybrid test set AUC: %s' % test_auc)

    #find similar tags
    for tag in (u'bayesian', u'regression', u'survival'):
        tag_id = tag_labels.tolist().index(tag)
        print('Most similar tags for %s: %s' % (tag_labels[tag_id],
                                                tag_labels[get_similar_tags(model, tag_id)]))


#collab_filtering()

hybrid_model()

