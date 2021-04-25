"""
Implements a simple recommendation system based on LightFM

References:https://www.datarevenue.com/en-blog/building-a-production-ready-recommendation-system
"""

"""Recommendation module.
This module deals with using LightFM models in production and includes a
LightFm subclass which provides a predict_online method to use in API or
similar scenarios.
"""
import operator

import numpy as np
import pandas as pd
import sparsity as sp
from scipy import sparse
from lightfm import LightFM
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey



class LFMRecommender(LightFM):
    """Recommender class based on the LightFM Model.
    The LightFM model is more expressive if an identity matrix is appended to
    feature matrices. It acts like a memory for the model, since it
    creates an individual embedding (vector of no_comp) for each user previously
    seen (during training).
    If the user is unknown from training but user_features are available,
    these can be passed to the model/class, and the model will try to give the best recommendations
    based on the available data. There will be an embedding for each
    feature used during training.
    Furthermore, baseline recommendations are computed and returned
    if the user is unknown and no user features are available.
    Finally, this class contains lots of checks on data integrity and can
    recover from things like shuffled or additional features.
    Parameters
    ----------
        indicators: 'users', 'items', 'both' or False
            whether to add identity matrices to the respective
            features matrices. Adds a user/item memory to the model.
        kwargs:
            remaining arguments are passed to the LightFM model.
    """

    def __init__(self, indicators='both', **kwargs):
        """Initialise model.
        Parameters
        ----------
        indicators: 'users', 'items', 'both' or False
            whether to add identity matrices to the respective
            features matrices. Adds a user/item memory to the model.
        kwargs:
            remaining arguments are passed to the LightFM model.
        """
        super().__init__(**kwargs)
        self.uid_map = pd.Series([])
        self.iid_map = pd.Series([])
        if indicators in ['both', 'users', 'items', False]:
            self.indicator_setting = indicators
        elif indicators:
            self.indicator_setting = 'both'
        else:
            raise ValueError("Invalid identity_matrix parameters: {}"
                             .format(indicators))
        self.user_feature_names = pd.Index([])
        self.item_feature_names = pd.Index([])
        self.baseline = pd.Series([])
        self._user_indicator = None
        self._item_indicator = None
        self._item_cache = LRUCache(maxsize=8)

    def fit_partial(self, interactions: sp.SparseFrame,
                    user_features: sp.SparseFrame = None,
                    item_features: sp.SparseFrame = None,
                    sample_weight=None,
                    epochs=1,
                    num_threads=1,
                    verbose=False):
        try:
            self._check_initialized()
        except ValueError:
            self.prepare(interactions, item_features, user_features)

        interactions = interactions.data
        user_features = getattr(user_features, 'data', None)
        item_features = getattr(item_features, 'data', None)

        user_features, item_features = self.append_indicators(
            user_features, item_features
        )

        super().fit_partial(interactions, user_features, item_features,
                            sample_weight, epochs, num_threads, verbose)

    def prepare(self, interactions, item_features, user_features):
        """Prepare model for fit and prediction.
        This method initialises many model attributes, like
        item and user mappings, as well as features. This is
        usually done automatically.
        In some rare cases, it might be useful – like
        when using append_identity on a untrained model
        (used in train_with_early_stopping).
        Parameters
        ----------
        interactions: SparseFrame
            train interactions
        item_features: SparseFrame, None
            item metadata features
        user_features: SparseFrame, None
            user metadata features
        Returns
        -------
        None
        """
        self.uid_map = pd.Series(np.arange(interactions.shape[0]),
                                 index=interactions.index)

        # TODO fix part where interactions are created with MultiIndex in cols
        if isinstance(interactions.columns, pd.MultiIndex):
            interactions._columns = interactions.columns.levels[0]

        self.iid_map = pd.Series(np.arange(interactions.shape[1]),
                                 index=interactions.columns)
        if self.indicator_setting:
            self._init_indicators()
        if not self.indicator_setting and \
                (user_features is None or item_features is None):
            raise ValueError("Can't estimate embeddings without indicators. "
                             "Try setting identity_matrix='both' or pass user "
                             "and item features to estimate embeddings.")

        self.user_feature_names = getattr(user_features, 'columns', None)
        self.item_feature_names = getattr(item_features, 'columns', None)

        self.baseline = pd.Series(
            np.asarray(interactions.mean(axis=0)).flatten(),
            index=interactions.columns,
            name='score') \
            .sort_values(ascending=False)

    def append_indicators(self, user_features, item_features):
        """Append indicator like used during training.
        Helper function mainly to use with LightFM evaluation functions.
        Parameters
        ----------
        user_features: csr_matrix
            user features without identity/indicators
        item_features: csr_matrix
            item_features without identity/indicators
        Returns
        -------
            uf_with_indicator, if_with_inidcator: csr_matrix
        """
        if self.indicator_setting in ['users', 'both']:
            if user_features is not None:
                user_features = sparse.hstack([user_features,
                                               self._user_indicator[:-1, :]])
            else:
                user_features = self._user_indicator[:-1, :]
        if self.indicator_setting in ['items', 'both']:
            if item_features is not None:
                item_features = sparse.hstack([item_features,
                                               self._item_indicator])
            else:
                item_features = self._item_indicator
        return user_features, item_features

    def _init_indicators(self):
        """Initialize indicator matrices."""
        if self.indicator_setting in ['both', 'users']:
            D = len(self.uid_map)
            self._user_indicator = sparse.vstack([
                sparse.identity(D, format='csr'),
                sparse.csr_matrix((1, D))
            ])
        if self.indicator_setting in ['items', 'both']:
            self._item_indicator = sparse.identity(
                len(self.iid_map), format='csr')

    def append_user_identity_row(self, v, idx):
        """Append single identity row to vector.
        Parameters
        ----------
        v: csr_matrix
            row_vector
        idx:
            identity index will determine the position of the positive
            entry in the appended identity
        Returns
        -------
            appended: csr_matrix
        """
        return sparse.hstack([v, self._user_indicator[idx, :]])

    def _check_missing_features(self, item_feat, user_feat):
        """Check for any missing features."""
        if user_feat is not None:
            user_feat_diff = set(self.user_feature_names) - \
                                set(user_feat.columns)
            if len(user_feat_diff):
                raise ValueError('Missing user features: {}'
                                 .format(user_feat_diff))

        if item_feat is not None and self.user_feature_names is not None:
            item_feat_diff = set(self.item_feature_names) -\
                             set(item_feat.columns)

            if len(item_feat_diff):
                raise ValueError('Missing item features: {}'
                                 .format(item_feat_diff))

    @cachedmethod(cache=operator.attrgetter('_item_cache'),
                  key=lambda _, __, item_ids: hashkey(item_ids))
    def get_item_data(self, item_features, item_ids):
        """Return item data.
        This creates the item feature csr and corresponding item names and
        numerical ids. Caches result in case same items are requested again.
        """
        item_ids = np.asarray(list(item_ids))
        if item_features is not None:
            assert item_features.shape[0] >= len(item_ids)
            assert set(item_ids).issubset(set(item_features.index))
            iid_map = pd.Series(np.arange(len(item_features)),
                                index=item_features.index)
        else:
            iid_map = self.iid_map
        iid_map = iid_map.reindex(item_ids)
        return self._construct_item_features(item_features, item_ids), \
            iid_map.values,\
            iid_map.index

    def predict_online(self, user_id, item_ids, item_features=None,
                       user_features=None, num_threads=1, use_baseline=False):
        """Helper method to use during API use.
        This method reads all available data and gives the best possible
        recommendation for a received sample.
        It also executes various checks on data integrity.
        Parameters
        ----------
        user_id: scalar
            user ids as provided during training
        item_ids: array like
            item ids as provided during training
        item_features: SparseFrame
        user_features: SparseFrame
        num_threads: int
            Number of threads to use during prediction
        use_baseline: true
            in case user is not known and no user features are passed and
            use_baseline=True baseline predictions will be returned . If
            use_baseline=False a KeyError will be raised.
        Returns
        -------
            predictions: pd.Series
                a mapping from item id to score (unsorted)
        """
        self._check_missing_features(item_features, user_features)

        if item_ids is not None:
            if isinstance(item_ids, pd.Index):
                item_ids = item_ids.tolist()
            item_names = tuple(item_ids)
        else:
            item_names = tuple(self.iid_map.index.tolist())

        item_feat_csr, num_item_ids, item_labels = \
            self.get_item_data(item_features, item_names)
        try:
            user_feat_csr = self._construct_user_features(user_id,
                                                          user_features)
        except KeyError:
            if use_baseline:
                return self.baseline
            else:
                raise

        # for single case prediction we always use id 0 as lightFm uses it as
        # index into the user feature matrix if the user was known during
        # training we append an identity matrix to indicate that the user
        # was known.
        pred = super().predict(0, num_item_ids,
                               item_feat_csr, user_feat_csr,
                               num_threads)

        pred = pd.Series(pred, index=item_labels)
        return pred

    def _construct_item_features(self, item_features, item_ids):
        """Create item features during predict."""
        # align feature names
        if self.indicator_setting in ['both', 'items']:
            item_indicator = sp.SparseFrame(self._item_indicator,
                                            index=self.iid_map.index)
            item_indicator = item_indicator.reindex(item_ids).data
        else:
            item_indicator = None

        if self.item_feature_names is None:
            return item_indicator

        item_feat_csr = item_features\
            .loc[:, self.item_feature_names]\
            .reindex(item_ids, axis=0)\
            .data
        if item_indicator is not None:
            item_feat_csr = sparse.hstack([item_feat_csr,
                                           item_indicator])
        return item_feat_csr

    def __setstate__(self, state):
        """Support unpickling older versions of this class."""
        if 'identity_matrix' in state:
            state['indicator_setting'] = state['identity_matrix']
        self.__dict__ = state

    def _construct_user_features(self, user_id, user_features):
        """Create user features for a single user."""
        # retrieve numerical user ids
        # abort and return baseline recommendations if user is not known
        # and no user features are passed
        user_known = True
        try:
            num_user_id = self.uid_map.loc[user_id]
        except KeyError:
            # Case we have no features nor the user was known we abort.
            if user_features is None:
                raise
            user_known = False
            num_user_id = 0

        if user_features is not None:
            if self.user_feature_names is None:
                raise ValueError('Model was trained without user features. '
                                 'But received user features for prediction.')

            user_feat_csr = user_features.loc[:, self.user_feature_names].data

            if user_feat_csr.shape[0] > 1:
                raise ValueError(
                    'Received user feature matrix with more than 1 row.')
        else:
            user_feat_csr = None
            if self.user_feature_names is not None and \
                            self.indicator_setting in [False, 'users']:
                raise ValueError("Need user features as used "
                                 "during training: {}"
                                 .format(self.user_feature_names))

        if self.indicator_setting in ['users', 'both']:
            # if no user_features were used during training
            # no need to handle further cases just use indicator row.
            if self.user_feature_names is None:
                user_feat_csr = self._user_indicator[num_user_id]
            # Append identity matrix only if user is known from training,
            # features have been passed and the identity_matrix flag is set.
            elif user_feat_csr is not None and user_known:
                user_feat_csr = self.append_user_identity_row(user_feat_csr,
                                                              num_user_id)
            elif user_feat_csr is None and user_known:
                empty_features = sparse.csr_matrix(
                    (1, len(self.user_feature_names)))
                user_feat_csr = self.append_user_identity_row(empty_features,
                                                              num_user_id)
            elif user_features is not None and not user_known:
                user_feat_csr = self.append_user_identity_row(
                    user_feat_csr, -1)
        return user_feat_csr