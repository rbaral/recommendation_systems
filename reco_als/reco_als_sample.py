"""
A simple working example of ALS recommender using MovieLens dataset.

Ref: Spark Website
"""

#basic imports
import os

#spark related imports
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

#constants
#path to the ratings file
FILE_RATINGS = "data/movielens_ratings.csv"


def get_spark():
    """
    init spark config
    :return:
    """
    spark = SparkSession \
        .builder \
        .appName("ALS sample movie recommender") \
        .getOrCreate()
    return spark


def get_data():
    """
    read the data and prepare training and test set
    :return:
    """
    ratings = spark.read.csv(FILE_RATINGS, inferSchema=True, header=True)
    training, test = ratings.randomSplit([0.8, 0.2])
    return ratings, training, test


def prepare_model(training, test):
    """
    Build the recommendation model using ALS on the training data
    :param training: the training data set
    :param test: the test data set
    :return: fitted ALS model
    """
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    return als, model


def get_user_reco(als, model):
    """
    recommend using the trained model
    :param als:
    :param model:
    :return:
    """
    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    movies = ratings.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)

    #display the results
    print("User recommendations")
    userRecs.show()
    print("Movie recommendations")
    movieRecs.show()

    print("User subset recommendations")
    userSubsetRecs.show()
    print("Movie subset recommendations")
    movieSubSetRecs.show()


spark = get_spark()
ratings, training, test = get_data()
als, model = prepare_model(training, test)
get_user_reco(als, model)






