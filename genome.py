import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from enum import Enum
from io import StringIO
from typing import Dict

import requests
from dotenv import load_dotenv
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (col, sum as fsum, abs as fabs,
                                   when, dense_rank, stddev, mean, count, lit)
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.window import Window

from utils import retry_settings as retry

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            filename=os.path.join(BASE_PATH, "logs",
                                  f"{os.path.basename(__file__).split('.')[0]}.log"), mode="w"),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

load_dotenv()

USERS_RESOURCE = "users"
POSTS_RESOURCE = "posts"

spark = SparkSession.builder \
    .appName("DummyJson") \
    .getOrCreate()


class DummyJsonProps(Enum):
    USERS = (
        StructType([
            StructField("id", IntegerType(), nullable=False),
            StructField("firstName", StringType(), nullable=True),
            StructField("lastName", StringType(), nullable=True),
            StructField("address", StructType([
                StructField("city", StringType(), nullable=True),
                StructField("state", StringType(), nullable=True)
            ]), nullable=True)
        ]), "id,firstName,lastName,address")
    POSTS = (
        StructType([
            StructField("id", IntegerType(), nullable=False),
            StructField("userId", IntegerType(), nullable=False),
            StructField("views", IntegerType(), nullable=True),
            StructField("reactions", StructType([
                StructField("likes", IntegerType(), nullable=True),
                StructField("dislikes", IntegerType(), nullable=True)
            ]), nullable=True)
        ]), "id,userId,views,reactions")
    FEMALE_USERS = (
        StructType([
            StructField("id", IntegerType(), nullable=False)
        ]), "id")
    USERS_POSTS = (
        StructType([
            StructField("id", IntegerType(), nullable=True),
            StructField("title", StringType(), nullable=True),
            StructField("body", StringType(), nullable=True),
            StructField("userId", IntegerType(), nullable=True),
        ]), "id,title,body,userId")

    def __init__(self, schema, fields):
        self.schema = schema
        self.fields = fields


@retry
def create_user(first_name, last_name, username, password):
    """Adding a new user will not add it into the server.
        It will simulate a POST request and will return the new created user with a new id"""
    endpoint = "https://dummyjson.com/users/add"
    response = requests.post(endpoint,
                             {"first_name": first_name, "last_name": last_name, username: "lidiya",
                              "password": password})
    response.raise_for_status()
    if response.status_code == 201:
        logging.info(f"The user {username} with first_name {first_name} and last_name {last_name}"
                     f" is successfully created")

@retry
def authorize(username, password) -> [str, str]:
    """Returns access_token and refresh_token for user existing in DummyJson dataset"""
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url="https://dummyjson.com/auth/login", json=payload, headers=headers).json()
    return response.get("accessToken", ''), response.get("refreshToken", '')


@retry
def get_json_data(resource: str, offset: int, limit: int, props: DummyJsonProps) -> DataFrame:
    """Fetches the resource and returns dataframe"""
    endpoint = f"https://dummyjson.com/{resource}"
    data = requests.get(f"{endpoint}?limit={limit}&skip={offset}&select={props.fields}")
    data.raise_for_status()
    return spark.createDataFrame(data.json().get(resource), schema=props.schema)

def generate_http_filters(filters: Dict[str, str]) -> str:
    parts = []
    for k, v in filters.items():
        parts.append(f"key={k}")
        parts.append(f"value={v}")
    return "&".join(parts)

@retry
def get_filtered_data_using_token(resource: str, filters: Dict[str, str], props: DummyJsonProps, access_token: str) -> DataFrame:
    filter_string = generate_http_filters(filters)
    endpoint ="https://dummyjson.com/users/filter"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    data = requests.get(f"{endpoint}?{filter_string}&select={props.fields}", headers=headers)
    data.raise_for_status()
    return spark.createDataFrame(data.json().get(resource), schema=props.schema)

@retry
def select_users_post_by_id(user_id: int, props: DummyJsonProps) -> object:
    endpoint=f"https://dummyjson.com/users/{user_id}/posts"
    data = requests.get(endpoint)
    data.raise_for_status()
    return data.json().get(POSTS_RESOURCE)

@retry
def get_total_count(resource: str) -> int:
    """Fetch total count from the API metadata if available"""
    endpoint = f"https://dummyjson.com/{resource}"
    response = requests.get(endpoint)
    response.raise_for_status()
    if response:
        return response.json().get("total")

def fetch_all_pages(resource: str, props: DummyJsonProps, page_size: int = 30,
                    max_workers: int = 5) -> DataFrame:
    """Fetch all pages dataset and returns dataframe with predefined schema"""
    total = get_total_count(resource)
    offsets = range(0, total, page_size)

    dataframes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_json_data, resource, offset, page_size, props)
            for offset in offsets
        ]
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                dataframes.append(df)

    if not dataframes:
        return spark.createDataFrame([], schema=props.schema)

    combined_df = dataframes[0]
    for df in dataframes[1:]:
        combined_df = combined_df.union(df)
    return combined_df

def fetch_all_posts(props: DummyJsonProps, max_workers: int = 5) -> DataFrame:
    """Fetches all posts of female users"""
    token = authorize()[0]
    female_users = get_filtered_data_using_token(USERS_RESOURCE, {"gender": "female"}, DummyJsonProps.FEMALE_USERS, token)

    all_posts_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(select_users_post_by_id, row.id, props)
            for row in female_users.toLocalIterator()
        ]
        for future in as_completed(futures):
            posts = future.result()
            if posts:
                all_posts_data.extend(posts)

    if not all_posts_data:
        return spark.createDataFrame([], schema=props.schema)

    return spark.createDataFrame(all_posts_data, schema=props.schema)


def iqr_outlier(df: DataFrame, column: str, factor: float = 1.5) -> DataFrame:
    """Detects outliers using IQR for multiple variables in a PySpark DataFrame."""
    logging.info(f"Starting calculating IQR for {column}")
    # Calculate Q1, Q3, and IQR
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    return df \
        .withColumns({"upper_bound": lit(q3 + factor * iqr), "lower_bound": lit(q1 - factor * iqr)}) \
        .where((col(column) <= col("lower_bound")) | (col(column) >= col("upper_bound"))) \
        .sort("userId")

def z_score_outlier(df: DataFrame, column: str, threshold: float=3.0) -> DataFrame:
    """Detects outliers from a PySpark DataFrame using the z-score method."""
    logging.info(f"Starting calculating z-scores for {column}")
    # Calculate mean, standard deviation and z-scores
    stats = df.select(mean(col(column)).alias('mean'), stddev(col(column)).alias('stddev')).collect()[0]
    mean_val = stats["mean"]
    stddev_val = stats["stddev"]

    return df \
        .withColumns({
        "mean": lit(mean_val),
        "stddev": lit(stddev_val)}) \
        .withColumn(f'{column}_z_score',
                    (col(column) - lit(mean_val)) / lit(stddev_val)) \
        .withColumn('abs_z_score', fabs(col(f'{column}_z_score'))) \
        .where(col('abs_z_score') >= threshold) \
        .sort("userId")

def write_df_to_csv(df: DataFrame, name: str) -> None:
    """writes df from memory to csv"""
    df.coalesce(1).write \
        .option("header", "true") \
        .option("sep", "|") \
        .mode("overwrite") \
        .csv(os.path.join(BASE_PATH, "datasets", name))

def log_dataframe(message: str, df: DataFrame, num_rows: int = 5) -> None:
    output_buffer = StringIO()

    with redirect_stdout(output_buffer):
        df.show(num_rows)

    logging.info(f"{message}\n{output_buffer.getvalue()}\n")

if __name__ == '__main__':
    # You should create a user
    create_user("lidiya", "hanicheva", "lidiya", "123456")
    # Log in and obtain both an access token and refresh token
    access_token, refresh_token = authorize("chloem", "chloempass")
    # Fetch all users and all posts (using pagination and retry logic)
    df_users = fetch_all_pages(USERS_RESOURCE, DummyJsonProps.USERS) \
        .select(
        col("id").alias("userId"),
        col("firstName"),
        col("lastName"),
        col("address.city").alias("city"),
        col("address.state").alias("state")
    )
    log_dataframe("Users dataset is saved do dataframe", df_users)
    write_df_to_csv(df_users, "users")

    df_posts = fetch_all_pages(POSTS_RESOURCE, DummyJsonProps.POSTS) \
        .select(
        col("id").alias("postId"),
        col("userId"),
        col("views"),
        col("reactions.likes").alias("likes"),
        col("reactions.dislikes").alias("dislikes")
    )
    log_dataframe("Posts dataset is saved do dataframe:", df_posts)
    write_df_to_csv(df_posts, "posts")
    # Using PySpark merge posts with users
    df = df_users \
        .join(df_posts.alias("p"), on="userId", how="left") \
        .drop("p.userId")
    log_dataframe("Users and posts datasets are joined:", df)
    write_df_to_csv(df, "users_posts_merged")

    #  Find the top 10 posts with the highest like counts
    top_10_posts = df_posts \
        .sort(col("likes").desc()) \
        .limit(10)
    log_dataframe("Top 10 posts with the highest like counts:", top_10_posts, 10)
    write_df_to_csv(top_10_posts, "top_10_posts")

    # Find the top 10 users with the highest total number of likes.
    top_liked_users = df.groupBy("userId") \
        .agg(fsum("likes").alias("total_likes")) \
        .sort(col("total_likes").desc()) \
        .limit(10)
    log_dataframe("Top 10 users with the highest total number of likes:", top_liked_users, 10)
    write_df_to_csv(top_liked_users, "top_liked_users")

    # Calculate how many posts each user has created
    created_posts = df.groupBy("userId") \
        .agg(count("postId").alias("total_posts")) \
        .sort("userId")
    log_dataframe("How many posts each user has created", created_posts)
    write_df_to_csv(created_posts, "created_posts")

    # Calculate the average number of likes per post for each user
    average_likes_per_post_per_user = df.groupBy("userId") \
        .agg(fsum("likes").alias("total_likes"), fsum("postId").alias("total_posts")) \
        .withColumn("average_likes_per_user",
                    when(col("total_posts") != 0,
                         col("total_likes") / col("total_posts"))
                    .otherwise(None)
                    )
    log_dataframe("The average number of likes per post for each user:", average_likes_per_post_per_user)
    write_df_to_csv(average_likes_per_post_per_user, "average_posts_per_post_per_user")

    # Calculate how many users haven't written any posts
    inactive_users = df.groupBy("userId") \
        .agg(count("postId").alias("total_posts_per_user")) \
        .where(col("total_posts_per_user") == 0) \
        .select("userId") \
        .agg(count("userId").alias("total_inactive_users"))
    log_dataframe("How many users haven't written any posts", inactive_users)
    write_df_to_csv(inactive_users, "inactive_users")

    # Calculate what are the top 3 cities where users live that received
    # the highest total number of likes from their posts?
    top_3_cities = df.groupBy(["userId", "city"]) \
        .agg(fsum("likes").alias("total_likes")) \
        .withColumn("city_rank",
                    dense_rank()
                    .over(Window.partitionBy(col("city")).orderBy(col("total_likes").desc()))) \
        .where(col("city_rank") == 1) \
        .sort(col("total_likes").desc()) \
        .limit(3)
    log_dataframe("What are the top 3 cities where users live "
                  "that received the highest total number of likes from their posts?",
                  top_3_cities, 3)
    write_df_to_csv(top_3_cities, "top_3_cities")

    # Find anomalies in likes per user
    logging.info("Anomalies in likes per user")

    # 1. Number of likes should be lower than the number of views
    more_likes_than_views = df \
        .where(col("likes") > col("views")) \
        .select(["userId", "postId", "likes", "views"]) \
        .sort("userId")
    log_dataframe("Anomalies: Posts where number of likes is higher than "
                  "the number of views", more_likes_than_views)
    write_df_to_csv(more_likes_than_views, "more_likes_than_views")

    # 2. Number of reactions (likes + dislikes) should be lower than the number of views
    more_likes_than_reactions = df_posts \
        .withColumn("total_reactions", col("likes") + col("dislikes")) \
        .where(col("total_reactions") > col("views")) \
        .select(["userId", "postId", "likes", "dislikes", "views"]) \
        .sort("userId")
    log_dataframe("Anomalies: Posts where number of reactions (likes + dislikes) "
                  "is higher than the number of views", more_likes_than_reactions)
    write_df_to_csv(more_likes_than_reactions, "more_likes_than_reactions")

    # 3. Users that is liked more than others - likes per users outliers
    likes_per_user = df.groupBy("userId").agg(fsum("likes").alias("total_likes"))

    likes_per_user_iqr = iqr_outlier(likes_per_user, "total_likes")
    log_dataframe("Anomalies: Outliers in likes per user. IQR", likes_per_user_iqr)
    write_df_to_csv(likes_per_user_iqr,"likes_per_user_iqr")

    likes_per_user_z_score = z_score_outlier(likes_per_user, "total_likes")
    log_dataframe("Anomalies: Outliers in likes per user. Z-score", likes_per_user_z_score)
    write_df_to_csv(likes_per_user_z_score, "likes_per_user_z_score")

    # 4. Posts that is liked more than others - likes per post
    likes_per_post = df.select("userId", "postId", "likes")

    likes_per_post_iqr = iqr_outlier(likes_per_post, "likes")
    log_dataframe("Anomalies: Outliers in likes per post. IQR", likes_per_post_iqr)
    write_df_to_csv(likes_per_post_iqr, "likes_per_post_iqr")

    likes_per_post_z_score = z_score_outlier(likes_per_post, "likes")
    log_dataframe("Anomalies: Outliers in likes per post. Z-score", likes_per_post_z_score)
    write_df_to_csv(likes_per_post_z_score,"likes_per_post_z_score")

    # 5. Outliers in likes per view
    likes_per_view = (df_posts
                      .select("userId", "postId", "likes", "views")
                      .withColumn("likes_per_view", when(col("views") != 0,
                                                         col("likes") / col("views"))
                                  .otherwise(None)
                                  ))

    likes_per_view_iqr = iqr_outlier(likes_per_view, "likes_per_view")
    log_dataframe("Anomalies: Outliers in likes per view. IQR", likes_per_view_iqr)
    write_df_to_csv(likes_per_view_iqr,"likes_per_view_iqr")

    likes_per_view_z_score = z_score_outlier(likes_per_view, "likes_per_view")
    write_df_to_csv(likes_per_view_z_score,"likes_per_view_z_score")
    log_dataframe("Anomalies: Outliers in likes per view. Z-score", likes_per_view_z_score)

    # 6. Outliers in likes per reaction (like or dislike)
    likes_per_reaction = (df_posts
                          .select("postId", "likes", "dislikes")
                          .withColumn("likes_per_reaction", when((col("likes") + col("dislikes")) != 0,
                                                                 col("likes") / (col("likes") + col("dislikes")))
                                      .otherwise(None)
                                      ))

    likes_per_reaction_iqr = iqr_outlier(likes_per_reaction, "likes_per_reaction")
    log_dataframe("Anomalies: Outliers in likes per reaction. IQR", likes_per_reaction_iqr)
    write_df_to_csv(likes_per_reaction_iqr, "likes_per_reaction_iqr")

    likes_per_reaction_z_score = z_score_outlier(likes_per_reaction, "likes_per_reaction")
    log_dataframe("Anomalies: Outliers in likes per reaction. Z-score", likes_per_reaction_z_score)
    write_df_to_csv(likes_per_reaction_z_score,"likes_per_reactions_z_score")