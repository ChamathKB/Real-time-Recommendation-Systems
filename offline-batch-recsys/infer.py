import numpy as np

import typing as t

import os
import logging
import time
import warnings
warnings.filterwarnings('ignore')

import nvtabular as nvt
import merlin.models.tf as mm
import tensorflow as tf

from nvtabular.ops import *

from merlin.datasets.synthetic import generate_data
from merlin.datasets.ecommerce import transform_aliccp
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags
from merlin.io.dataset import Dataset

import redis.asyncio as redis
from redis.commands.json.path import Path
import asyncio

redis_conn = redis.Redis(
    host="redis-inference-store",
    port=6379,
    decode_responses=True
)

DATA_DIR='/model-data/aliccp'
OUTPUT_RETRIEVAL_DATA_DIR=os.path.join(DATA_DIR, "processed")

topk_model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'query_tower'))
dlrm = tf.keras.models.load_model(os.path.join(DATA_DIR, 'dlrm'))

train_retrieval = Dataset(os.path.join(OUTPUT_RETRIEVAL_DATA_DIR, "train", "*.parquet"), part_size="500MB")
schema = train_retrieval.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER]).without(['user_id_raw', 'item_id_raw', 'click'])

item_fs = unique_rows_by_features(train_retrieval, Tags.ITEM, Tags.ITEM_ID).to_ddf().compute()
user_fs = unique_rows_by_features(train_retrieval, Tags.USER, Tags.USER_ID).to_ddf().compute()


# Softmax sample function for ordering
def softmax_sample(recs: np.array, scores: np.array) -> np.array:
    arr = np.exp(scores)/sum(np.exp(scores))
    top_item_idx = (-arr).argsort()
    return recs[top_item_idx]

def generate_recommendations_batch(topk_model, dlrm, input_data, batch_size: int, tags):
    user_loader = mm.Loader(unique_rows_by_features(input_data, tags.USER, tags.USER_ID), batch_size=batch_size, shuffle=False)
    
    # Load a batch
    for batch in user_loader:
            
        # Generate candidates for this batch of users
        users = batch[0]['user_id']
        _, candidate_item_ids = topk_model(batch[0])
        
        # For each user + candidate items, score with the DLRM
        for user, candidates in zip(users.numpy(), candidate_item_ids.numpy()):
            try:
                num_recs = len(candidates)
                user_id = user[0]

                # get user features
                user_features = user_fs[user_fs.user_id == user_id]
                raw_user_id = user_features.user_id_raw.to_numpy()[0]
                user_features = user_features.append([user_features]*(num_recs-1), ignore_index=True)

                # get item features
                item_features = item_fs[item_fs.item_id.isin(candidates)].reset_index(drop=True)
                raw_item_ids = item_features.item_id_raw.to_numpy()

                # combine into feature vectors
                item_features[user_features.columns] = user_features
                item_features = Dataset(item_features)
                item_features.schema = schema.without(['click'])

                # Score with ranking model
                inputs = mm.Loader(item_features, batch_size=num_recs)
                inputs = next(iter(inputs))
                scores = dlrm(inputs[0]).numpy().reshape(-1)

                # Rank
                recs = softmax_sample(raw_item_ids, scores)

                yield raw_user_id, recs
            except Exception as e:
                logging.info(user_id, str(e))

async def store_recommendations(recommendations: t.Iterable, redis_conn: redis.Redis):
    """
    Store the recommendations generated for each User in Redis.

    Parameters:
        recommendations (t.Iterable): A generator over a dictionary where the keys are user_ids and the values are lists of recommendations
        redis_conn (redis.Redis): A Redis connection object used to store the recommendations in Redis
    """
    async def store_as_json(user_id: str, recs: list):
        """
        Store an individual user's latest recommendations in Redis.

        Parameters:
            user_id (str): The user id of the user whose recommendations are being stored
            recs (list): A list of item_ids representing the recommendations for the user
        """
        entry = {
            "user_id": int(user_id),
            "recommendations": [int(rec) for rec in recs]
        }
        # Set the JSON object in Redis
        await redis_conn.json().set(f"USER:{user_id}", Path.root_path(), entry)
        
    # Write the recommendations to Redis as a JSON object
    for user_id, recs in recommendations:
        await store_as_json(user_id, recs)


# Create Recommendation Denerator
recommendations = generate_recommendations_batch(topk_model, dlrm, train_retrieval, 32, Tags)

# Run the process - may take a few minutes
# await store_recommendations(recommendations, redis_conn=redis_conn)

# run await function using async 
async def main():
    await store_recommendations(recommendations, redis_conn=redis_conn) 

if __name__ == "__main__":
    asyncio.run(main())