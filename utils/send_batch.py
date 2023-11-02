import os
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
import s3fs


def query_batch_api(
    username: str, password: str, data: pd.DataFrame, nb_echos_max: int = 5, prob_min: float = 0.01
):
    base_url = "https://codification-ape.lab.sspcloud.fr/predict-batch"
    params = {"nb_echos_max": nb_echos_max, "prob_min": prob_min}
    url = f"{base_url}?{urlencode(params)}"

    # Create the request body as a dictionary from the DataFrame
    request_body = data.to_dict(orient="list")
    response = requests.post(url, json=request_body, auth=(username, password))
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400:
        print(response.json()["detail"])
    else:
        print("Error occurred while querying the API.")
        return None


df = pd.read_csv("website/exemples/exemple.csv", dtype=str)
df.replace(np.nan, "", inplace=True)
query_batch_api("codification-ape", "codification-sirene4", df, prob_min=0.01)


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

x = ds.dataset(
    "s3://projet-ape/log_files/preprocessed/",
    partitioning=["date"],
    format="parquet",
    filesystem=fs,
)


s3_path = "s3://projet-ape/log_files/preprocessed/"

dataset = pq.ParquetDataset(s3_path, filesystem=fs)
