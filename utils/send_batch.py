import os
import sys
from urllib.parse import urlencode

import pandas as pd
import pyarrow.dataset as ds
import requests
import s3fs


def query_batch_api(
    username: str,
    password: str,
    data: pd.DataFrame,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
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


def reclassify_surface(d: str):
    try:
        if int(d) < 120:
            return "1"
        elif 121 <= int(d) <= 399:
            return "2"
        elif 400 <= int(d) <= 2499:
            return "3"
        else:
            return "4"
    except ValueError:
        return "null"


def get_filesystem():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    return fs


def format_query(
    df: pd.DataFrame,
):
    subset = df.copy()
    subset["surface"] = subset["surface"].apply(reclassify_surface)
    subset = subset[subset["text_description"] != "NA"]
    return subset[["text_description", "type_", "nature", "surface", "event"]]


def main(log_file_path: str, date_to_log: str):
    # Define file system
    fs = get_filesystem()

    # Open Dataset
    data = (
        ds.dataset(
            f"{log_file_path}",
            partitioning=["date", "sourceAppel"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter(
            (ds.field("sourceAppel") == "sourceAppel=WF")
            & (ds.field("date") == f"date={date_to_log}")
        )
        .to_pandas()
    )

    # Harmonize dataset for the query
    data = format_query(data)

    query_batch_api(os.getenv("API_USERNAME"), os.getenv("API_PASSWORD"), data, prob_min=0.0)


if __name__ == "__main__":
    log_file_path = str(sys.argv[1])
    date_to_log = str(sys.argv[2])

    main(log_file_path, date_to_log)
