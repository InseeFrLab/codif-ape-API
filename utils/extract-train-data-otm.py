import os
import sys

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm


def is_naf_code(text: str):
    return text[:4].isdigit() and text[4].isalpha()


def transform_json_to_dataframe(json_dir: str):
    transformed_data = []
    files_only = [item for item in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, item))]
    for filename in tqdm(files_only):
        with open(os.path.join(json_dir, filename), "r") as file:
            data = json.load(file)

        annotation_date = data["task"]["updated_at"]
        # Get data variables
        libelle = data["task"]["data"]["libelle"]
        liasse_numero = data["task"]["data"]["liasse_numero"]
        date_modification = data["task"]["data"]["date_modification"]
        mode_calcul_ape = data["task"]["data"]["mode_calcul_ape"]
        evenement_type = data["task"]["data"]["evenement_type"]
        liasse_type = data["task"]["data"]["liasse_type"]
        activ_surf_et = str(data["task"]["data"]["activ_surf_et"])
        activ_nat_et = data["task"]["data"]["activ_nat_et"]
        activ_nat_lib_et = data["task"]["data"]["activ_nat_lib_et"]
        activ_sec_agri_et = data["task"]["data"]["activ_sec_agri_et"]
        cj = data["task"]["data"]["cj"]

        # Number of skips
        skips = int(data["was_cancelled"])
        # Get annotated data without skips and adjust from UI's bugs
        if len(data["result"]) > 0:
            apet_manual = ""
            commentary = ""
            rating = 0
            # Retrieve comment
            if "text" in data["result"][0]["value"]:
                commentary = data["result"][0]["value"]["text"][0]
            # Retrieve choice result
            if len(data["result"]) > 1:
                if "choices" in data["result"][1]["value"]:
                    if "Oui" in data["result"][0]["value"]["choices"]:
                        if is_naf_code(data["result"][1]["value"]["choices"][0]):
                            apet_manual = data["result"][1]["value"]["choices"][0]
                # Retrieve taxonomy result
                else:
                    if "taxonomy" in data["result"][1]["value"]:
                        taxonomy_values = data["result"][1]["value"]["taxonomy"][0][-1]
                        apet_manual = taxonomy_values.replace(".", "")  # delete . in apet_manual
                    # Check if apet is in comment and fill empty apet (due to LS bug)
                    if is_naf_code(commentary):
                        print("Potential LS bug --> NAF code in comment: " + commentary)
                        apet_manual = commentary
                # Retrieve rating result
                if len(data["result"]) > 2:
                    if "rating" in data["result"][2]["value"]:
                        rating = data["result"][2]["value"]["rating"]
        # Créer un dictionnaire pour les données transformées
        transformed_row = {
            "liasse_numero": liasse_numero,
            "libelle": libelle,
            "evenement_type": evenement_type,
            "liasse_type": liasse_type,
            "activ_surf_et": activ_surf_et,
            "activ_nat_et": activ_nat_et,
            "activ_nat_lib_et": activ_nat_lib_et,
            "activ_sec_agri_et": activ_sec_agri_et,
            "cj": cj,
            "date_modification": date_modification,
            "annotation_date": annotation_date,
            "mode_calcul_ape": mode_calcul_ape,
            "apet_manual": apet_manual,
            "commentary": commentary,
            "rating": rating,
            "skips": skips,
        }

        # Append in list
        transformed_data.append(transformed_row)

    # Convert to Dataframe
    results = pd.DataFrame(transformed_data)

    # Count skipped and unclassifiable
    print("Number of skips: " + str(len(results[results["skips"] != 0])))
    print("Rate of skips: " + str(len(results[results["skips"] != 0])/len(results)))
    print("Number of unclassifiable: " + str(len(results[results["apet_manual"] == "XXXXX"])))
    print("Rate of unclassifiable: " + str(len(results[results["apet_manual"] == "XXXXX"])/len(results)))

    # Keep only unskipped and classifiable annotations
    results = results[results["skips"] == 0]
    results = results[results["apet_manual"] != "XXXXX"]
    print("Number of lines: " + str(len(results)))

    return results


def save_to_s3(table: pa.Table, bucket: str, path: str, category: str):
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    pq.write_table(table, f"s3://{bucket}/{path}/training_data_{category}_NAF2025.parquet/", filesystem=fs)


def main(annotation_results_path: str, annotation_preprocessed_path: str, category: str):
    # Upload log file from API pod and filter by date
    data = transform_json_to_dataframe(annotation_results_path)

    # Add date column for partitionning
    data["date"] = pd.to_datetime(data["date_modification"]).dt.strftime("%Y-%m-%d")

    data = (
        data[
            [
                "liasse_numero",
                "libelle",
                "evenement_type",
                "liasse_type",
                "activ_surf_et",
                "activ_nat_et",
                "activ_nat_lib_et",
                "activ_sec_agri_et",
                "cj",
                "date",
                "mode_calcul_ape",
                "apet_manual",
                "rating"
            ]
        ]
        .rename(
            columns={
                "libelle": "text_description",
                "liasse_type": "type_",
                "activ_nat_et": "nature",
                "activ_surf_et": "surface",
                "evenement_type": "event",
            }
        )
        .reset_index(drop=True)
    )

    # data['surface'] = data['surface'].astype(bytes)
    # Translate pd.DataFrame into pa.Table
    arrow_table = pa.Table.from_pandas(data)

    # Save logs in a partionned parquet file in s3
    save_to_s3(arrow_table, "projet-ape", f"/{annotation_preprocessed_path}/", category)


if __name__ == "__main__":
    annotation_results_path = str(sys.argv[1])
    annotation_preprocessed_path = str(sys.argv[2])
    category = str(sys.argv[3])

    main(annotation_results_path, annotation_preprocessed_path, category)