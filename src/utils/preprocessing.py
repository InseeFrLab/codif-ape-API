import json
import os
import string
from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
import unidecode
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from s3fs import S3FileSystem

from api.models.forms import SingleForm


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


fs = get_file_system()

with fs.open("projet-ape/data/mappings.json") as file_in:
    mappings = json.load(file_in)


text_feature = "libelle"
textual_features = ["NAT_LIB", "AGRI"]
categorical_features = ["TYP", "NAT", "SRF", "CJ", "CRT"]

SURFACE_COLS = ["activ_surf_et", "SRF"]


def clean_text_feature(text: list[str], remove_stop_words=True):
    """
    Cleans a text feature.

    Args:
        text (list[str]): List of text descriptions.
        remove_stop_words (bool): If True, remove stopwords.

    Returns:
        list[str]: List of cleaned text descriptions.

    """

    # Define stopwords and stemmer

    nltk.download("stopwords", quiet=True)
    stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
    stemmer = SnowballStemmer(language="french")

    # Remove of accented characters
    text = np.vectorize(unidecode.unidecode)(np.array(text))

    # To lowercase
    text = np.char.lower(text)

    # Remove one letter words
    def mylambda(x):
        return " ".join([w for w in x.split() if len(w) > 1])

    text = np.vectorize(mylambda)(text)

    # Remove duplicate words and stopwords in texts
    # Stem words
    libs_token = [lib.split() for lib in text.tolist()]
    libs_token = [sorted(set(libs_token[i]), key=libs_token[i].index) for i in range(len(libs_token))]
    if remove_stop_words:
        text = [" ".join([stemmer.stem(word) for word in libs_token[i] if word not in stopwords]) for i in range(len(libs_token))]
    else:
        text = [" ".join([stemmer.stem(word) for word in libs_token[i]]) for i in range(len(libs_token))]

    # Return clean DataFrame
    return text


def clean_categorical_features(df: pd.DataFrame, categorical_features: List[str], y: Optional[str] = None) -> pd.DataFrame:
    """
    Cleans the categorical features for pd.DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame.
        categorical_features (List[str]): Names of the categorical features.
        y (str): Name of the variable to predict.

    Returns:
        df (pd.DataFrame): DataFrame.
    """
    for surface_col in SURFACE_COLS:
        if surface_col in categorical_features:
            df[surface_col] = df[surface_col].astype(float)
            df = categorize_surface(df, surface_col)
    for variable in categorical_features:
        if variable not in SURFACE_COLS:  # Mapping already done for this variable
            if len(set(df[variable].unique()) - set(mappings[variable].keys())) > 0:
                raise ValueError(
                    f"Missing values in mapping for {variable} ",
                    set(df[variable].unique()) - set(mappings[variable].keys()),
                )
            df[variable] = df[variable].apply(mappings[variable].get)
    if y is not None:
        if len(set(df[y].unique()) - set(mappings[y].keys())) > 0:
            raise ValueError(
                f"Missing values in mapping for {y}, ",
                set(df[y].unique()) - set(mappings[y].keys()),
            )
        df[y] = df[y].apply(mappings[y].get)

    return df


def clean_textual_features(
    df: pd.DataFrame,
    textual_features: List[str],
) -> pd.DataFrame:
    """
    Cleans the other textual features for pd.DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame.
        textual_features (List[str]): Names of the other textual features.
        method (str): The method when the function is used (training or
            evaluation).
        recase (bool): if True, try applying standard casing.

    Returns:
        df (pd.DataFrame): DataFrame.
    """
    for textual_feature in textual_features:
        df[textual_feature] = clean_text_feature(df[textual_feature], remove_stop_words=True)
        df[textual_feature] = df[textual_feature].str.replace(
            "nan", ""
        )  # empty string instead of "nan" (nothing will be added to the libelle)
        df[textual_feature] = df[textual_feature].apply(
            lambda x: " " + x if x != "" else x
        )  # add a space before the text because it will be concatenated to the libelle

    return df


def categorize_surface(df: pd.DataFrame, surface_feature_name: int, like_sirene_3: bool = True) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.
        surface_feature_name (str): Name of the surface feature.
        like_sirene_3 (bool): If True, categorize like Sirene 3.

    Returns:
        pd.DataFrame: DataFrame with a new column "surf_cat".
    """
    df_copy = df.copy()
    # Check surface feature exists
    if surface_feature_name not in df.columns:
        raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
    # Check surface feature is a float variable
    if not (pd.api.types.is_float_dtype(df[surface_feature_name])):
        raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

    if like_sirene_3:
        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        # Log transform the surface
        df_copy["surf_log"] = np.log(df[surface_feature_name])

        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy


def preprocess_inputs(inputs: list[SingleForm]) -> dict:
    """
    Preprocess both single and batch inputs using shared logic.
    """

    df = pd.DataFrame([form.model_dump() for form in inputs])

    df = df.rename(
        {
            "description_activity": text_feature,
            "other_nature_activity": textual_features[0],
            "precision_act_sec_agricole": textual_features[1],
            "type_form": categorical_features[0],
            "nature": categorical_features[1],
            "surface": categorical_features[2],
            "cj": categorical_features[3],
            "activity_permanence_status": categorical_features[4],
        },
        axis=1,
    )

    for feature in textual_features:
        df[feature] = df[feature].fillna(value="")
    for feature in categorical_features:
        df[feature] = df[feature].fillna(value="NaN")

    df[text_feature] = clean_text_feature(df[text_feature], remove_stop_words=True)
    df = clean_textual_features(df, textual_features)
    df[text_feature] = df[text_feature] + df[textual_features].apply(lambda x: "".join(x), axis=1)

    # Clean categorical features
    df = clean_categorical_features(df, categorical_features=categorical_features)
    df = df.drop(columns=textual_features)

    return df
