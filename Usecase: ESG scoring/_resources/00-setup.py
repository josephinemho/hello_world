# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# DBTITLE 1,Install required libraries
# MAGIC %pip install beautifulsoup4==4.10.0 PyPDF2==1.26.0 gensim==4.1.2 wordcloud==1.8.1 pyLDAvis==3.3.1 nltk==3.6.1

# COMMAND ----------

# MAGIC %run ../../../_resources/00-global-setup $reset_all_data=$reset_all_data $db_prefix=fsi

# COMMAND ----------

# MAGIC %run ./esg_utils

# COMMAND ----------

import re
from pyspark.sql.functions import col
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pyspark.sql.functions import struct, count, col
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import IsolationForest
