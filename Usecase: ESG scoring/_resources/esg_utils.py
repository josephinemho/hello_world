# Databricks notebook source
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
import mlflow.sklearn
import nltk
nltk.download('wordnet')
nltk.download('punkt')

# COMMAND ----------

import pandas as pd
from io import StringIO

esg_df = pd.read_csv(StringIO("""organisation|description|ticker|url|logo
Alliance Data Systems|Alliance Data Systems Corporation, together with its subsidiaries, provides data-driven and transaction-based marketing and customer loyalty solutions primarily in the United States and Canada|ADS|https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/a/NYSE_ADS_2018.pdf|https://www.responsibilityreports.com/HostedData/CompanyLogos/NYSE_ADS.png
American Express|American Express Company provides charge and credit payment card products, and travel-related services worldwide.|AXP|https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/a/NYSE_AXP_2018.pdf|https://www.responsibilityreports.com/HostedData/CompanyLogos/NYSE_AXP.png
Capital One|Capital One Financial Corporation operates as the bank holding company for the Capital One Bank (USA), National Association and Capital One, National Association, which provide various financial products and services in the United States, Canada, and the United Kingdom|COF|https://www.responsibilityreports.com/Click/1640|https://www.responsibilityreports.com/HostedData/CompanyLogos/untitledgildfsp.bmp
Discover Financial|Discover Financial Services operates as a credit card issuer and electronic payment services company primarily in the United States. The company offers Discover Card-branded credit cards to individuals and small businesses over the Discover Network|DFS|https://www.responsibilityreports.com/Click/2357|https://www.responsibilityreports.com/HostedData/CompanyLogos/NYSE_DFS.png
Equifax|Equifax Inc. collects, organizes, and manages various financial, demographic, employment, and marketing information solutions for businesses and consumers|EFX|https://www.responsibilityreports.com/Click/1346|https://www.responsibilityreports.com/HostedData/CompanyLogos/NYSE_EFX.png
PayPal|PayPal is a leading technology platform company that enables digital and mobile payments on behalf of consumers and merchants worldwide. They put their customers at the center of everything they do. They strive to increase our relevance for consumers, merchants, friends and family to access and move their money anywhere in the world, anytime, on any platform and through any device.|PYPL|https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/p/NASDAQ_PYPL_2018.pdf|https://www.responsibilityreports.com/HostedData/CompanyLogos/NASDAQ_PYPL.png
Provident Financial|Provident Financial plc provides personal credit products to non-standard lending market in the United Kingdom and Ireland.|PFG|https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/p/LSE_PFG_2017.pdf|https://www.responsibilityreports.com/HostedData/CompanyLogos/NFJHGHJ.PNG"""),sep='|')

# COMMAND ----------

# DBTITLE 1,Transform bytes from pdf to a text
from pyspark.sql.functions import pandas_udf, col, array_max, regexp_extract, arrays_zip, array, lit, lower, row_number
import pandas as pd
from PyPDF2 import PdfFileReader
from io import BytesIO
from pyspark.sql import functions as F
from pyspark.sql.window import Window

@pandas_udf("string")
def extract_text_from_pdf(s: pd.Series) -> pd.Series:
    #Scan the folder having the pdf files
    def binary_to_text(binary_data):
      pdf = PdfFileReader(BytesIO(binary_data), strict=False)  
      text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
      return "\n".join(text)
    return s.apply(binary_to_text)

# COMMAND ----------

#Join the dataframe with metadata previously extracted (logo & organization name/description)
def join_with_metadata(pdf_df): 
  #Extract the images
  logo = spark.read.format('binaryFile').load('/mnt/field-demos/fsi/responsibilityreports/logo')
  logo = logo.withColumn("ticker", regexp_extract("path", "[\w-]+?(?=\.)", 0))
  logo = logo.withColumnRenamed("content", "logo_content")
  #Join images with metadata
  esg = spark.createDataFrame(esg_df)
  metadata = logo.join(esg, on='ticker').select("organisation", "ticker", "logo_content", "description")
  pdf_df = pdf_df.withColumn("ticker", regexp_extract("path", "[\w-]+?(?=\.)", 0))
  pdf_df = pdf_df.withColumnRenamed("content", "pdf_content").select("pdf_content", "ticker")
  #Join metadata + logo with pdf content
  return metadata.join(pdf_df, on='ticker').select("organisation", "pdf_content", "logo_content", "ticker", "description")


# COMMAND ----------

import string
import re
from typing import Iterator
from pyspark.sql.functions import explode

def text_to_sentence(text):
  # remove non ASCII characters
  printable = set(string.printable)
  text = ''.join(filter(lambda x: x in printable, text))
  
  lines = []
  prev = ""
  for line in text.split('\n'):
    # aggregate consecutive lines where text may be broken down
    # only if next line starts with a space or previous does not end with a dot.
    if(line.startswith(' ') or not prev.endswith('.')):
        prev = prev + ' ' + line
    else:
        # new paragraph
        lines.append(prev)
        prev = line
        
  # don't forget left-over paragraph
  lines.append(prev)

  # clean paragraphs from extra space, unwanted characters, urls, etc.
  # best effort clean up, consider a more versatile cleaner
  sentences = []
  
  for line in lines:
      # removing header number
      line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
      # removing trailing spaces
      line = line.strip()
      # words may be split between lines, ensure we link them back together
      line = re.sub(r'\s?-\s?', '-', line)
      # remove space prior to punctuation
      line = re.sub(r'\s?([,:;\.])', r'\1', line)
      # ESG contains a lot of figures that are not relevant to grammatical structure
      line = re.sub(r'\d{5,}', r' ', line)
      # remove mentions of URLs
      line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
      # remove multiple spaces
      line = re.sub(r'\s+', ' ', line)
      # remove multiple dot
      line = re.sub(r'\.+', '.', line)
      
      # split paragraphs into well defined sentences using nltk
      for part in nltk.sent_tokenize(line):
        sentences.append(str(part).strip())
  return sentences

@pandas_udf("array<string>")
def extract_statements(itr: Iterator[pd.Series]) -> Iterator[pd.Series]:
    nltk.download('wordnet')
    nltk.download('punkt')
    for s in itr:
        yield s.apply(text_to_sentence)

# COMMAND ----------

from pyspark.sql.functions import length
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.utils import simple_preprocess

#lemmatizer UDF
@pandas_udf("string")
def lemmatize_text(s: pd.Series) -> pd.Series:
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()
  def lemmatize(text):
    results = []
    for token in simple_preprocess(text):
      stem = stemmer.stem(lemmatizer.lemmatize(token))
      if (len(stem) > 3):
        results.append(stem)
    return ' '.join(results)
  
  return s.apply(lemmatize)

from nltk import word_tokenize          
from sklearn.feature_extraction import text

#lemmatizer as sklearn tokenizer
class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_word = text.ENGLISH_STOP_WORDS
    def __call__(self, articles):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(articles) if t not in self.stop_word and len(t) > 3]

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
nltk.download('punkt')
import mlflow
import mlflow.pyfunc
import sklearn
import gensim

def get_LDA_model_env():
  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['scikit-learn=={}'.format(sklearn.__version__)]
  conda_env['dependencies'][2]['pip'] += ['gensim=={}'.format(gensim.__version__)]
  conda_env['dependencies'][2]['pip'] += ['nltk=={}'.format(nltk.__version__)]
  conda_env['dependencies'][2]['pip'] += ['pandas=={}'.format(pd.__version__)]
  conda_env['dependencies'][2]['pip'] += ['numpy=={}'.format(np.__version__)]
  return conda_env
  
class LDAModel(mlflow.pyfunc.PythonModel):
  import nltk
    
  def __init__(self, pipeline):
    self.pipeline = pipeline  
    self.nltk_initiated = False
    
  def init_nltk(self):
    if not self.nltk_initiated:
      nltk.download('wordnet')
      nltk.download('punkt')
      self.nltk_initiated = True

  def predict(self, model_input):
    self.init_nltk()
    #special case with single param model
    if isinstance(model_input, pd.DataFrame):
        #take the first column of our dataframe. Can be named (statement) or not (mlflow.pyfunc.spark_udf calls it without named column)
        model_input = model_input[model_input.columns[0]]
    predictions = pipeline.transform(model_input)
    return predictions.tolist()
  
#return the position of the max value (np argmax equivalent)
@pandas_udf("float")
def array_max_position(s: pd.Series) -> pd.Series:
    return s.apply(lambda x : np.argmax(x))


# COMMAND ----------

#Download the given artifact and return it as a json object
def get_json_artifact(run_id, artifact):
  import json
  client = MlflowClient()
  path = client.download_artifacts(run_id, artifact, '/tmp')
  with open(path) as json_file:
      return json.load(json_file)

# COMMAND ----------

# DBTITLE 1,Chart helpers
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt  

def esg_heatmap(esg_group, color):
  scaler = MinMaxScaler(feature_range = (0, 1))
  esg_group = pd.pivot_table(
    esg_group, 
    values='esg', 
    index='organisation',
    columns=['policy'], 
    aggfunc=np.mean)

  esg_focus = pd.DataFrame(scaler.fit_transform(esg_group), columns=esg_group.columns)
  esg_focus.index = esg_group.index

  # plot heatmap, showing main area of focus for each company across topics we learned
  sns.set(rc={'figure.figsize':(12,8)})
  sns.heatmap(esg_focus, annot=False, cmap=color)
  plt.show()
  
def build_esg_heatmap(esg_group, color, topic_names_df): 
  esg_group = esg_group.toPandas()
  esg_group['topics'] = esg_group['probabilities'].apply(lambda xs: [[i, x] for i, x in enumerate(xs)])
  esg_group = esg_group.explode('topics').reset_index(drop=True)
  esg_group['topic_id'] = esg_group['topics'].apply(lambda x: x[0])
  esg_group['probability'] = esg_group['topics'].apply(lambda x: x[1])
  esg_group = esg_group[['organisation', 'topic_id', 'probability']]
  esg_group = esg_group.merge(topic_names_df.toPandas(), on='topic_id')
  esg_group = esg_group.rename({'probability': 'esg'}, axis=1)
  esg_heatmap(esg_group, color)

  
  
def plot_esg_bar(esg_csr_data):
  esg_csr_data['sum'] = esg_csr_data.sum(axis=1)
  esg_csr_data = esg_csr_data.sort_values(by='sum', ascending=False).drop('sum',  axis=1)
  esg_csr_data.plot.bar(
    rot=90, 
    stacked=False, 
    color={"E": "#A1D6AF", "S": "#D3A1D6", "G": "#A1BCD6"},
    title='ESG score based on corporate disclosures',
    ylabel='ESG score',
    ylim=[0, 100],
    figsize=(16, 8))

# COMMAND ----------

#zip the probabilities with the topic informations (from the model runheatmap)
def with_topics(col_proba, topic_names):
  #make sure we're sorted by id
  topic_names = sorted(topic_names, key = lambda i: i['topic_id'])
  topic_id = array([lit(i) for i in range(len(topic_names))])
  topic = array([lit(topic_names[i]['topic']) for i in range(len(topic_names))])
  policy = array([lit(topic_names[i]['policy']) for i in range(len(topic_names))])
  return arrays_zip(col_proba.alias('probability'), topic_id.alias('topic_id'), topic.alias('topic'), policy.alias('policy'))
