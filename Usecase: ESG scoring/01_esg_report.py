# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC # Analyze CSR reports to score organisations & drive investment
# MAGIC 
# MAGIC Any large scale organisation is now facing tremendous pressure from their shareholders to disclose more information about their environmental, social and governance strategies. Typically released on their websites on a yearly basis as a form of a PDF document, companies communicate on their key ESG initiatives across multiple themes such as how they value their employees, clients or customers, how they positively contribute back to society or even how they reduce (or commit to reduce) their carbon emissions.
# MAGIC 
# MAGIC In this notebook, we would like to programmatically access CSR reports from top tier financial services institutions and learn key ESG initiatives across different topics.
# MAGIC 
# MAGIC Ultimately, we'll have structured ESG informations for each organisation and be able to leverage them to drive our investments.
# MAGIC 
# MAGIC To achieve this result, we'll implement the following steps: 
# MAGIC 
# MAGIC <img src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-0.png' width=1000>
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffsi%2Fesg%2Fnotebook_esg_report&dt=FSI_ESG_REPORT">
# MAGIC <!-- [metadata={"description":"Analyse ESG content using NLP. Text preprocessing, topic modeling with LDA.",
# MAGIC  "authors":["antoine.amend@databricks.com"],
# MAGIC  "db_resources":{"Dashboards": ["ESG Report"]},
# MAGIC          "search_tags":{"vertical": "fsi", "step": "Presentation", "components": ["autoloader", "mlflow", "NLP"]},
# MAGIC                       "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Load PDF using Databricks Autoloader and extract text content
# MAGIC 
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-1.png' width=600>
# MAGIC 
# MAGIC We search for publicly available corporate sustainability documents from publicly traded organizations. Instead of going through each company website, one could access information from [responsibilityreports.com](https://www.responsibilityreports.com), manually browsing for content or automatically scraping new records (please check T&Cs). 
# MAGIC For the purpose of this demo, we'll download and use a subset of CSR reports (pdf) for a few financial services organizations. These pdf are stored in our cloud storage.
# MAGIC 
# MAGIC Once the PDF are available, we can leverage Databricks Autoloader to scan the folder, read each PDF and use `PyPDF2` library to extract the text from the pdf binary content. For more information, please refer to the utility [notebook]($./_resources/esg_utils).

# COMMAND ----------

df = spark.readStream.format("cloudFiles") \
           .option('cloudFiles.format', 'binaryFile') \
           .load('/mnt/field-demos/fsi/responsibilityreports/pdf')
  
#Extract the content from the pdf
df = df.withColumn('content', extract_text_from_pdf(col('content')))
#Add external metadata (organisation name, description)
df = join_with_metadata(df)

df.writeStream \
    .trigger(once=True) \
    .option("checkpointLocation", cloud_storage_path+"/checkpoints/raw_documents") \
    .table("raw_documents").awaitTermination()

# COMMAND ----------

# MAGIC %sql select organisation, substring(pdf_content, 0, 5000), logo_content from raw_documents

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Text preparation - sentence extraction
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-2.png' width=600>
# MAGIC PDFs are highly unstructured by nature with text that is often scattered across multiple lines, pages, columns. From a simple set of regular expressions to a more complex NLP model (we use a [nltk](https://www.nltk.org/) trained pipeline), we show how to extract clean sentences from raw text documents. 
# MAGIC 
# MAGIC For more information, please refer to the  utility [notebook]($./_resources/esg_utils).

# COMMAND ----------

# DBTITLE 1,Splitting the text to sentences
raw_documents = spark.read.table("raw_documents")
#Transform the raw text in a collection of sentence
clean_sentence_df = raw_documents.withColumn("statement", extract_statements(col("pdf_content"))) \
                                 .withColumn("statement", explode(col("statement"))) \
                                 .filter(length(col("statement")) >= 255) \
                                 .select("statement", "ticker", "organisation")

clean_sentence_df.write.mode("overwrite").saveAsTable("clean_sentences")
display(spark.read.table("clean_sentences"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Topic modeling
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-3.png' width=600>
# MAGIC ### Model training
# MAGIC In this section, we apply [latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA) to learn topics descriptive to CSR reports. We want to be able to better understand and sumarize complex CSR reports into a specific ESG related themes (such as 'valuing employees'). The challenge of topic modelling is to extract good quality of topics that are clear and meaningful. This depends heavily on the quality of text preprocessing, the amount of data to learn from and the strategy of finding the optimal number of topics. With more data (more PDFs), we may learn more meaningful insights.

# COMMAND ----------

sentence_df = spark.read.table("clean_sentences").toPandas()
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode',
                                lowercase = True)
dataset = sentence_df["statement"]

# COMMAND ----------

# MAGIC %md
# MAGIC In the actual solution accelerator, we show how to tune LDA hyperparameters through the use of [hyperopt](https://github.com/hyperopt/hyperopt) that we distribute over spark executors. In this demo, we explicitly specify the number of topics (5).

# COMMAND ----------

with mlflow.start_run(run_name='esg_lda') as run:
  mlflow.sklearn.autolog(silent=True, log_models=False)
  lda = LatentDirichletAllocation(n_components=5, random_state=42)
  
  # train pipeline
  pipeline = make_pipeline(vectorizer, lda)
  pipeline.fit(dataset)

  # log model
  signature = infer_signature(dataset, pipeline.transform(dataset))
  input_example = {"statement": "We monitor our annual spend with suppliers that are CSR certified."}
  mlflow.sklearn.log_model(LDAModel(pipeline), 'pipeline', signature = signature, input_example = input_example, conda_env = get_LDA_model_env())
  #Let's get back the run ID as we'll need to add other figures in our run from another cell
  run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### Interpreting results
# MAGIC We want to evaluate model relevance using more domain expertise. Would those topics make sense from an ESG perspective? Do we have clear categories defined spanning accross the Environmental, Social and Governance broader categories? By interacting with our model through simple visualization, we want to name each topic into a specific policy in line with [GRI standards](https://www.globalreporting.org/standards/) (if possible). 

# COMMAND ----------

lda_display = pyLDAvis.sklearn.prepare(lda, vectorizer.transform(sentence_df["statement"]), vectorizer, mds='tsne')
pyLDAvis.display(lda_display)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving interpretation with model
# MAGIC For different PDF documents, we may have to look at wordcloud visualizations above and rename below topics accordingly. As mentioned, different data will yield different insights, and more documents to learn from may result into more specific themes / categories. 

# COMMAND ----------

#Getting back the main run
with mlflow.start_run(run_id):
  #Give a name to the topics. This could be done automatically with a small set of data labelled, or manually in this case
  topic_names = [{"topic_id": 0, "topic": "S", "policy": "ethics"},
                 {"topic_id": 1, "topic": "S", "policy": "community"},
                 {"topic_id": 2, "topic": "S", "policy": "workplace"},
                 {"topic_id": 3, "topic": "G", "policy": "governance"},
                 {"topic_id": 4, "topic": "E", "policy": "environment"}]
  
  mlflow.log_dict(topic_names, "topic_names.json")
  #Save our HTML cluster visualisation
  pyLDAvis.save_html(lda_display, '/tmp/lda_esg.html')
  mlflow.log_artifact('/tmp/lda_esg.html')

# COMMAND ----------

# MAGIC %md
# MAGIC We can infer topic distribution for each sentence in our reports that we store to a delta table. 

# COMMAND ----------

#save the model in MLFlow registry
model_registered = mlflow.register_model("runs:/"+run_id+"/pipeline", "field_demos_ESG_topic_modeling")
#push model to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name = "field_demos_ESG_topic_modeling", 
                                      version = model_registered.version, 
                                      stage = "Production", 
                                      archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Building topics using our Model
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-4.png' width=600>
# MAGIC Once the model is saved in MLFlow, we can load it and get the topic for each statement

# COMMAND ----------

#                                                                                   Stage/version   
#                                                                 Model name              |        Return type
#                                                                     |                   |             |
model_topic = mlflow.pyfunc.spark_udf(spark, 'models:/field_demos_ESG_topic_modeling/Production', "array<float>")
#We can also register the model as SQL function
spark.udf.register("model_topic", model_topic)

# COMMAND ----------

# MAGIC %sql select model_topic(statement), * from clean_sentences

# COMMAND ----------

topic_df = spark.read.table("clean_sentences") \
                       .withColumn("probabilities", model_topic(col("statement"))) \
                       .withColumn('probability', array_max(col("probabilities"))) \
                       .withColumn('topic_id', array_max_position(col("probabilities")))
topic_df.write.mode("overwrite").saveAsTable("topic_sentences")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Extract initiatives
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-5.png' width=600>
# MAGIC Using a partitioning window, we extract the most descriptive policies for each organization. Despite some obvious misclassification (remember that we used topic modelling to discover unknown themes rather than classifying known labels), we show how one could dramatically simplify a complex PDF document of hundreds of pages into specific initiatives, answering questions like "*What did company X do with regards to environmental policy?*"

# COMMAND ----------

#Get the name of our topics from the model artifact
topic_names = get_json_artifact(run_id, "topic_names.json")
topic_names_df = sqlContext.read.json(sc.parallelize(topic_names))

#get the top 1 sentence for each topic and each organisation
window = Window.partitionBy('organisation', 'topic_id').orderBy(F.desc('probability'))
initiatives = (spark.read.table("topic_sentences")
                   .withColumn('rank', F.row_number().over(window))
                  .join(topic_names_df, ['topic_id'])
                  .filter(F.col('probability') > 0.8)
                  .filter(F.col('rank') == 1)
                  .orderBy(F.desc('probability'))
                  .select('organisation', 'statement', 'topic', 'policy', 'ticker', 'probabilities', 'topic_id'))

initiatives.write.mode('overwrite').saveAsTable("esg_initiatives")

display(spark.read.table("esg_initiatives"))

# COMMAND ----------

# DBTITLE 1,Adding a Zorder to speedup search by organization or ticker
# MAGIC %sql OPTIMIZE esg_initiatives ZORDER BY (ticker)

# COMMAND ----------

# MAGIC %md
# MAGIC As our framework was built around the use of AI, the themes we learned from will be consistent across every organisations. In addition to summarizing complex PDF documents, this framework can be used to objectively compare initiatives across organisations, answering questions like "*How much more does company X communicate on the wellbeing of their employees compare to company Y?*". For that purpose, we create a simple pivot table that will summarize companies' focus across our machine learned policies.

# COMMAND ----------

esg_group = spark.read.table("esg_initiatives")
build_esg_heatmap(esg_group, 'Greens', topic_names_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we were able to create a framework that helps financial analysts objectively assess the sustainable impact of their investments, retailers to compare the ethical posture of their suppliers or organisations to compare their environmental initiatives with their closest competitors.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## CSR score
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-6.png' width=600>
# MAGIC In the previous section, we set the foundations to a AI driven ESG framework by learning key ESG initiatives as opposition to broad statements. By looking at how descriptive each statement is, we create a simple score by rank ordering organisations. 

# COMMAND ----------

esg_iniatives = spark.read.table("esg_initiatives")
number_orgs = esg_iniatives.select("ticker").distinct().count()
window = Window.partitionBy('topic_id').orderBy('esg')

df = (
  esg_iniatives
    .withColumn('probabilities', with_topics(col('probabilities'), topic_names)) \
    .withColumn('probabilities', explode(col('probabilities')))
    .groupBy('probabilities.topic_id', 'organisation')
    .agg(F.sum('probabilities.probability').alias('esg'))
    .withColumn('rank', F.row_number().over(window))
    .withColumn('score', F.round(col('rank') * 100 / F.lit(number_orgs)))
    .select('organisation', 'topic_id', 'score')
    .write
      .mode('overwrite')
      .saveAsTable('csr_scores')
)
display(spark.read.table('csr_scores'))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Building CSR Dashboard
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-7.png' width=600>
# MAGIC We store our scores on a delta table that will be combined in our next notebook with news analytics and can be visualized as-is. We represent companies ESG focus across the E, S and G using a simple bar chart.

# COMMAND ----------

esg_csr_data = ( 
  spark
    .read
    .table('csr_scores')
    .join(topic_names_df, ['topic_id'])
    .groupBy('organisation', 'topic')
    .agg(F.avg('score').alias('score'))
    .toPandas()
    .pivot(index='organisation', columns='topic', values='score')
)  
plot_esg_bar(esg_csr_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Dashboard with Databricks SQL
# MAGIC Once our final tables are created, we can leverage Databricks SQL to create our interactive dashboard, or use an exteran BI tool.
# MAGIC 
# MAGIC <img src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-dashboard.png' width=1000 />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take away
# MAGIC In this section, we set the foundations to a data driven ESG framework. We've demonstrated how AI can be used to extract key ESG initiatives from unstructured PDF documents and use this intelligence to create a more objective way to quantify ESG strategies from public companies. With the vocabulary we have learned using topic modelling, we can re-use that model to see how much of these initiatives were actually followed through and what was the media reception using news analytics data.
# MAGIC 
# MAGIC Continue with the [the second part]($./02_esg_scoring)
