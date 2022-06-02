# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC # News analytics
# MAGIC 
# MAGIC As covered in the previous section, we were able to compare businesses side by side across different ESG initiatives. Although we created a simple ESG score, we want our score **not to be subjective but truly data driven**. In other terms, we do not want to solely base our assumptions on companies’ official disclosures but rather on how companies' reputations are perceived in the media, across all 3 environmental, social and governance variables. 
# MAGIC 
# MAGIC For that purpose, we will be using [GDELT](https://www.gdeltproject.org/), the global database of event location and tones as a building block to that framework.
# MAGIC 
# MAGIC Here is the flow we'll implement at a high level: start by downloading the GDELT data and use them to compute ESG score for each organisation.
# MAGIC 
# MAGIC <img src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-scoring-gdelt-0.png' width=900>
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffsi%2Fesg%2Fnotebook_esg_gdelt_report&dt=FSI_ESG_GDELT_REPORT">
# MAGIC <!-- [metadata={"description":"Score ESG with GDELT content using NLP. Text preprocessing, topic modeling with LDA.",
# MAGIC  "authors":["antoine.amend@databricks.com"],
# MAGIC  "db_resources":{"Dashboards": ["ESG Report"]},
# MAGIC          "search_tags":{"vertical": "fsi", "step": "Presentation", "components": ["autoloader", "mlflow", "NLP"]},
# MAGIC                       "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Retrieve news
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-scoring-gdelt-1.png' width=600>
# MAGIC *Supported by Google Jigsaw, the [GDELT](https://www.gdeltproject.org/) Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes, images and events driving our global society every second of every day, creating a free open platform for computing on the entire world.* 
# MAGIC 
# MAGIC For the purpose of that demo, we pre-fetched news articles data for the few financial services organisations we are interested in and saved them in our local cloud storage. 
# MAGIC 
# MAGIC For more information about the acquisition and processing of GDELT files, please refer to our [ESG solution accelerator](https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/fsi/esg_scoring/index.html#esg_scoring_4-0.html)

# COMMAND ----------

# DBTITLE 1,Exploring the GDELT table
gdelt_df = spark.read.load('/mnt/field-demos/fsi/gdelt/raw')
display(gdelt_df)

# COMMAND ----------

# MAGIC %sql  -- plot Options: bar, key: "date", value: "article_count"
# MAGIC select count(*) as article_count, to_date(date) as date, organisation 
# MAGIC   from delta.`/mnt/field-demos/fsi/gdelt/raw` group by date, organisation order by date

# COMMAND ----------

# MAGIC %md
# MAGIC As part of the preparation exercise, we already processed GDELT records to only capture key attributes such as title and sentiment for only those specific organisations.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Classify news
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-scoring-gdelt-2.png' width=600>
# MAGIC Although GDELT provide us with a lot of content already, we can take an extra step and scrape the actual article content (we show how to in the associated [solution accelerator](https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/fsi/esg_scoring/index.html#esg_scoring_4-3.html)).  However, for the purpose of that demo, we focus on article title that is already part of GDELT metadata. 
# MAGIC 
# MAGIC In this section, we retrieve the model we trained in our previous stage and apply its predictive value to news title in order to describe news articles into consistent categories.

# COMMAND ----------

# MAGIC %md
# MAGIC We retrieve the NLP model previously trained on CSR reports as a MLflow artifact that we wrap to a user defined function for maximum parallelism. For that purpose, we make use of `pandas_udf` to apply native python operations on spark dataframes.

# COMMAND ----------

#                                                                                Stage/version   
#                                                              Model name              |        Return type
#                                                                  |                   |             |
classify = mlflow.pyfunc.spark_udf(spark, 'models:/field_demos_ESG_topic_modeling/Production', "array<float>")
#Get our topics name from the model artifact
run_id = mlflow.pyfunc.load_model('models:/field_demos_ESG_topic_modeling/Production').metadata.run_id
topic_names = get_json_artifact(run_id, "topic_names.json")

# COMMAND ----------

# MAGIC %md
# MAGIC For each article, we want to extract the topic Id associated to each probability in order to derive an ESG score based on article tone and topic relevance and save them as our gold table.

# COMMAND ----------

gdelt_probabilities = gdelt_df.withColumn('probabilities', classify('title')) \
                              .select('date', 'organisation', 'url', 'tone', 'title', 'probabilities')

gdelt_probabilities.withColumn('probabilities', with_topics(col('probabilities'), topic_names)) \
                   .write.mode('overwrite').saveAsTable('gdelt_topics')

display(spark.read.table('gdelt_topics'))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ESG score
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-scoring-gdelt-3.png' width=600>
# MAGIC 
# MAGIC Similar to our previous notebook, we can represent the media / sentiment coverage for each organisation. How much more negative, or positive each organisation is mentioned in the news against our ESG policies?

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

esg_weighted =(spark
  .read
  .table('gdelt_topics')
  .withColumn('probabilities', explode('probabilities'))
  .select('*', 'probabilities.*')
  .withColumn('weightedTone', col('probability') * col('tone'))
  .groupBy('organisation', 'policy', 'topic', 'topic_id')
      .agg(
        F.sum('weightedTone').alias('weightedTone'),
        F.sum('probability').alias('weights')
      )
 .withColumn('esg', col('weightedTone') / col('weights')))

esg_heatmap(esg_weighted.toPandas(), 'Blues')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC In the previous section, we showed how the intel we learned from CSR reports could be transfered into the world of news to describe any article against a set of ESG policies. 
# MAGIC 
# MAGIC Using sentiment analysis (tone is part of the GDELT metadata), we aim at detecting how much more "positive" or "negative" a company is perceived across those machine learned policies. We create a score internal to each company across its 'E', 'S' and 'G' dimensions.

# COMMAND ----------

orgs = esg_weighted.select('organisation').distinct().count()
(esg_weighted
    .withColumn('score', row_number().over(Window.partitionBy('policy').orderBy('esg')))
    .withColumn('score', col('score') * lit(100) / lit(orgs))
    .write
      .mode('overwrite')
      .saveAsTable('gdelt_scores'))

spark.read.table('gdelt_scores').display()

# COMMAND ----------

esg_gdelt_data = ( 
  spark
    .read
    .table('gdelt_scores')
    .groupBy('organisation', 'topic')
    .agg(F.avg('score').alias('score'))
    .toPandas()
    .pivot(index='organisation', columns='topic', values='score')
)
plot_esg_bar(esg_gdelt_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have been able to describe news information using the insights we learned from corporate responsibility reports. Only setting the foundation, we highly recommend user to train a model with more data to learn more specific ESG categories and access underlying HTML articles content as previously described. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Walking the talk
# MAGIC <img style='float: right; margin-left: 10px' src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-scoring-gdelt-4.png' width=600>
# MAGIC We can combine our insights generated from our 2 notebooks to get a sense of how much each companies' initiative were followed through and what was the media coverage (positive or negative). 
# MAGIC 
# MAGIC Given our objective and data driven approach (rather than subjective scoring going through a PDF document), we can detect organisations "walking the talk". 

# COMMAND ----------

esg_walk = spark.read.table('gdelt_scores').withColumnRenamed('score', 'walk')
#NOTE: csr_scores is from the first notebook, please run it first to be able to compare the organisation publication with gdelt content!
esg_talk = spark.read.table('csr_scores').withColumnRenamed('score', 'talk').withColumn('organisation', lower(col('organisation')))

display(
  esg_walk
    .join(esg_talk, on=['organisation', 'topic_id'])
    .withColumn('walkTheTalk', F.col('walk') - F.col('talk'))
    .orderBy(F.desc('walkTheTalk'))
    .select('organisation', 'topic_id', 'policy', 'walk', 'talk')
)
#display Bar Plot, Keys: organisations, Values: walk, talk

# COMMAND ----------

# MAGIC %md
# MAGIC On the left side, we see organizations scoring higher using news analytics than CSR reports. Those companies may not disclose a lot of information as part of their yearly disclosures (and arguably may have a poor ESG score from rating agencies) but consistently do good. Their support to communities or environmental impact is noticed and positive. 
# MAGIC 
# MAGIC On the other side come organisations disclosing more than actually doing or organisations constantly mentionned negatively in the press. However, you may notice that we did not take media bias into account here. Organisations mentioned more frequently than others tend to have a more negative score due to the negative nature of news analytics.
# MAGIC 
# MAGIC We leave this as an open discussion for future solution.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Dashboard with Databricks SQL
# MAGIC We can now explore the entire Dashboard, comparing walk/talk for each company.
# MAGIC 
# MAGIC <img src='https://github.com/QuentinAmbard/databricks-demo/raw/main/fsi/resources/esg-report-dashboard.png' width=1000 />
# MAGIC 
# MAGIC [Open the interactive Dashboard](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/606ab63e-37e7-4e7b-be61-c073631eda49-esg_csr_demo?o=1444828305810485&p_org=paypal)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take Away
# MAGIC Throughout this series of notebooks, we introduced a novel approach to environmental, social and governance to objective quantify the ESG impact of public organisations using AI. By combining corporate disclosure and news analytics data, we demonstrated how machine learning could be used to bridge the gap between what companies say and what companies actually do. We touched on a few technical concepts such as MLFlow and Delta lake to create the right foundations for you to extend and adapt to specific requirements. 
