# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-9.1_ML-green)]()
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-201-orange)]()
# MAGIC [![POC](https://img.shields.io/badge/DEMO-15mn-yellow)]()
# MAGIC 
# MAGIC *The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. 
# MAGIC 
# MAGIC In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their environmental, social and governance (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. 
# MAGIC 
# MAGIC This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). 
# MAGIC 
# MAGIC In this solution, we offer a novel approach to sustainable finance by combining NLP techniques and news analytics to extract key strategic ESG initiatives and learn companies' commitments to corporate responsibility.*
# MAGIC 
# MAGIC 
# MAGIC ___
# MAGIC + [STAGE 1]($./01_esg_report): learning an ESG vocabulary from CSR reports
# MAGIC + [STAGE 2]($./02_esg_scoring): classifying news articles against ESG policies
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffsi%2Fesg%2Fnotebook_parent&dt=FSI_ESG_PARENT">
# MAGIC <!-- [metadata={"description":"Introduction to ESG Demo. Present the contenxt and what will be implemented.",
# MAGIC  "authors":["antoine.amend@databricks.com"],
# MAGIC  "db_resources":{"Dashboards": ["ESG Report"]},
# MAGIC          "search_tags":{"vertical": "fsi", "step": "Presentation", "components": ["autoloader", "mlflow", "NLP"]},
# MAGIC                       "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://databricks-web-files.s3.us-east-2.amazonaws.com/notebooks/fsi/esg_scoring/images/reference_architecture.png width="800px">
