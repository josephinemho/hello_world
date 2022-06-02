# Databricks notebook source
# MAGIC %md ### Setup database required for DBSQL Dashboard
# MAGIC This database is shared globally, no need to run it per user, only once per workspace

# COMMAND ----------

# MAGIC %sql create database if not exists field_demos_fsi ;
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`csr_initiatives` LOCATION '/mnt/field-demos/fsi/csr/initiatives';
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`csr_scores` LOCATION '/mnt/field-demos/fsi/csr/scores';
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`csr_statements` LOCATION '/mnt/field-demos/fsi/csr/statements';
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`csr_topics` LOCATION '/mnt/field-demos/fsi/csr/topics';
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`gdelt_gold` LOCATION '/mnt/field-demos/fsi/gdelt/gold';
# MAGIC CREATE TABLE if not exists `field_demos_fsi`.`gdelt_score` LOCATION '/mnt/field-demos/fsi/gdelt/score';

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- GRANT ALL PRIVILEGES ON DATABASE field_demos_fsi TO `quentin.ambard@databricks.com`;
# MAGIC -- REVOKE MODIFY ON DATABASE field_demos_fsi FROM admins;
# MAGIC -- GRANT USAGE, SELECT ON DATABASE field_demos_fsi TO users;
