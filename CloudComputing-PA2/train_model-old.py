from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import re

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load data from system
training_data = spark.read.csv("/Users/sandramariaroy/Downloads/TrainingDataset.csv", header=True, sep=";", quote='"', inferSchema=True)
validation_data = spark.read.csv("/Users/sandramariaroy/Downloads/TrainingDataset.csv", header=True, sep=";", quote='"', inferSchema=True)

print("printing 1st schema")
# Print the schema to check column names
training_data.printSchema()
# Show a few rows of the DataFrame to inspect data
training_data.show(5)

print("printing 2nd schema")
# Function to clean column names by removing double quotes
def clean_column_name(name):
    # return re.sub(r'^"+|"+$', '', name)
    return re.sub(r'^"+|"+$', '', name).replace('""', '')
# Apply the cleaning function to all column names
for col_name in training_data.columns:
    clean_name = clean_column_name(col_name)
    training_data = training_data.withColumnRenamed(col_name, clean_name)
# Verify the cleaned column names
training_data.printSchema()

print("3rd schema")
# Define label and feature columns
label_column = "quality"
feature_columns = [col for col in training_data.columns if col != label_column]
# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(training_data)
# Check the DataFrame with features column
df.select("features", label_column).show(5)
# Train the model
lr = LogisticRegression(labelCol=label_column, featuresCol="features")
model = lr.fit(df)

# Assemble features
assembler = VectorAssembler(inputCols=training_data.columns[:-1], outputCol="features")
training_data = assembler.transform(training_data).select("features", "quality")
validation_data = assembler.transform(validation_data).select("features", "quality")

# Train logistic regression model
lr = LogisticRegression(labelCol="quality", featuresCol="features", maxIter=10)
lr_model = lr.fit(training_data)

# Save the model
lr_model.save("/Users/sandramariaroy/Downloads/wine-quality-model")

# Evaluate model
predictions = lr_model.transform(validation_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score}")

spark.stop()
