

CS 643, Cloud Computing - Programming Assignment 2

**Goal:** The purpose of this individual assignment is to learn how to develop parallel machine learning (ML)
applications in Amazon AWS cloud platform. Specifically, you will learn: 
(1) how to use Apache Spark to
train an ML model in parallel on multiple EC2 instances; (2) how to use Spark’s MLlib to develop and use
an ML model in the cloud; (3) How to use Docker to create a container for your ML model to simplify
model deployment.

**Description:** You have to build a wine quality prediction ML model in Spark over AWS. The model must
be trained in parallel on multiple EC2 instances. Then, you need to save and load the model in an application
that will perform wine quality prediction; this application will run on one EC2 instance. The assignment
must be implemented in Java, Scala, or Python on Ubuntu Linux. The details of the assignment are
presented below:  
• Input for model training: we share 2 datasets with you for your ML model. Each row in a dataset is
for one specific wine, and it contains some physical parameters of the wine as well as a quality
score. Both datasets are available in Canvas, under Programming Assignment 2.
o TrainingDataset.csv: you will use this dataset to train the model in parallel on multiple EC2
instances.
o ValidationDataset.csv: you will use this dataset to validate the model and optimize its
performance (i.e., select the best values for the model parameters).  
• Input for prediction testing: Your prediction program should take the pathname to the test file as
a command line parameter. The pathname contains both the directory location and the filename
of the test file, e.g., /home/tom/pa2/ValidationDataset.csv. For your testing, you may use the
validation file ValidationDataset.csv provided with this project. In grading, we may test your
program using another file, which has a similar structure with the two datasets above, to test the
functionality and performance of your prediction application.  
• Output: The output of your application will be a measure of the prediction performance, specifically
the F1 score, which is available in MLlib.  
• Model Implementation: You have to develop a Spark application that uses MLlib to train for wine
quality prediction using the training dataset. You will use the validation dataset to check the
performance of your trained model and to potentially tune your ML model parameters for best
performance. You should start with a simple classification model (e.g., logistic regression) from
MLlib, but you can try multiple ML models to see which one leads to better performance. For
classification models, you can use 10 classes (the wine scores are from 1 to 10).
Docker container: You have to build a Docker container for your prediction application. In this
way, the prediction model can be quickly deployed across many different environments. You can
learn more about Docker in Module 13 (slides are already posted in Canvas). This is an example
of the Docker command, which will be run to test your Docker container:
sudo Docker run yourDockerImage testFilePath (you may use the '-v' parameter of the 'docker
run' command to map part of the host file system on the container’s file system)  
• The model training is done in parallel on 4 EC2 instances.  
• The prediction with or without Docker is done on a single EC2 instance.

**Submission:** You will submit in Canvas, under Programming Assignment 2, a text/Word/pdf file that
contains:
• A link to your code in GitHub. The code includes the code for parallel model training and the code
for the prediction application.
• A link to your container in Docker Hub.
This file must also describe step-by-step how to set-up the cloud environment and run the model training
and the application prediction. For the application prediction, you should provide instructions on how to
run it with and without Docker.

**Grading:**
- Parallel training implementation – 50 points
- Single machine prediction application – 25 points
- Docker container for prediction application – 25 points

# Attempted Solution and Step-by-step how to set-up the cloud environment and run the application.
Here's a step-by-step guide to creating a wine quality prediction ML model with Spark MLlib on AWS EC2 instances, replete with code and explanations.

## I. Set up AWS EC2 instances:
a. Launch four EC2 instances for model training and one EC2 instance for prediction.  
b. Install Java, Spark, and Docker on each instance.  

Install Java  
sudo apt update
sudo yum install java-11-amazon-corretto -y

Download and install Spark  
wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz  
tar xvf spark-3.1.2-bin-hadoop3.2.tgz  
sudo mv spark-3.1.2-bin-hadoop3.2 /opt/spark

Set environment variables  
echo "export SPARK_HOME=/opt/spark" >> ~/.bashrc  
echo "export PATH=$PATH:/opt/spark/bin" >> ~/.bashrc  
source ~/.bashrc  

## II. Prepare the dataset.
a. Upload the TrainingDataset.csv and ValidationDataset.csv files to the EC2.

## III. Create the Spark ML model training application in Python.
a. Create a new Python project using your choice IDE(Here I did in IntelliJ).
b. Create a new Python file in the Project called train_model.py for training.
b. Create a new Python file in the Project called predict_model.py for predicting.

## IV. Train the model on AWS EC2 instances:
a. Copy the Python file to all 4 EC2 instances.  
b. pip install depedencies.   
sudo yum install python3-pip -y
pip install numpy  
c. SSH into each EC2 instance and submit the Spark job using the spark-submit command:    
spark-submit 

## V. Setup Spark master slaves on EC2's
a. Run the master EC2.  
/opt/spark/sbin/start-master.sh   
b. Go to the Spark master web UI at http://<master-node-ip>:8080 ie here http://54.237.58.37:8080/
c. Run on worker EC2  
/opt/spark/sbin/start-worker.sh spark://54.237.58.37:7077  
d. check logs with  'tail -f /opt/spark/logs/spark-*.out'

## VI. Run train and predict program in EC2 with master-worker Spark
a. Train in master EC2 only and this dictributes to workers.
/opt/spark/bin/spark-submit \
--master spark://<master-ip>:7077 \
--deploy-mode client \
/path/to/your/training_script.py  
B. Run predict.
/opt/spark/bin/spark-submit \
--master spark://<master-ip>:7077 \
--deploy-mode client \
/path/to/your/predict_model.py /path/to/model /path/to/validation_dataset.csv

## VII. Run the prediction application on a single EC2 instance with Docker:
a. SSH into the EC2 instance.  
b. Install docker in EC2.  
sudo yum install -y docker   
c. start and enable docker  
sudo systemctl start docker  
sudo systemctl enable docker
d. Add user to have docker permissions  
sudo usermod -aG docker $USER  
e. Then exit and ssh back into ec2 for changes to take effect  
Can test docker with: docker run hello-world  
f. Then build image with   
docker build -t sandraroy37/wine-quality-prediction-prog-assignment2 .  
check image built by running 'docker images'
g. Push image to Dockerhub.  
docker login  
docker push sandraroy37/wine-quality-prediction-prog-assignment2:latest  
h. Pull the Docker image from the container registry and run the Docker container:  
docker run -it sandraroy37/wine-quality-prediction-prog-assignment2:latest ValidationDataset.csv

**Summary**  
The model training will be split across four EC2 servers, utilizing Spark's parallel processing capabilities.

The trained model will be saved and then used in the prediction application.

The prediction application will be distributed as a Docker container, making it simple to deploy and run in any environment.
The prediction application will produce the F1 score, which indicates the trained model's performance on the test dataset.
