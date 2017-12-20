// Import SparkSession and Logistic Regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
// Create a Spark Session
var spark = SparkSession.builder().getOrCreate()
// Use Spark to read in the Advertising csv file.
var data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

val colnames = data.columns



val firstrow = data.head(1)(0)
println("\n")
for(n <- Range(1,colnames.length))
{
  println(colnames(n))
  println(firstrow(n))
  println("\n")
}


val timedata = data.withColumn("Hour",hour(data("Timestamp")))
var logregdataall = (timedata.select(data("Clicked on Ad").as("label"),
                    $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage",
                    $"Hour",$"Male"))
var logregdata = logregdataall.na.drop()
// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
val assembler = (new VectorAssembler()
                 .setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male"))
                 .setOutputCol("features"))
// Use randomSplit to create a train test split of 70/30
val Array(training,testing) = logregdata.randomSplit(Array(0.7,0.3), seed = 12345)

// Import Pipeline
import org.apache.spark.ml.Pipeline
// Create a new LogisticRegression object called lr
val lr = new LogisticRegression()
// Create a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler,lr))
// Fit the pipeline to training set.
val model = pipeline.fit(training)

// Get Results on Test Set with transform
val results = model.transform(testing)
println("\n")
////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////

// For Metrics and Evaluation import MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// Convert the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double,Double)].rdd
// Instantiate a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)
// Print out the Confusion matrix
println("Confusion Matrix:")
println(metrics.confusionMatrix)
println("\n")
println("Accuracy:")
println(metrics.accuracy)
