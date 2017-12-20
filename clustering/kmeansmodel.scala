// Import SparkSession
import org.apache.spark.sql.SparkSession
// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark Session Instance
val spark = SparkSession.builder().getOrCreate()
// Import Kmeans clustering Algorithm
import org.apache.spark.ml.clustering.KMeans
// Load the Wholesale Customers Data
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")

data.printSchema

val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
// Remember there is no Label column
val assembler = (new VectorAssembler()
                .setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))
                .setOutputCol("features"))

// Use the assembler object to transform the feature_data
// Call this new data training_data
val training_data = assembler.transform(feature_data).select("features")
// Create a Kmeans Model with K=3
val kmeans = new KMeans().setK(9).setSeed(1L)
// Fit that model to the training_data
val model = kmeans.fit(training_data)
// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSE = model.computeCost(training_data)
println(s"Within Set Sum Of Squared Errors: $WSSE")

// Shows the result.
println("Cluster Centers")
model.clusterCenters.foreach(println)
