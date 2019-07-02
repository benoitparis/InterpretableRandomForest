package paris.benoit

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.IndexToString

object Test {
  
  def main(args: Array[String]): Unit = {
    
    System.setProperty("hadoop.home.dir", "C:\\spark\\spark-2.4.2-bin-hadoop2.7");

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("InterpretableRF")
      .setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val customSchema = StructType(Array(
      StructField("PassengerId", LongType, true),
      StructField("Survived", StringType, true),
      StructField("Pclass", StringType, true),
      StructField("Name", StringType, true),
      StructField("Sex", StringType, true),
      StructField("Age", LongType, true),
      StructField("SibSp", LongType, true), //Number of Siblings/Spouses Aboard
      StructField("Parch", LongType, true), //Number of Parents/Children Aboard
      StructField("Ticket", StringType, true),
      StructField("Fare", DoubleType, true),
      StructField("Cabin", StringType, true),
      StructField("Embarked", StringType, true) //Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
    ))
    
    val dfCsvModelData = sqlContext.read
      .format("csv")
      .option("header", "true")
      .schema(customSchema)
      .load("data/titanic.csv")

    val dfCleanedModelData =
      dfCsvModelData
        .na.fill("")
        .na.fill(0)

    dfCleanedModelData.persist()

    val labelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("Survived_Indexed")
      .fit(dfCleanedModelData)

    val featurePclassIndexer = new StringIndexer()
      .setInputCol("Pclass")
      .setOutputCol("Pclass_Indexed")
      .fit(dfCleanedModelData)
    val featureSexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("Sex_Indexed")
      .fit(dfCleanedModelData)
    val featureEmbarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("Embarked_Indexed")
      .fit(dfCleanedModelData)

    val featureNamesArray =
      Array("PassengerId", "Pclass_Indexed", "Sex_Indexed", "Age", "SibSp", "Parch", "Fare", "Embarked_Indexed")

    val assembler = new VectorAssembler()
      .setInputCols(featureNamesArray)
      .setOutputCol("features")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("features_Indexed")
      .setMaxCategories(4)

    
    val Array(trainingData, testData) = dfCleanedModelData.randomSplit(Array(0.75, 0.25))

    val rf = new RandomForestClassifier()
      .setLabelCol("Survived_Indexed")
      .setFeaturesCol("features_Indexed")
      .setRawPredictionCol("rawPrediction")
      .setProbabilityCol("probabilityRF")
      .setImpurity("entropy")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(
        labelIndexer, featurePclassIndexer, featureSexIndexer, featureEmbarkedIndexer, assembler, featureIndexer, rf, labelConverter
    ))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val rfModel = model.stages(6).asInstanceOf[RandomForestClassificationModel]
    
    val contributions = InterpretableRF.interpret(rfModel, predictions, sc, featureNamesArray)
    
    contributions
      .map({ case (id, arr) => (id, "{" + arr.mkString(",") + "}")})
      .saveAsTextFile("out/contributions")

  }
  
}