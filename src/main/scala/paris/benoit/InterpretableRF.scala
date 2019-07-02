package paris.benoit

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier }
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer, VectorIndexer }
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD

object InterpretableRF {
  
  
  def sumArrays(a: Array[Double], b: Array[Double]) = {
    a.zip(b).map({ case (x, y) => x + y })
  }

  def substractArrays(a: Array[Double], b: Array[Double]) = {
    a.zip(b).map({ case (x, y) => x - y })
  }

  def getOneHot(index: Double, size: Int) = {
    val labelArrayPosition = Array.fill(size) { 0.0 }.zipWithIndex.map(_._2.toDouble)
    labelArrayPosition
      .map(_ == index)
      .map(_ match {
        case true => 1.0
        case false => 0.0
      })
  }
  
  
  def getNodeInfoWithParentSplitList(inspectableRootNode: org.apache.spark.ml.tree.inspectable.InspectableNode) = {
    val inspectableNodeInfoList = inspectableRootNode.asInspectableNodeInfoList(true)
    val inspectableNodeInfoMap =
      inspectableNodeInfoList
        .map(node => node.identity -> node)
        .toMap
    val inspectableNodeInfoListWithParentSplit =
      inspectableNodeInfoList
        .map(nodeInfo =>
          (
            nodeInfo.identity.replaceAll("(([^\\.])+|(\\.([^\\.])))(\\.F)?$", ""),
            nodeInfo))
        .map(_ match { case (k, v) => (v, inspectableNodeInfoMap.get(k)) })
        .map(_ match {
          case (enfant, None) => (enfant, ("", -1.0: Double, "", 0.0, Array(): Array[Double]))
          case (enfant, Some(parent)) => (
            enfant,
            if (enfant.isLeftChild) {
              if (parent.splitType.equals("ContinuousSplit")) {
                (parent.splitType, parent.featureIndex: Double, "<=", parent.threshold, Array(): Array[Double])
              } else {
                (parent.splitType, parent.featureIndex: Double, "", 0.0, parent.leftCategories)
              }
            } else {
              if (parent.splitType.equals("ContinuousSplit")) {
                (parent.splitType, parent.featureIndex: Double, ">", parent.threshold, Array(): Array[Double])
              } else {
                (parent.splitType, parent.featureIndex: Double, "", 0.0, parent.rightCategories)
              }
            })
        })
    inspectableNodeInfoListWithParentSplit
  }

  def interpret(rfModel: RandomForestClassificationModel, predictions: Dataset[Row], sc: SparkContext, featureNamesArray: Array[String]): RDD[(Long, Array[Double])] = {

    // Forest with parent info
    val inspectableForestIndexed =
      rfModel
        .trees
        .zipWithIndex
        .map(treeIndex => (treeIndex._2, org.apache.spark.ml.tree.inspectable.InspectableNode.fromClassicalNode(treeIndex._1.rootNode, "tree-" + treeIndex._2 + ".root")));

    val predictionsWithForestPaths =
      predictions
        .rdd
        .map(point => {
          val forestPaths =
            inspectableForestIndexed
              .map(inspectableRootNode =>
                (inspectableRootNode._1, inspectableRootNode._2.inspectablePredictImpl(point.getAs[org.apache.spark.ml.linalg.Vector]("features"))))
          (point, forestPaths)
        });

    // TODO 2 magic: nombre classes à chopper
    val labelArrayPosition = Array.fill(2) { 0.0 }.zipWithIndex.map(_._2.toDouble)
    def getOneHot2(index: Double) = {
      labelArrayPosition
        .map(_ == index)
        .map(_ match {
          case true => 1.0
          case false => 0.0
        })
    }

    // TODO attention c'est le predictedLabel et pas le prediction.. faudra overhaul
    val treePathsWithTestNumInstanceStats =
      predictionsWithForestPaths
        .map(rowPath => ((rowPath._1.getAs[Long]("PassengerId"), rowPath._1.getAs[Double]  ("Survived_Indexed")), rowPath._2))
        .flatMap(dataTrees => dataTrees._2.map(tree => (dataTrees._1, tree._2)))
        .flatMap(dataTree => dataTree._2.map(node => (dataTree._1, node)))
        .map(dataNode => ((dataNode._2), getOneHot2(dataNode._1._2))) // treepath, onehot
        .reduceByKey((a, b) => sumArrays(a, b))
        .map({
          case (node, sum) => (
            node,
            (
              sum.reduce(_ + _),
              sum.map(_ / sum.reduce(_ + _))))
        })

    val treePathsWithTestNumInstanceStatsAndLocalDelta =
      treePathsWithTestNumInstanceStats
        .map(nodeInfo =>
          (
            nodeInfo._1.replaceAll("(([^\\.])+|(\\.([^\\.])))(\\.F)?$", ""),
            nodeInfo))
        .leftOuterJoin(treePathsWithTestNumInstanceStats)
        .map({
          case (identity, (enfant, parent)) =>
            (enfant._1,
              (
                enfant._2,
                parent match {
                  case None => (enfant._2._1, enfant._2._2)
                  case Some(parent) => (enfant._2._1 - parent._1, substractArrays(enfant._2._2, parent._2))
                }))
        })

    val nodesWithParentDecision =
      sc.parallelize(
        inspectableForestIndexed
          .map({
            case (index, inspectableRootNode) =>
              (index, getNodeInfoWithParentSplitList(inspectableRootNode))
          })
          .flatMap(_._2)
          .map({ case (node, parentDecision) => (node.identity, (node, parentDecision)) }))

    // TODO mapper reste
    val nodesWithTestNumInstanceStats =
      nodesWithParentDecision
        .join(treePathsWithTestNumInstanceStatsAndLocalDelta)

    val contributions =
      predictionsWithForestPaths
        .map(rowPath => (rowPath._1.getAs[Long]("PassengerId"), rowPath._2))
        .flatMap(dataTrees => dataTrees._2.map(tree => (dataTrees._1, tree._2)))
        .flatMap(dataTree => dataTree._2.map(node => (node, dataTree._1))) // treepath, siret
        .join(nodesWithTestNumInstanceStats)
        .map({
          case (treepath, (key, ((nodeInfo, parentSplit), (stats, statsDeltaParent)))) =>
            (key, getOneHot(parentSplit._2, featureNamesArray.length).map(-_ * statsDeltaParent._2(0)))
        })
        .reduceByKey(sumArrays _)
   
   (contributions)

//    val predictionsWithForestPathsWithFeatureLift =
//      predictionsWithForestPaths
//        .map(rowPath => (rowPath._1.getAs[Long]("PassengerId"), rowPath))
//        .join(featureLift)
//        .map({
//          case (key, (data, featureLift)) =>
//            (data, featureLift)
//        })
//
//    predictionsWithForestPathsWithFeatureLift
//      .saveAsTextFile("out/predictions4.csv")
      

//    val predictionsWithForestPathsWithFeatureLiftExport =
//      predictionsWithForestPathsWithFeatureLift
//        .map(item => ((item._1._1, "{" + item._1._2.map(_._2.mkString(",")).mkString(",") + "}"), "{" + item._2.mkString(",") + "}"))

  }
}