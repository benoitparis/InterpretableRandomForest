package org.apache.spark.ml.tree.inspectable

case class InspectableNodeInfo(
  val identity: String,
  val gain: Double,
  val prediction: Double, 
  val impurity: Double, 
  val isLeaf: Boolean, 
  val featureIndex: Int,
  val splitType: String,
  val threshold: Double,
  val leftCategories: Array[Double],
  val rightCategories: Array[Double],
  val numDescendants: Int,
  val subtreeDepth: Int,
  val isLeftChild: Boolean
) 