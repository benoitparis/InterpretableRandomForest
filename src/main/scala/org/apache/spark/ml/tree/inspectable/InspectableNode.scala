/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.inspectable

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.{ImpurityStats, InformationGainStats => OldInformationGainStats, Node => OldNode, Predict => OldPredict}

import org.apache.spark.ml.tree._

/**
 * Decision tree node interface.
 */
sealed abstract class InspectableNode extends Serializable {
  
  def identity: String

  // TODO: Add aggregate stats (once available).  This will happen after we move the DecisionTree
  //       code into the new API and deprecate the old API.  SPARK-3727

  /** Prediction a leaf node makes, or which an internal node would make if it were a leaf node */
  def prediction: Double

  /** Impurity measure at this node (for training data) */
  def impurity: Double

  /**
   * Statistics aggregated from training data at this node, used to compute prediction, impurity,
   * and probabilities.
   * For classification, the array of class counts must be normalized to a probability distribution.
   */
  private[ml] def impurityStats: ImpurityCalculator

  /** Recursive prediction helper method */
  private[ml] def predictImpl(features: Vector): InspectableLeafNode
  
  /** Version custom de predictImpl qui ramène une liste des noeuds. On donne pas la prédiction, de toute façon ça a déjà été fait dans le pipeline. et on peut tjs join sur le leaf */
  
  def inspectablePredictImpl(features: Vector): collection.immutable.List[String]
  
  
  /** Une description de l'arbre sour forme de liste de ses noeuds
    TODO: pourquoi copier des truc quand tu peux donner le noeud entier?
      -> pour sortir du type system?
    */
  def asInspectableNodeInfoList(isLeftChild: Boolean): collection.immutable.List[InspectableNodeInfo]

  
  /**
   * Get the number of nodes in tree below this node, including leaf nodes.
   * E.g., if this is a leaf, returns 0.  If both children are leaves, returns 2.
   */
  private[tree] def numDescendants: Int

  /**
   * Recursive print function.
   * @param indentFactor  The number of spaces to add to each level of indentation.
   */
  private[tree] def subtreeToString(indentFactor: Int = 0): String

  /**
   * Get depth of tree from this node.
   * E.g.: Depth 0 means this is a leaf node.  Depth 1 means 1 internal and 2 leaf nodes.
   */
  private[tree] def subtreeDepth: Int

  /**
   * Create a copy of this node in the old Node format, recursively creating child nodes as needed.
   * @param id  Node ID using old format IDs
   */
  private[ml] def toOld(id: Int): OldNode

  /**
   * Trace down the tree, and return the largest feature index used in any split.
   * @return  Max feature index used in a split, or -1 if there are no splits (single leaf node).
   */
  private[ml] def maxSplitFeatureIndex(): Int

  /** Returns a deep copy of the subtree rooted at this node. */
  private[tree] def deepCopy(): InspectableNode
  
}

object InspectableNode {

  def fromClassicalNode(classicalNode: org.apache.spark.ml.tree.Node, identity: String) : InspectableNode = {

    if ("org.apache.spark.ml.tree.InternalNode" == classicalNode.getClass.getName) {
      val classicalNodeTyped = classicalNode.asInstanceOf[org.apache.spark.ml.tree.InternalNode];
      
      // TODO move this elsewhere
      val fieldPrediction = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("prediction")
      fieldPrediction.setAccessible(true)
      val newPrediction = fieldPrediction.get(classicalNodeTyped).asInstanceOf[Double]
      
      val fieldImpurity = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("impurity")
      fieldImpurity.setAccessible(true)
      val newImpurity = fieldImpurity.get(classicalNodeTyped).asInstanceOf[Double]
      
      val fieldGain = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("gain")
      fieldGain.setAccessible(true)
      val newGain = fieldGain.get(classicalNodeTyped).asInstanceOf[Double]
      
      val fieldLeftChild = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("leftChild")
      fieldLeftChild.setAccessible(true)
      val newLeftChild = fromClassicalNode(fieldLeftChild.get(classicalNodeTyped).asInstanceOf[org.apache.spark.ml.tree.Node], identity + ".L")
      
      val fieldRightChild = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("rightChild")
      fieldRightChild.setAccessible(true)
      val newRightChild = fromClassicalNode(fieldRightChild.get(classicalNodeTyped).asInstanceOf[org.apache.spark.ml.tree.Node], identity + ".R")
      
      val fieldSplit = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("split")
      fieldSplit.setAccessible(true)
      val newSplit = fieldSplit.get(classicalNodeTyped).asInstanceOf[Split]
      
      val fieldImpurityStats = classOf[org.apache.spark.ml.tree.InternalNode].getDeclaredField("impurityStats")
      fieldImpurityStats.setAccessible(true)
      val newImpurityStats = fieldImpurityStats.get(classicalNodeTyped).asInstanceOf[ImpurityCalculator]
      
      new InspectableInternalNode(identity, newPrediction, newImpurity, newGain, newLeftChild, newRightChild, newSplit, newImpurityStats)
    } else {
      val classicalNodeTyped = classicalNode.asInstanceOf[org.apache.spark.ml.tree.LeafNode];
      
      val fieldPrediction = classOf[org.apache.spark.ml.tree.LeafNode].getDeclaredField("prediction")
      fieldPrediction.setAccessible(true)
      val newPrediction = fieldPrediction.get(classicalNodeTyped).asInstanceOf[Double]
      
      val fieldImpurity = classOf[org.apache.spark.ml.tree.LeafNode].getDeclaredField("impurity")
      fieldImpurity.setAccessible(true)
      val newImpurity = fieldImpurity.get(classicalNodeTyped).asInstanceOf[Double]
      
      val fieldImpurityStats = classOf[org.apache.spark.ml.tree.LeafNode].getDeclaredField("impurityStats")
      fieldImpurityStats.setAccessible(true)
      val newImpurityStats = fieldImpurityStats.get(classicalNodeTyped).asInstanceOf[ImpurityCalculator]
      
      new InspectableLeafNode(identity + ".F", newPrediction, newImpurity, newImpurityStats)
      
    }
  }
}

/**
 * Decision tree leaf node.
 * @param prediction  Prediction this node makes
 * @param impurity  Impurity measure at this node (for training data)
 */
class InspectableLeafNode (
    override val identity: String,
    override val prediction: Double,
    override val impurity: Double,
    override private[ml] val impurityStats: ImpurityCalculator) extends InspectableNode {

  override def toString: String =
    s"LeafNode(prediction = $prediction, impurity = $impurity)"
  
  override def asInspectableNodeInfoList(isLeftChild: Boolean): collection.immutable.List[InspectableNodeInfo] = {
    List(InspectableNodeInfo(
      identity,
      gain = -1.0,
      prediction, 
      impurity, 
      isLeaf = true, 
      featureIndex = -1,
      splitType = "", //getClass.getSimpleName
      threshold = 0,
      leftCategories = null,
      rightCategories = null,
      numDescendants,
      subtreeDepth,
      isLeftChild
    ))
  }
  
  override private[ml] def predictImpl(features: Vector): InspectableLeafNode = this
  
  override def inspectablePredictImpl(features: Vector): collection.immutable.List[String] = {
    List(identity)
  }


  override private[tree] def numDescendants: Int = 0

  override private[tree] def subtreeToString(indentFactor: Int = 0): String = {
    val prefix: String = " " * indentFactor
    prefix + s"Predict: $prediction\n"
  }

  override private[tree] def subtreeDepth: Int = 0

  override private[ml] def toOld(id: Int): OldNode = {
    new OldNode(id, new OldPredict(prediction, prob = impurityStats.prob(prediction)),
      impurity, isLeaf = true, None, None, None, None)
  }

  override private[ml] def maxSplitFeatureIndex(): Int = -1

  override private[tree] def deepCopy(): InspectableNode = {
    new InspectableLeafNode(identity, prediction, impurity, impurityStats)
  }
}

/**
 * Internal Decision Tree node.
 * @param prediction  Prediction this node would make if it were a leaf node
 * @param impurity  Impurity measure at this node (for training data)
 * @param gain Information gain value. Values less than 0 indicate missing values;
 *             this quirk will be removed with future updates.
 * @param leftChild  Left-hand child node
 * @param rightChild  Right-hand child node
 * @param split  Information about the test used to split to the left or right child.
 */
class InspectableInternalNode (
    val identity: String,
    override val prediction: Double,
    override val impurity: Double,
    val gain: Double,
    val leftChild: InspectableNode,
    val rightChild: InspectableNode,
    val split: Split,
    override private[ml] val impurityStats: ImpurityCalculator) extends InspectableNode {

  // Note to developers: The constructor argument impurityStats should be reconsidered before we
  //                     make the constructor public.  We may be able to improve the representation.

  override def toString: String = {
    s"InspectableInternalNode(prediction = $prediction, impurity = $impurity, split = $split)"
  }
 
  
  override def asInspectableNodeInfoList(isLeftChild: Boolean): collection.immutable.List[InspectableNodeInfo] = {
    
    val leftChildNodeInfoList = leftChild.asInspectableNodeInfoList(true)
    val rightChildNodeInfoList = rightChild.asInspectableNodeInfoList(false)
      
    var threshold: Double = 0.0
    var leftCategories: Array[Double] = Array()
    var rightCategories: Array[Double] = Array()
    split match {
      case contSplit: ContinuousSplit =>
        threshold = contSplit.threshold
      case catSplit: CategoricalSplit =>
        leftCategories = catSplit.leftCategories
        rightCategories = catSplit.rightCategories
    }
    
    new InspectableNodeInfo(
      identity,
      gain,
      prediction, 
      impurity, 
      isLeaf = false, 
      featureIndex = split.featureIndex,
      splitType = split.getClass.getSimpleName,
      threshold,
      leftCategories,
      rightCategories,
      numDescendants,
      subtreeDepth,
      isLeftChild
    ).asInstanceOf[org.apache.spark.ml.tree.inspectable.InspectableNodeInfo] :: leftChildNodeInfoList.asInstanceOf[collection.immutable.List[org.apache.spark.ml.tree.inspectable.InspectableNodeInfo]] ::: rightChildNodeInfoList.asInstanceOf[collection.immutable.List[org.apache.spark.ml.tree.inspectable.InspectableNodeInfo]]
  }
  
  override private[ml] def predictImpl(features: Vector): InspectableLeafNode = {
    if (split.shouldGoLeft(features)) {
      leftChild.predictImpl(features)
    } else {
      rightChild.predictImpl(features)
    }
  }
  
  
  override def inspectablePredictImpl(features: Vector): collection.immutable.List[String] = {
    if (split.shouldGoLeft(features)) {
      identity :: leftChild.inspectablePredictImpl(features)
    } else {
      identity :: rightChild.inspectablePredictImpl(features)
    }
  }

  override private[tree] def numDescendants: Int = {
    2 + leftChild.numDescendants + rightChild.numDescendants
  }

  override private[tree] def subtreeToString(indentFactor: Int = 0): String = {
    val prefix: String = " " * indentFactor
    prefix + s"If (${InspectableInternalNode.splitToString(split, left = true)})\n" +
      leftChild.subtreeToString(indentFactor + 1) +
      prefix + s"Else (${InspectableInternalNode.splitToString(split, left = false)})\n" +
      rightChild.subtreeToString(indentFactor + 1)
  }

  override private[tree] def subtreeDepth: Int = {
    1 + math.max(leftChild.subtreeDepth, rightChild.subtreeDepth)
  }

  override private[ml] def toOld(id: Int): OldNode = {
    assert(id.toLong * 2 < Int.MaxValue, "Decision Tree could not be converted from new to old API"
      + " since the old API does not support deep trees.")
    new OldNode(id, new OldPredict(prediction, prob = impurityStats.prob(prediction)), impurity,
      isLeaf = false, Some(split.toOld), Some(leftChild.toOld(OldNode.leftChildIndex(id))),
      Some(rightChild.toOld(OldNode.rightChildIndex(id))),
      Some(new OldInformationGainStats(gain, impurity, leftChild.impurity, rightChild.impurity,
        new OldPredict(leftChild.prediction, prob = 0.0),
        new OldPredict(rightChild.prediction, prob = 0.0))))
  }

  override private[ml] def maxSplitFeatureIndex(): Int = {
    math.max(split.featureIndex,
      math.max(leftChild.maxSplitFeatureIndex(), rightChild.maxSplitFeatureIndex()))
  }

  override private[tree] def deepCopy(): InspectableNode = {
    new InspectableInternalNode(identity, prediction, impurity, gain, leftChild.deepCopy(), rightChild.deepCopy(),
      split, impurityStats)
  }
}

private object InspectableInternalNode {

  /**
   * Helper method for [[Node.subtreeToString()]].
   * @param split  Split to print
   * @param left  Indicates whether this is the part of the split going to the left,
   *              or that going to the right.
   */
  private def splitToString(split: Split, left: Boolean): String = {
    val featureStr = s"feature ${split.featureIndex}"
    split match {
      case contSplit: ContinuousSplit =>
        if (left) {
          s"$featureStr <= ${contSplit.threshold}"
        } else {
          s"$featureStr > ${contSplit.threshold}"
        }
      case catSplit: CategoricalSplit =>
        val categoriesStr = catSplit.leftCategories.mkString("{", ",", "}")
        if (left) {
          s"$featureStr in $categoriesStr"
        } else {
          s"$featureStr not in $categoriesStr"
        }
    }
  }
}
