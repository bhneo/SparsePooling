package com.cxyx.common.alchemist

import com.alibaba.fastjson.JSONObject
import com.cxyx.common.alchemist.tools.{ToolsFile, ToolsHdfs}
import com.microsoft.ml.spark.lightgbm.{LightGBMClassifier, LightGBMRanker}
import jodd.util.collection.IntArrayList
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
//import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression}
import org.apache.spark.mllib.classification.{StreamingLogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.log4j.Logger
import org.apache.spark.sql.DataFrame

import java.nio.file.{Files, Paths}

/**
  * Created by zhaolei on 2021/04/26.
  */
object ModelUtils extends Constant {

  def makeModel(trainData:DataFrame,
                config:JSONObject,
                featureCols:Seq[String],
                initModelHDFS:String,
                pathLocal:String,
                modelHDFS:String,
                fMapHDFS:String,
                retrain:Boolean,
                log:Logger): Unit = {
    ToolsFile.mkDir(pathLocal)

    println(s"parsing model param: ")

    config.getString(MODEL_TYPE) match {
      case MODEL_TYPE_XGB =>
        makeXGB(trainData, config, pathLocal, modelHDFS, fMapHDFS, log, retrain)
      case MODEL_TYPE_LGB =>
        makeLGB(trainData, config, featureCols, pathLocal, modelHDFS, log, retrain)
      case MODEL_TYPE_LGB_RANKER =>
        makeLGBRanker(trainData, config, featureCols, pathLocal, modelHDFS, log, retrain)
      case MODEL_TYPE_LR =>
        makeLR(trainData, config, initModelHDFS, pathLocal, modelHDFS, log, retrain)
      case _ =>
        throw new RuntimeException(s"model not support!")
    }
  }

  def makeLR(trainData:DataFrame,
             json:JSONObject,
             initModelHDFS:String,
             pathLocal:String,
             pathHDFS:String,
             log:Logger,
             retrain:Boolean=false): Unit = {
    if(!retrain && ToolsHdfs.checkHdfsFileExists(log, s"${pathHDFS}/1")) {
      println(s"model exists: $pathHDFS/1")
    } else {
      println(
        s"""
           |params:
           |featuresCol : ${json.getString("featuresCol")}
           |labelCol : ${json.getString("labelCol")}
           |objective : ${json.getString("objective")}
           |iterations : ${json.getIntValue("iterations")}
          """.stripMargin)

      if(initModelHDFS != null && initModelHDFS.nonEmpty) {
        println(s"load model from ${initModelHDFS}")
        val initModel = LogisticRegressionModel.load(initModelHDFS)
        initModel
      }

      val lr = new StreamingLogisticRegressionWithSGD()
        .setStepSize()
        .setNumIterations()
        .setInitialWeights().trainOn(trainData.writeStream)


//      val lr = new LogisticRegression()
//        .setFeaturesCol(json.getString("featuresCol"))
//        .setLabelCol(json.getString("labelCol"))
//        .setMaxIter(json.getInteger("iterations"))
//        .setRegParam(json.getDoubleValue("regParam"))

//      println(s"model params as follow: \n ${lr.explainParams()}")

//      val model = lr.fit(trainData)

      /** ****************  model 1  ******************* */
      model.write.overwrite().save(s"$pathHDFS/1")

      /** ****************  model 2  ******************* */
      model.save(s"$pathHDFS/2")
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/1",
        s"$pathLocal"
      )
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/2",
        s"$pathLocal"
      )
      val numFeatures = model.numFeatures
      println(s"save model (numFeatures: $numFeatures) HDFS: ${pathHDFS} local: ${pathLocal}")
    }
  }

  def makeLGBRanker(trainData:DataFrame,
                    json:JSONObject,
                    featureCols:Seq[String],
                    pathLocal:String,
                    pathHDFS:String,
                    log:Logger,
                    retrain:Boolean=false): Unit = {
    if(!retrain && ToolsHdfs.checkHdfsFileExists(log, s"${pathHDFS}/binary/complexParams/lightGBMBooster")) {
      println(s"model exists: $pathHDFS/binary/complexParams/lightGBMBooster")
    } else {
      println(
        s"""
           |params:
           |featuresCol : ${json.getString("featuresCol")}
           |labelCol : ${json.getString("labelCol")}
           |objective : ${json.getString("objective")}
           |boostingType : ${json.getString("boostingType")}
           |group: ${json.getString("group")}
           |maxPosition: ${json.getIntValue("maxPosition")}
           |metric : ${json.getString("metric")}
           |learningRate : ${json.getDoubleValue("learningRate")}
           |maxDepth : ${json.getIntValue("maxDepth")}
           |trainingMetric : ${json.getBooleanValue("trainingMetric")}
           |barrierMode : ${json.getBooleanValue("barrierMode")}
           |leaves : ${json.getIntValue("leaves")}
           |iterations : ${json.getIntValue("iterations")}
           |featureFraction : ${json.getDoubleValue("featureFraction")}
           |baggingFraction : ${json.getDoubleValue("baggingFraction")}
           |baggingFreq : ${json.getIntValue("baggingFreq")}
           |maxBin : ${json.getIntValue("maxBin")}
           |isUnbalance : ${json.getBooleanValue("isUnbalance")}
           |minDataInLeaf : ${json.getIntValue("minDataInLeaf")}
           |minSumHessionInLeaf : ${json.getDoubleValue("minSumHessionInLeaf")}
           |lambda1 : ${json.getDoubleValue("lambda1")}
           |lambda2 : ${json.getDoubleValue("lambda2")}
           |categoricalSlotNames : ${json.getString("categoricalSlotNames")}
           |weightCol : ${json.getString("weightCol")}
           |workers : ${json.getIntValue("workers")}
          """.stripMargin)

      var lgb = new LightGBMRanker()
        .setFeaturesCol(json.getString("featuresCol"))
        .setLabelCol(json.getString("labelCol"))
        .setObjective(json.getString("objective"))
        .setBoostingType(json.getString("boostingType"))
        .setGroupCol(json.getString("group"))
        .setMaxPosition(json.getIntValue("maxPosition"))
        .setMetric(json.getString("metric"))
        .setNumTasks(json.getIntValue("workers"))
        .setIsProvideTrainingMetric(json.getBooleanValue("trainingMetric"))
        .setUseBarrierExecutionMode(json.getBooleanValue("barrierMode"))
        .setMaxDepth(json.getIntValue("maxDepth"))
        .setNumLeaves(json.getIntValue("leaves"))
        .setNumIterations(json.getIntValue("iterations"))
        .setLearningRate(json.getDoubleValue("learningRate"))
        .setLambdaL1(json.getDoubleValue("lambda1"))
        .setLambdaL2(json.getDoubleValue("lambda2"))
        .setFeatureFraction(json.getDoubleValue("featureFraction"))
        .setBaggingFraction(json.getDoubleValue("baggingFraction"))
        .setBaggingFreq(json.getIntValue("baggingFreq"))
        .setMaxBin(json.getIntValue("maxBin"))
        .setMinDataInLeaf(json.getIntValue("minDataInLeaf"))
        .setMinSumHessianInLeaf(json.getDoubleValue("minSumHessionInLeaf"))
        .setVerbosity(1)

      if (json.getString("weightCol").nonEmpty && json.getString("weightCol") != "-"){
        println(s"set weight col: ${json.getString("weightCol")}")
        lgb = lgb.setWeightCol(json.getString("weightCol"))
      }

      val categoricalSlotNames = json.getString("categoricalSlotNames")
      if (categoricalSlotNames != "-") {
        val categoricalSlotIndexes = new IntArrayList()
        categoricalSlotNames.split(",").foreach(x=>{
          if (featureCols.contains(x.trim)) {
            categoricalSlotIndexes.add(featureCols.indexOf(x))
          } else {
            println(s"$x is not in features from FMap!")
          }
        })

        println(s"set indexes ${categoricalSlotIndexes.toArray.mkString(",")} to categorical features!")
        lgb = lgb.setCategoricalSlotIndexes(categoricalSlotIndexes.toArray)
      }

      println(s"model params as follow: \n ${lgb.explainParams()}")
      val model = lgb.fit(trainData)

      /** ****************  model binary  ******************* */
      model.write.overwrite().save(s"$pathHDFS/binary")

      /** ****************  model txt  ******************* */
      model.saveNativeModel(s"$pathHDFS/txt", overwrite = true)
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/binary",
        s"$pathLocal"
      )
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/txt",
        s"$pathLocal"
      )
      val numFeatures = model.numFeatures
      println(s"save model (numFeatures: $numFeatures) HDFS: ${pathHDFS} local: ${pathLocal}")
    }
  }

  def makeLGB(trainData:DataFrame,
              json:JSONObject,
              featureCols:Seq[String],
              pathLocal:String,
              pathHDFS:String,
              log:Logger,
              retrain:Boolean=false): Unit = {
    if(!retrain && ToolsHdfs.checkHdfsFileExists(log, s"${pathHDFS}/binary/complexParams/lightGBMBooster")) {
      println(s"model exists: $pathHDFS/binary/complexParams/lightGBMBooster")
    } else {
      println(
        s"""
           |params:
           |featuresCol : ${json.getString("featuresCol")}
           |labelCol : ${json.getString("labelCol")}
           |objective : ${json.getString("objective")}
           |boostingType : ${json.getString("boostingType")}
           |metric : ${json.getString("metric")}
           |learningRate : ${json.getDoubleValue("learningRate")}
           |maxDepth : ${json.getIntValue("maxDepth")}
           |trainingMetric : ${json.getBooleanValue("trainingMetric")}
           |barrierMode : ${json.getBooleanValue("barrierMode")}
           |leaves : ${json.getIntValue("leaves")}
           |iterations : ${json.getIntValue("iterations")}
           |featureFraction : ${json.getDoubleValue("featureFraction")}
           |baggingFraction : ${json.getDoubleValue("baggingFraction")}
           |baggingFreq : ${json.getIntValue("baggingFreq")}
           |maxBin : ${json.getIntValue("maxBin")}
           |isUnbalance : ${json.getBooleanValue("isUnbalance")}
           |minDataInLeaf : ${json.getIntValue("minDataInLeaf")}
           |minSumHessionInLeaf : ${json.getDoubleValue("minSumHessionInLeaf")}
           |lambda1 : ${json.getDoubleValue("lambda1")}
           |lambda2 : ${json.getDoubleValue("lambda2")}
           |categoricalSlotNames : ${json.getString("categoricalSlotNames")}
           |weightCol : ${json.getString("weightCol")}
           |earlyStoppingRounds: ${json.getIntValue("earlyStoppingRounds")}
           |workers : ${json.getIntValue("workers")}
          """.stripMargin)

      var lgb = new LightGBMClassifier()
        .setFeaturesCol(json.getString("featuresCol"))
        .setLabelCol(json.getString("labelCol"))
        .setObjective(json.getString("objective"))
        .setBoostingType(json.getString("boostingType"))
        .setMetric(json.getString("metric"))
        .setNumTasks(json.getIntValue("workers"))
        .setIsProvideTrainingMetric(json.getBooleanValue("trainingMetric"))
        .setUseBarrierExecutionMode(json.getBooleanValue("barrierMode"))
        .setIsUnbalance(json.getBooleanValue("isUnbalance"))
        .setMaxDepth(json.getIntValue("maxDepth"))
        .setNumLeaves(json.getIntValue("leaves"))
        .setNumIterations(json.getIntValue("iterations"))
        .setLearningRate(json.getDoubleValue("learningRate"))
        .setLambdaL1(json.getDoubleValue("lambda1"))
        .setLambdaL2(json.getDoubleValue("lambda2"))
        .setFeatureFraction(json.getDoubleValue("featureFraction"))
        .setBaggingFraction(json.getDoubleValue("baggingFraction"))
        .setBaggingFreq(json.getIntValue("baggingFreq"))
        .setMaxBin(json.getIntValue("maxBin"))
        .setMinDataInLeaf(json.getIntValue("minDataInLeaf"))
        .setMinSumHessianInLeaf(json.getDoubleValue("minSumHessionInLeaf"))
        .setEarlyStoppingRound(json.getIntValue("earlyStoppingRounds"))
        .setVerbosity(1)

      if (json.getString("weightCol").nonEmpty && json.getString("weightCol") != "-"){
        println(s"set weight col: ${json.getString("weightCol")}")
        lgb = lgb.setWeightCol(json.getString("weightCol"))
      }

      val categoricalSlotNames = json.getString("categoricalSlotNames")
      if (categoricalSlotNames != "-") {
        val categoricalSlotIndexes = new IntArrayList()
        categoricalSlotNames.split(",").foreach(x=>{
          if (featureCols.contains(x)) {
            categoricalSlotIndexes.add(featureCols.indexOf(x))
          } else {
            println(s"$x is not in features from FMap!")
          }
        })

        println(s"set indexes ${categoricalSlotIndexes.toArray.mkString(",")} to categorical features!")
        lgb = lgb.setCategoricalSlotIndexes(categoricalSlotIndexes.toArray)
      }
      println(s"model params as follow: \n ${lgb.explainParams()}")
      val model = lgb.fit(trainData)

      /** ****************  model binary  ******************* */
      model.write.overwrite().save(s"$pathHDFS/binary")

      /** ****************  model txt  ******************* */
      model.saveNativeModel(s"$pathHDFS/txt", overwrite = true)
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/binary",
        s"$pathLocal"
      )
      ToolsHdfs.getHdfsFile(
        log,
        s"$pathHDFS/txt",
        s"$pathLocal"
      )
      val numFeatures = model.numFeatures
      println(s"save model (numFeatures: $numFeatures) HDFS: ${pathHDFS} local: ${pathLocal}")
    }
  }

  def makeXGB(trainData:DataFrame,
              json:JSONObject,
              pathLocal:String,
              pathHDFS:String,
              fMapHDFS:String,
              log:Logger,
              retrain:Boolean=false): Unit = {
    if(!retrain && ToolsHdfs.checkHdfsFileExists(log, s"${pathHDFS}/data/XGBoostClassificationModel")) {
      println(s"model exists: $pathHDFS/data/XGBoostClassificationModel")
    } else {
      println(
        s"""
           |params:
           |featuresCol : ${json.getString("featuresCol")}
           |labelCol : ${json.getString("labelCol")}
           |objective : ${json.getString("objective")}
           |metric : ${json.getString("metric")}
           |trainTestRatio : ${json.getString("trainTestRatio")}
           |learningRate : ${json.getDoubleValue("learningRate")}
           |maxDepth : ${json.getIntValue("maxDepth")}
           |iterations : ${json.getIntValue("iterations")}
           |featureFraction : ${json.getDoubleValue("featureFraction")}
           |baggingFraction : ${json.getDoubleValue("baggingFraction")}
           |treeMethod : ${json.getString("treeMethod")}
           |scalePosWeight : ${json.getDoubleValue("scalePosWeight")}
           |autoScalePosRatio : ${json.getIntValue("autoScalePosRatio")}
           |minChildWeight : ${json.getIntValue("minChildWeight")}
           |alpha : ${json.getDoubleValue("alpha")}
           |lambda : ${json.getDoubleValue("lambda")}
           |gamma : ${json.getDoubleValue("gamma")}
           |categoricalSlotNames : ${json.getString("categoricalSlotNames")}
           |weightCol : ${json.getString("weightCol")}
           |workers : ${json.getIntValue("workers")}
          """.stripMargin)

      val scalePosWeight = getScalePosWeight(trainData,
        json.getIntValue("autoScalePosRatio"),
        json.getDoubleValue("scalePosWeight"))

      var xgb = new XGBoostClassifier()
        .setFeaturesCol(json.getString("featuresCol"))
        .setLabelCol(json.getString("labelCol"))
        .setObjective(json.getString("objective"))
        .setEvalMetric(json.getString("metric"))
        .setTrainTestRatio(json.getDoubleValue("trainTestRatio"))
        .setNumWorkers(json.getIntValue("workers"))
        .setMaxDepth(json.getIntValue("maxDepth"))
        .setNumRound(json.getIntValue("iterations"))
        .setEta(json.getDoubleValue("learningRate"))
        .setAlpha(json.getDoubleValue("alpha"))
        .setLambda(json.getDoubleValue("lambda"))
        .setGamma(json.getDoubleValue("gamma"))
        .setColsampleBytree(json.getDoubleValue("featureFraction"))
        .setSubsample(json.getDoubleValue("baggingFraction"))
        .setMinChildWeight(json.getIntValue("minChildWeight"))
        .setTreeMethod(json.getString("treeMethod"))
        .setTrainTestRatio(0.9)
        .setScalePosWeight(scalePosWeight)

      if (json.getString("weightCol").nonEmpty && json.getString("weightCol") != "-"){
        println(s"set weight col: ${json.getString("weightCol")}")
        xgb = xgb.setWeightCol(json.getString("weightCol"))
      }

      println(s"model params as follow: \n ${xgb.explainParams()}")
      val model = xgb.fit(trainData)

      /******************  model写hdfs  ********************/
      model.write.overwrite().save(pathHDFS)

      /******************  model写本地  ********************/
      model.nativeBooster.saveModel(s"${pathLocal}/local_xgb_model")
      ToolsHdfs.putHdfsFile(
        log,
        s"$pathHDFS/local_xgb_model",
        s"$pathLocal/local_xgb_model"
      )

      ToolsHdfs.getHdfsFile(log, fMapHDFS, s"$pathLocal/fmap")
      saveTreeToLocalFile(s"$pathLocal/local_xgb_tree", model,s"$pathLocal/fmap")
      println(s"save tree: $pathLocal")

      val numFeatures = model.numFeatures
      println(s"save model (numFeatures: $numFeatures) HDFS: ${pathHDFS} local: ${pathLocal}")

      println(model.summary.toString())
    }
  }

  private def saveTreeToLocalFile(filePath: String, model: XGBoostClassificationModel, fmap: String): Unit = {
    val content = model.nativeBooster.getModelDump(fmap, true).mkString("")
    try {
      Files.delete(Paths.get(filePath))
    } catch {
      case _: Exception =>
    }
    Files.write(Paths.get(filePath), content.getBytes)
  }

  def getScalePosWeight(data: DataFrame, autoScalePosRatio: Int, scalePosWeight:Double): Double = {
    // 如果设置了scalePosWeight，直接使用指定的值
    if (scalePosWeight > 1.0) {
      println(s"直接使用指定的 scalePosWeight: ${scalePosWeight}")
      return scalePosWeight
    }

    // 看是否启用了自动计算scalePosWeight
    if (autoScalePosRatio < 1) {
      println("输入的 autoScalePosRatio < 1.0，设置 scalePosWeight=1.0")
      1.0
    } else {
      // 按输入的ratio，根据正负例的差值，进行调整
      val neg = data.filter("label=0.0").count()
      val pos = data.filter("label=1.0").count()
      val rawRatio = neg.toFloat / pos.toFloat
      val scalePosWeight = if (pos < neg) {
        rawRatio / autoScalePosRatio
      } else {
        println(s"正例 > 负例，scalePosWeight 强制设置为：1.0")
        1.0
      }
      println(s"自动调整 scalePosWeight: Pos=${pos} Neg=${neg} Neg/Pos=${rawRatio} autoScalePosRatio=${autoScalePosRatio} scalePosWeight=${scalePosWeight}")

      scalePosWeight
    }
  }

}
