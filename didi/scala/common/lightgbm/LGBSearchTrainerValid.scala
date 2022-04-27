package com.cxyx.common.lightgbm

import com.alibaba.fastjson.JSON
import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import com.microsoft.ml.spark.lightgbm.{LightGBMClassificationModel, LightGBMClassifier}
import jodd.util.collection.IntArrayList
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import scopt.OptionParser
import com.cxyx.common.alchemist.tools.{ToolsFile, ToolsHdfs}

/**
  * Created by didi on 2020/10/13.
  */
object LGBSearchTrainerValid extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  var paramsConfig = ParamsConfig()

  override def run(): Unit = {
    getFmap
    getConfig
    search
  }

  def getFmap = {
    println(s"get fmap: ${param.fmap_hdfs}")
    ToolsHdfs.getHdfsFile(log, param.fmap_hdfs, s"${param.fmap}/data")

    ToolsFile.mkDir(param.path_local_model)
  }

  def getConfig = {

    println(s"parsing : ${param.config}")
    val json = JSON.parseObject(param.config)
    val keyset = json.keySet()

    println(s"keyset : ${keyset.toArray().mkString(",")}")

    if (keyset.contains("learningRate")) paramsConfig.learningRate = json.getString("learningRate")
    if (keyset.contains("maxDepth")) paramsConfig.maxDepth = json.getString("maxDepth")
    if (keyset.contains("trainingMetric")) paramsConfig.trainingMetric = json.getBooleanValue("trainingMetric")
    if (keyset.contains("barrierMode")) paramsConfig.barrierMode = json.getBooleanValue("barrierMode")
    if (keyset.contains("leaves")) paramsConfig.leaves = json.getString("leaves")
    if (keyset.contains("iterations")) paramsConfig.iterations = json.getString("iterations")
    if (keyset.contains("featureFraction")) paramsConfig.featureFraction = json.getString("featureFraction")
    if (keyset.contains("baggingFraction")) paramsConfig.baggingFraction = json.getString("baggingFraction")
    if (keyset.contains("baggingFreq")) paramsConfig.baggingFreq = json.getString("baggingFreq")
    if (keyset.contains("maxBin")) paramsConfig.maxBin = json.getString("maxBin")
    if (keyset.contains("categoricalSlotNames")) paramsConfig.categoricalSlotNames = json.getString("categoricalSlotNames")
    if (keyset.contains("isUnbalance")) paramsConfig.isUnbalance = json.getBooleanValue("isUnbalance")
    if (keyset.contains("minDataInLeaf")) paramsConfig.minDataInLeaf = json.getString("minDataInLeaf")
    if (keyset.contains("minSumHessionInLeaf")) paramsConfig.minSumHessionInLeaf = json.getString("minSumHessionInLeaf")
    if (keyset.contains("lambda1")) paramsConfig.lambda1 = json.getString("lambda1")
    if (keyset.contains("lambda2")) paramsConfig.lambda2 = json.getString("lambda2")
    if (keyset.contains("weightCol")) paramsConfig.weightCol = json.getString("weightCol")
    if (keyset.contains("workers")) paramsConfig.workers = json.getIntValue("workers")

    println(
      s"""
         |learningRate : ${paramsConfig.learningRate}
         |maxDepth : ${paramsConfig.maxDepth}
         |trainingMetric : ${paramsConfig.trainingMetric}
         |barrierMode : ${paramsConfig.barrierMode}
         |leaves : ${paramsConfig.leaves}
         |iterations : ${paramsConfig.iterations}
         |featureFraction : ${paramsConfig.featureFraction}
         |baggingFraction : ${paramsConfig.baggingFraction}
         |baggingFreq : ${paramsConfig.baggingFreq}
         |maxBin : ${paramsConfig.maxBin}
         |categoricalSlotNames : ${paramsConfig.categoricalSlotNames}
         |isUnbalance : ${paramsConfig.isUnbalance}
         |minSumHessionInLeaf : ${paramsConfig.minSumHessionInLeaf}
         |lambda1 : ${paramsConfig.lambda1}
         |lambda2 : ${paramsConfig.lambda2}
         |weightCol : ${paramsConfig.weightCol}
         |workers : ${paramsConfig.workers}
      """.stripMargin)
  }

  def search(): Unit = {
    val (trainDF, testDF) = prepare
    var lgboost = new LightGBMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setObjective("binary")
      .setBoostingType("gbdt")
      .setMetric("auc")
      .setNumTasks(paramsConfig.workers)
      .setIsProvideTrainingMetric(paramsConfig.trainingMetric)
      .setUseBarrierExecutionMode(paramsConfig.barrierMode)
      .setIsUnbalance(paramsConfig.isUnbalance)
      .setVerbosity(1)

//    if (paramsConfig.categoricalSlotNames != "-") {
//      val categoricalSlotIndexes = new IntArrayList()
//      for(i <- 0 to paramsConfig.categoricalSlotNames.split(",").length) {
//        categoricalSlotIndexes.add(i)
//      }
//      lgboost=lgboost.setCategoricalSlotIndexes(categoricalSlotIndexes.toArray)
//    }

    var paramGrid = Array[String]()
    var boosters = Array[LightGBMClassifier]()
    for (maxDepth <- paramsConfig.maxDepth.split(s",")) {
      for (leaves <- paramsConfig.leaves.split(s",")) {
        for (iteration <- paramsConfig.iterations.split(s",")) {
          for (learningRate <- paramsConfig.learningRate.split(s",")) {
            for (lambda1 <- paramsConfig.lambda1.split(s",")) {
              for (lambda2 <- paramsConfig.lambda2.split(s",")) {
                for (featureFraction <- paramsConfig.featureFraction.split(s",")) {
                  for (baggingFraction <- paramsConfig.baggingFraction.split(s",")) {
                    for (baggingFreq <- paramsConfig.baggingFreq.split(s",")) {
                      for (maxBin <- paramsConfig.maxBin.split(s",")) {
                        for (minSumHessionInLeaf <- paramsConfig.minSumHessionInLeaf.split(s",")) {
                          for (minDataInLeaf <- paramsConfig.minDataInLeaf.split(s",")) {
                            paramGrid = paramGrid :+
                              s"""
                                 |learningRate : ${learningRate.toFloat}
                                 |maxDepth : ${maxDepth.toInt}
                                 |leaves : ${leaves.toInt}
                                 |iterations : ${iteration.toInt}
                                 |featureFraction : ${featureFraction.toFloat}
                                 |baggingFraction : ${baggingFraction.toFloat}
                                 |baggingFreq : ${baggingFreq.toInt}
                                 |maxBin : ${maxBin.toInt}
                                 |minSumHessionInLeaf : ${minSumHessionInLeaf.toFloat}
                                 |minDataInLeaf : ${minDataInLeaf.toInt}
                                 |lambda1 : ${lambda1.toFloat}
                                 |lambda2 : ${lambda2.toFloat}
                          """.stripMargin
                            boosters = boosters :+ lgboost
                              .setMaxDepth(maxDepth.toInt)
                              .setNumLeaves(leaves.toInt)
                              .setNumIterations(iteration.toInt)
                              .setLearningRate(learningRate.toFloat)
                              .setLambdaL1(lambda1.toFloat)
                              .setLambdaL2(lambda2.toFloat)
                              .setFeatureFraction(featureFraction.toFloat)
                              .setBaggingFraction(baggingFraction.toFloat)
                              .setBaggingFreq(baggingFreq.toInt)
                              .setMaxBin(maxBin.toInt)
                              .setMinSumHessianInLeaf(minSumHessionInLeaf.toFloat)
                              .setMinDataInLeaf(minDataInLeaf.toInt)
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    val evaluator = new BinaryClassificationEvaluator()
          .setLabelCol("label")
          .setRawPredictionCol("rawPrediction")
          .setMetricName("areaUnderROC")
    var metrics = Array[(Float, Float)]()
    var models = Array[LightGBMClassificationModel]()

    println(s"\nstart search ${boosters.length} models in total\n")
    for( i <- boosters.indices){
      val model = boosters(i).fit(trainDF)
      models = models :+ model
      val result = (evaluator.evaluate(Utils.customPrediction(model.transform(trainDF))).toFloat,
        evaluator.evaluate(Utils.customPrediction(model.transform(testDF))).toFloat)
      metrics = metrics :+ result
      println(s"lgboost params as follow: \n ${paramGrid(i)}")
      println(s"train AUC: ${result._1}, test AUC: ${result._2}")
      println(s"\n")
    }

    val bestMetric = metrics.maxBy(_._2)
    val bestParam = paramGrid(metrics.indexOf(bestMetric))
    println(s"best param: \n ${bestParam}")
    println(s"best result: \n")
    println(s"train AUC: ${bestMetric._1}, test AUC: ${bestMetric._2}")
    println(s"\n")

    val bestModel = models(metrics.indexOf(bestMetric))
    /******************  model binary  ********************/
    bestModel.write.overwrite().save(param.best_model)

    /******************  model txt  ********************/
    bestModel.saveNativeModel(s"${param.best_model}",true)
    ToolsHdfs.getHdfsFile(
      log,
      s"${param.best_model}",
      s"${param.path_local_model}"
    )

    val numFeatures = bestModel.numFeatures
    println(s"save best model (numFeatures: $numFeatures) HDFS: ${param.best_model} local: ${param.path_local_model}")
  }


  def getDataSet: (DataFrame, DataFrame) = {

    val trainCache = param.path_train_pre + "/train"
    val trainData = if(ToolsHdfs.checkHdfsFileExists(log, trainCache)) {
      println(s"${trainCache} exists, read it directly!")
      spark.read.parquet(trainCache)
    } else {
      println(s"${trainCache} not exists, read from original files!")
      val originData = ToolsHdfs.readRangeParquet(spark,log,param.path_train_pre,param.train_start,param.train_end)

      originData.saveCompressedParquet(trainCache)
      spark.read.parquet(trainCache)
    }

    println(s"train count : ${trainData.count()}")
    trainData.groupBy("label").count().show()

    val testData = spark.read.parquet(s"${param.path_train_pre}/${param.test_date}")
    println(s"test count : ${testData.count()}")
    testData.groupBy("label").count().show()

    (trainData, testData)
  }

  def prepare: (DataFrame, DataFrame) = {

    import spark.implicits._

    val (trainDFOri, testDFOri) = getDataSet

    var trainDF = trainDFOri
      .map(r=>{
        val features = r.getAs[Vector]("features").toArray
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          new DenseVector(features)
        )
      }).toDF("user_id","goods_id","label","features")
    var testDF = testDFOri
      .map(r=>{
        val features = r.getAs[Vector]("features").toArray
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          new DenseVector(features)
        )
      }).toDF("user_id","goods_id","label","features")

    val n_train = trainDF.count()
    if (n_train < 1)
      log.error("Training data is empty!")

    val n_test = testDF.count()

    val n_class = trainDF.dropDuplicates("label").count().toInt
    val n_feature = trainDF.first().getAs[Vector]("features").size
    println(s"n(train)=${n_train} n(test)=${n_test} n(class)=${n_class} n(feature)=${n_feature}")

    trainDF = mergeCategoryFeatures(trainDF.join(trainDFOri.drop("features"),Seq("user_id","goods_id","label"),"left"))
    testDF = mergeCategoryFeatures(testDF.join(testDFOri.drop("features"),Seq("user_id","goods_id","label"),"left"))

    trainDF = trainDF.repartition(paramsConfig.workers).cache()
    testDF = testDF.cache()

    (trainDF, testDF)
  }

  def mergeCategoryFeatures(df: DataFrame): DataFrame = {
    import spark.implicits._
    var inputDF = df.withColumn("label", $"label".cast(IntegerType))
    // 离散列
    val cateCols = paramsConfig.categoricalSlotNames
    println(s"categorical columns: ${cateCols}")
    // feature列
    var vecCols:Array[String] = null
    if(cateCols != "-"){
      // 原始列
      val conCols = Array("features")
      vecCols = cateCols.split(",") ++ conCols
      inputDF = inputDF.select("label", vecCols: _*)
      if(cateCols != "-"){
        println(s"cast categorical columns to double...")
        cateCols.split(",").foreach(col => {
          inputDF = inputDF.withColumn(col, $"$col".cast(DoubleType))
        })
      }

      val assembler = new VectorAssembler().setInputCols(vecCols).setOutputCol("merged_features")
      inputDF = assembler.transform(inputDF).drop("features").withColumnRenamed("merged_features", "features")
      println(s"assembled dataframe:")
      inputDF.show(10)
    }
    inputDF
  }

  case class ParamsConfig(
                           var learningRate: String = "0.1",
                           var trainingMetric: Boolean = false,
                           var barrierMode: Boolean = false,
                           var maxDepth: String = "-1",
                           var leaves: String = "31",
                           var iterations: String = "100",
                           var featureFraction: String = "1.0",
                           var baggingFraction: String = "1.0",
                           var baggingFreq: String = "0",
                           var maxBin: String = "255",
                           var categoricalSlotNames: String = "",
                           var isUnbalance: Boolean = false,
                           var minSumHessionInLeaf: String = "0.001",
                           var minDataInLeaf: String = "20",
                           var lambda1: String = "0.0",
                           var lambda2: String = "0.0",
                           var weightCol:String = "",
                           var workers: Int = 500
                         )

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[String]("train_start").text("train_start").
      action((x, c) => c.copy(train_start = x))
    opt[String]("train_end").required().text("train_end").
      action((x, c) => c.copy(train_end = x))
    opt[String]("test_date").required().text("test_date").
      action((x, c) => c.copy(test_date = x))
    opt[String]("path_train_pre").required().text("path_train_pre").
      action((x, c) => c.copy(path_train_pre = x))
    opt[String]("config").required().text("config").
      action((x, c) => c.copy(config = x))
    opt[String]("fmap_hdfs").required().text("fmap_hdfs").
      action((x, c) => c.copy(fmap_hdfs = x))
    opt[String]("fmap").required().text("fmap").
      action((x, c) => c.copy(fmap = x))
    opt[String]("path_local_model").required().text("path_local_model").
      action((x, c) => c.copy(path_local_model = x))
    opt[String]("best_model").required().text("best_model").
      action((x, c) => c.copy(best_model = x))
  }

  case class Config(
                     train_start: String = null,
                     train_end: String = null,
                     test_date: String = null,
                     path_train_pre: String = null,
                     config: String = null,
                     fmap_hdfs: String = null,
                     fmap: String = null,
                     path_local_model: String = null,
                     best_model: String = null
                   )
}
