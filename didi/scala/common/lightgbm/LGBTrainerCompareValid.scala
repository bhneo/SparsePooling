package com.cxyx.common.lightgbm

import java.io.PrintWriter

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

import scala.io.Source

/**
  * Created by didi on 2020/10/13.
  */
object LGBTrainerCompareValid extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  var paramsConfig = ParamsConfig()

  override def run(): Unit = {
    getFmap
    getConfig
    getTrain
  }

  def getFmap = {
    log.info(s"get fmap: ${param.fmap_hdfs}")
    ToolsHdfs.getHdfsFile(log, param.fmap_hdfs, s"${param.fmap}/data")

    ToolsFile.mkDir(param.path_local_model)
  }

  def getConfig = {

    log.info(s"parsing : ${param.config}")
    val json = JSON.parseObject(param.config)
    val keyset = json.keySet()

    log.info(s"keyset : ${keyset.toArray().mkString(",")}")

    if (keyset.contains("learningRate")) paramsConfig.learningRate = json.getDoubleValue("learningRate")
    if (keyset.contains("maxDepth")) paramsConfig.maxDepth = json.getIntValue("maxDepth")
    if (keyset.contains("trainingMetric")) paramsConfig.trainingMetric = json.getBooleanValue("trainingMetric")
    if (keyset.contains("barrierMode")) paramsConfig.barrierMode = json.getBooleanValue("barrierMode")
    if (keyset.contains("leaves")) paramsConfig.leaves = json.getIntValue("leaves")
    if (keyset.contains("iterations")) paramsConfig.iterations = json.getIntValue("iterations")
    if (keyset.contains("featureFraction")) paramsConfig.featureFraction = json.getDoubleValue("featureFraction")
    if (keyset.contains("baggingFraction")) paramsConfig.baggingFraction = json.getDoubleValue("baggingFraction")
    if (keyset.contains("baggingFreq")) paramsConfig.baggingFreq = json.getIntValue("baggingFreq")
    if (keyset.contains("maxBin")) paramsConfig.maxBin = json.getIntValue("maxBin")
    if (keyset.contains("categoricalSlotNames")) paramsConfig.categoricalSlotNames = json.getString("categoricalSlotNames")
    if (keyset.contains("isUnbalance")) paramsConfig.isUnbalance = json.getBooleanValue("isUnbalance")
    if (keyset.contains("minDataInLeaf")) paramsConfig.minDataInLeaf = json.getIntValue("minDataInLeaf")
    if (keyset.contains("minSumHessionInLeaf")) paramsConfig.minSumHessionInLeaf = json.getDoubleValue("minSumHessionInLeaf")
    if (keyset.contains("lambda1")) paramsConfig.lambda1 = json.getDoubleValue("lambda1")
    if (keyset.contains("lambda2")) paramsConfig.lambda2 = json.getDoubleValue("lambda2")
    if (keyset.contains("weightCol")) paramsConfig.weightCol = json.getString("weightCol")
    if (keyset.contains("workers")) paramsConfig.workers = json.getIntValue("workers")

    log.info(
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

  def getTrain = {
    val (trainDF, testDF) = prepare
    val model = train(trainDF)
    evaluate(model, trainDF, testDF)
  }

  def train(trainDF: DataFrame): LightGBMClassificationModel = {
    var lgboost = new LightGBMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setObjective("binary")
      .setBoostingType("gbdt")
      .setMetric("auc")
      .setIsProvideTrainingMetric(paramsConfig.trainingMetric)
      .setUseBarrierExecutionMode(paramsConfig.barrierMode)
      .setIsUnbalance(paramsConfig.isUnbalance)
      .setMaxDepth(paramsConfig.maxDepth)
      .setNumLeaves(paramsConfig.leaves)
      .setNumIterations(paramsConfig.iterations)
      .setLearningRate(paramsConfig.learningRate)
      .setLambdaL1(paramsConfig.lambda1)
      .setLambdaL2(paramsConfig.lambda2)
      .setFeatureFraction(paramsConfig.featureFraction)
      .setBaggingFraction(paramsConfig.baggingFraction)
      .setBaggingFreq(paramsConfig.baggingFreq)
      .setMaxBin(paramsConfig.maxBin)
      .setMinDataInLeaf(paramsConfig.minDataInLeaf)
      .setMinSumHessianInLeaf(paramsConfig.minSumHessionInLeaf)
      .setVerbosity(2)

//    if (paramsConfig.categoricalSlotNames != "-") {
//      val categoricalSlotIndexes = new IntArrayList()
//      for(i <- 0 to paramsConfig.categoricalSlotNames.split(",").length) {
//        categoricalSlotIndexes.add(i)
//      }
//      lgboost=lgboost.setCategoricalSlotIndexes(categoricalSlotIndexes.toArray)
//    }

    log.info(s"lgboost params as follow: \n ${lgboost.explainParams()}")
    val model = lgboost.fit(trainDF)

    /******************  model binary  ********************/
    model.write.overwrite().save(param.output_model)

    /******************  model txt  ********************/
    model.saveNativeModel(s"${param.output_model}",true)
    ToolsHdfs.getHdfsFile(
      log,
      s"${param.output_model}",
      s"${param.path_local_model}"
    )

    /******************  fscore  ********************/
    for( x <- List("split","gain") ){
      val pathFeatureScore = s"${param.path_local_model}/fscore_${x}"
      val names = Source.fromFile(s"${param.fmap}/data").getLines().toSeq
        .map(x=>{
          x.split("\\t")(1)
        }).toArray
      var cols:Array[String] = null
      if(paramsConfig.categoricalSlotNames != "-"){
        cols = Array.concat(paramsConfig.categoricalSlotNames.split(","), names)
      } else {
        cols = names
      }
      val fvs = cols.zip(model.getFeatureImportances(s"${x}")).toMap
                  .toSeq.sortBy(_._2).reverse.map(x=>s"${x._1}\t${x._2}").mkString("\n")

      val file = new PrintWriter(pathFeatureScore)
      file.println(fvs)
      file.close()
      log.info(s"save split feature score to : $pathFeatureScore")

      log.info(Yellow("[DEBUG] ") + "--------------- topK feature score")
      Source.fromFile(pathFeatureScore, "utf-8").getLines().toSeq
        .slice(0, 100).zipWithIndex
        .foreach(x => println(s"${x._2}\t${x._1}"))
    }

    val numFeatures = model.numFeatures
    log.info(s"save model (numFeatures: $numFeatures) HDFS: ${param.output_model} local: ${param.path_local_model}")

    model
  }

  def evaluate(model: LightGBMClassificationModel,
               trainDF: DataFrame,
               testDF: DataFrame
              ) = {

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    // 为了方便复制，这些参数不要用log形式
    println("----------- Evaluation on training data -----------")
    val predictTrainDF = Utils.customPrediction(model.transform(trainDF))
    val trainAUC = evaluator.evaluate(predictTrainDF)
    println(s"AUC on train data $trainAUC")

    println("----------- Evaluation on test data -----------")
    val predictTestDF = Utils.customPrediction(model.transform(testDF))
    val testAUC = evaluator.evaluate(predictTestDF)
    println(s"AUC on test data $testAUC")

    // 输出预测结果的十分位数
    println("----------- 整个训练集合的分布 （train + test） -----------")
    Utils.showProbPercentiles(predictTrainDF.union(predictTestDF))

  }


  def getDataSet: (DataFrame, DataFrame) = {

    val trainCache = param.path_train_pre + "/train"
    val trainData = if(ToolsHdfs.checkHdfsFileExists(log, trainCache)) {
      log.info(s"${trainCache} exists, read it directly!")
      spark.read.parquet(trainCache)
    } else {
      log.info(s"${trainCache} not exists, read from original files!")
      val originData = ToolsHdfs.readRangeParquet(spark,log,param.path_train_pre,param.train_start,param.train_end)

      originData.saveCompressedParquet(trainCache)
      spark.read.parquet(trainCache)
    }

    log.info(s"train count : ${trainData.count()}")
    trainData.groupBy("label").count().show()

    val testData = spark.read.parquet(s"${param.path_train_pre}/${param.test_date}")
    log.info(s"test count : ${testData.count()}")
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
    log.info(s"n(train)=${n_train} n(test)=${n_test} n(class)=${n_class} n(feature)=${n_feature}")

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
    log.info(s"categorical columns: ${cateCols}")
    // feature列
    var vecCols:Array[String] = null
    if(cateCols != "-"){
      // 原始列
      val conCols = Array("features")
      vecCols = cateCols.split(",") ++ conCols
      inputDF = inputDF.select("label", vecCols: _*)
      if(cateCols != "-"){
        log.info(s"cast categorical columns to double...")
        cateCols.split(",").foreach(col => {
          inputDF = inputDF.withColumn(col, $"$col".cast(DoubleType))
        })
      }

      val assembler = new VectorAssembler().setInputCols(vecCols).setOutputCol("merged_features")
      inputDF = assembler.transform(inputDF).drop("features").withColumnRenamed("merged_features", "features")
      log.info(s"assembled dataframe:")
      inputDF.show(10)
    }
    inputDF
  }

  case class ParamsConfig(
                           var learningRate: Double = 0.1,
                           var trainingMetric: Boolean = false,
                           var barrierMode: Boolean = false,
                           var maxDepth: Int = -1,
                           var leaves: Int = 31,
                           var iterations: Int = 100,
                           var featureFraction: Double = 1.0,
                           var baggingFraction: Double = 1.0,
                           var baggingFreq: Int = 0,
                           var maxBin: Int = 255,
                           var categoricalSlotNames: String = "",
                           var isUnbalance: Boolean = false,
                           var minDataInLeaf: Int = 20,
                           var minSumHessionInLeaf: Double = 1e-3,
                           var lambda1: Double = 0.0,
                           var lambda2: Double = 0.0,
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
    opt[String]("output_model").required().text("output_model").
      action((x, c) => c.copy(output_model = x))
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
                     output_model: String = null
                   )
}
