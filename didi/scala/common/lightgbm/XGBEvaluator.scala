package com.cxyx.common.lightgbm

import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import scopt.OptionParser

import java.io.ByteArrayOutputStream

/**
  * Created by didi on 2020/10/13.
  */
object XGBEvaluator extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  override def run(): Unit = {
    log.info(s"load data from ${param.pathData}")
    var df=spark.read.parquet(param.pathData)
    df = mergeCategoryFeatures(df)

    if (param.modelBinary != null) {
      predict(df, param.modelBinary, "binary")
    } else {
      log.error("model path is not set!")
    }
  }

  def predict(df:DataFrame, modelPath:String, tag:String): Unit ={
    log.info(s"load model : ${modelPath}")
    val model = XGBoostClassificationModel.load(modelPath)
    log.info(s"loaded model, n(feature)=${model.numFeatures}")

    val rawResultDF = model.transform(Utils.caseVector(df, "features"))

    // 为了兼容cluster的logger
    val outCapture = new ByteArrayOutputStream
    Console.withOut(outCapture) {
      rawResultDF.limit(5).show()
    }
    log.info(s"lgboost raw result: \n${new String(outCapture.toByteArray)}")

    val res = Utils.customPrediction2(rawResultDF)
      .drop("features")
      .drop("rand") 
      .dropDuplicates(Array("user_id","goods_id","label"))
      .cache()
    log.info(s"res count : ${res.count()}")
    res.show(10)

    res.saveCompressedParquet(s"${param.pathOutput}_${tag}")
    log.info(s"successfully write parquet to ${param.pathOutput}_${tag}")
    evaluate(model, res)
  }

  def evaluate(model: XGBoostClassificationModel,
               df: DataFrame
              ) = {

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")

    println("----------- Evaluation on test data -----------")
    val testAUC = evaluator.evaluate(df)
    println(s"AUC on test data $testAUC")

  }

  def mergeCategoryFeatures(df: DataFrame): DataFrame = {
    import spark.implicits._
    var inputDF = df.withColumn("label", $"label".cast(IntegerType))
    // 离散列
    val cateCols = param.categoricalSlotNames
    log.info(s"categorical columns: ${cateCols}")
    // feature列
    var vecCols:Array[String] = null
    if(cateCols != "-"){
      // 原始列
      val conCols = Array("features")
      vecCols = cateCols.split(",") ++ conCols
      val allCols = Seq("user_id","goods_id") ++ vecCols.toSeq
      inputDF = inputDF.select("label", allCols:_*)
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
      // inputDF.saveCompressedParquet(s"${param.pathData}_cate.ckp")
    }
    inputDF
  }

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[String]("pathData").required().text("pathData").
      action((x, c) => c.copy(pathData = x))
    opt[String]("categoricalSlotNames").required().text("categoricalSlotNames").
      action((x, c) => c.copy(categoricalSlotNames = x))

    opt[String]("modelTxt").required().text("modelTxt").
      action((x, c) => c.copy(modelTxt = x))
    opt[String]("modelBinary").required().text("modelBinary").
      action((x, c) => c.copy(modelBinary = x))

    opt[String]("pathOutput").required().text("pathOutput").
      action((x, c) => c.copy(pathOutput = x))
  }

  case class Config(
                     pathData:String = null,
                     categoricalSlotNames:String = null,
                     pathOutput: String = null,
                     modelTxt: String = null,
                     modelBinary: String = null
                   )

}

