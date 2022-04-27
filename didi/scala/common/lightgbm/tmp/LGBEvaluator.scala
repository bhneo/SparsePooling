package com.cxyx.common.lightgbm.tmp

import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import com.microsoft.ml.spark.lightgbm.LightGBMRankerModel
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.DataFrame
import scopt.OptionParser

/**
  * Created by didi on 2020/10/13.
  */
object LGBEvaluator extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  override def run(): Unit = {
    log.info(s"load data from ${param.pathData}")
    val df = prepare
    predict(df, param.modelPath, "")
  }


  def prepare: (DataFrame) = {

    import spark.implicits._

    val originData = spark.read.parquet(param.pathData)
    originData.cache()
    log.info(s"originData count : ${originData.count()}")

    var df: DataFrame = null
    df = originData
      .map(r => {
        val features = r.getAs[Vector]("features").toArray
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          r.getAs[String]("request_id"),
          new DenseVector(features)
        )
      }).toDF("user_id", "goods_id", "label", "request_id", "features")
    df.cache()
    log.info(s"all set count : ${df.count()}")
    df
  }


  def predict(df: DataFrame, modelPath: String, tag: String): Unit = {
    log.info(s"load model : ${modelPath}")
    val model = LightGBMRankerModel.loadNativeModelFromFile(modelPath)
    log.info(s"loaded model, n(feature)=${model.numFeatures}")

    val rawResultDF = model.transform(df)
    rawResultDF.saveCompressedParquet(s"${param.pathOutput}")

    // 为了兼容cluster的logger
    //    val outCapture = new ByteArrayOutputStream
    //    Console.withOut(outCapture) {
    //      rawResultDF.limit(5).show()
    //    }
    //    log.info(s"lgboost raw result: \n${new String(outCapture.toByteArray)}")
    //
    //    val res = Utils.customPrediction2(rawResultDF)
    //      .drop("features")
    //      .drop("rand")
    //      .dropDuplicates(Array("user_id", "goods_id", "label"))
    //      .cache()
    //    log.info(s"res count : ${res.count()}")
    //    res.show(10)
    //
    //    res.saveCompressedParquet(s"${param.pathOutput}_${tag}")
    //    log.info(s"successfully write parquet to ${param.pathOutput}_${tag}")
    //    evaluate(model, res)
  }

  //  def evaluate(model: LightGBMClassificationModel,
  //               df: DataFrame
  //              ) = {
  //
  //    val evaluator = new BinaryClassificationEvaluator()
  //      .setLabelCol("label")
  //      .setRawPredictionCol("probability")
  //      .setMetricName("areaUnderROC")
  //
  //    println("----------- Evaluation on test data -----------")
  //    val testAUC = evaluator.evaluate(df)
  //    println(s"AUC on test data $testAUC")
  //
  //  }

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[String]("pathData").required().text("pathData").
      action((x, c) => c.copy(pathData = x))

    opt[String]("pathOutput").required().text("pathOutput").
      action((x, c) => c.copy(pathOutput = x))

    opt[String]("modelPath").required().text("modelPath").
      action((x, c) => c.copy(modelPath = x))
  }

  case class Config(
                     pathData: String = null,
                     pathOutput: String = null,
                     modelPath: String = null
                   )

}