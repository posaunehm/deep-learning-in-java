package hm.posaune.deepLearningInJava;

import hm.posaune.deepLearningInJava.util.GaussianDistribution;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by posaunehm on 2016/12/16.
 */
public class SingleLayerNN {
    private int train_N;
    private int test_N;
    private int nIn;
    final Random random = new Random();


    public void executeTest(int train_n, int test_n, int nIn) throws IOException {

        this.train_N = train_n;
        this.test_N = test_n;
        this.nIn = nIn;

        double[][] train_X = new double[train_N][nIn];
        int[] train_T = new int[train_N];

        double[][] test_X = new double[test_N][nIn];
        int[] test_T = new int[test_N];
        int[] prediction_T = new int[test_N];

        final int epochs = 2000;
        final double learningRate = 1.;


        final GaussianDistribution g1 = new GaussianDistribution(-2.0,1,random);
        final GaussianDistribution g2 = new GaussianDistribution(2.0,1,random);

        for(int i = 0; i < train_N / 2; i++){
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = 1;
        }

        for(int i = train_N / 2; i < train_N; i++){
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = -1;
        }

        for(int i = 0; i < test_N / 2; i++){
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = 1;
        }

        for(int i = test_N / 2; i < test_N; i++){
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = -1;
        }

        Perception classifier = new Perception(nIn);

        int epoch = 0;

        while (true){
            int classified = 0;

            for(int i =0; i < train_N; i++){
                classified += classifier.train(train_X[i], train_T[i], learningRate);
            }

            if(classified == train_N){
                break;
            }

            epoch++;
            if(epoch >= epochs){
                break;
            }
        }

        System.out.println(String.format("%d data used for training.", epoch));

        for(int i =0; i < test_N; i++){
            prediction_T[i] = classifier.predict(test_X[i]);
        }

        XYSeriesCollection data = new XYSeriesCollection();

        XYSeries series_1 = new XYSeries("系列1");
        XYSeries series_2 = new XYSeries("系列2");

        data.addSeries(series_1);
        data.addSeries(series_2);

        for(int i =0; i < test_N / 2; i++){
            series_1.add(test_X[i][0], test_X[i][1]);
        }

        for(int i =test_N / 2; i < test_N; i++){
            series_2.add(test_X[i][0], test_X[i][1]);
        }



        JFreeChart chart_1 = ChartFactory.createScatterPlot(
                "実データ",
                "X",
                "Y",
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false);

        ChartUtilities.saveChartAsPNG(
                new File("./dataSource.png"),
                chart_1, 300, 300);

        XYSeriesCollection predictionData = new XYSeriesCollection();

        XYSeries series_prediction_1 = new XYSeries("系列1");
        XYSeries series_prediction_2 = new XYSeries("系列2");

        predictionData.addSeries(series_prediction_1);
        predictionData.addSeries(series_prediction_2);

        for(int i =0; i < test_N; i++){
            if(prediction_T[i] == 1) {
                series_prediction_1.add(test_X[i][0], test_X[i][1]);
            }else{
                series_prediction_2.add(test_X[i][0], test_X[i][1]);
            }
        }

        JFreeChart chart_2 = ChartFactory.createScatterPlot(
                "予測値",
                "X",
                "Y",
                predictionData,
                PlotOrientation.VERTICAL,
                true,
                false,
                false);

        ChartUtilities.saveChartAsPNG(
                new File("./prediction.png"), chart_2, 300, 300);
    }
}
