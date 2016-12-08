package hm.posaune.deepLearningInJava;

import hm.posaune.deepLearningInJava.util.GaussianDistribution;

import java.util.Random;

public class Application {

    public static void main(String[] args) {
        final int train_N = 1000;
        final int test_N = 200;
        final int nIn = 2;

        double[][] train_X = new double[train_N][nIn];
        int[] train_T = new int[train_N];

        double[][] test_X = new double[test_N][nIn];
        int[] test_T = new int[test_N];
        int[] prediction_T = new int[test_N];

        final int epochs = 2000;
        final double learningRate = 1.;

        final Random random = new Random();

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

        for(int i =0; i < test_N; i++){
            prediction_T[i] = classifier.predict(test_X[i]);
        }
    }
}
