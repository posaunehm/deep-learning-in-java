package hm.posaune.deepLearningInJava;

/**
 * Created by posaunehm on 2016/12/08.
 */
public class Perception {
    private final int nIn;
    private final double[] w;

    public Perception(int nIn) {
        this.nIn = nIn;

        w = new double[nIn];
    }

    public int train(double[] data, int result, double learningRate) {
        int classfied = 0;
        double predictionValue = 0;

        for (int i = 0; i < nIn; i++) {
            predictionValue += w[i] * data[i];
        }

        if ((predictionValue >= 0 && result == 1)
                || predictionValue < 0 && result == -1) {
            classfied = 1;
        } else {
            for (int i = 0; i < nIn; i++) {
                w[i] += learningRate * data[i] * result;
            }
        }

        return classfied;
    }

    public int predict(double[] data) {
        double predictionValue = 0;

        for (int i = 0; i < nIn; i++) {
            predictionValue += w[i] * data[i];
        }

        if(predictionValue >= 0){
            return 1;
        }else{
            return -1;
        }
    }
}
