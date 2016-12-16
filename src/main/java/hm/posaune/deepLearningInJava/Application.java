package hm.posaune.deepLearningInJava;

import java.io.IOException;

public class Application {

    public static void main(String[] args) throws IOException {
        SingleLayerNN singleLayerNN = new SingleLayerNN();

        singleLayerNN.executeTest(1000, 200, 2);

    }
}
