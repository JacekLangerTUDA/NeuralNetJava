package neural.network;

import com.google.gson.Gson;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Main Entry point.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 01.11.2022</p>
 */
public class Main {


  /**
   * The base path to the weights file.
   */
  public static final String BASEPATH = "resources/weights/w-%s.json";
  public static final String TRAIN_IMAGE_PATH = "resources/data/train-images.idx3-ubyte";
  public static final String TRAIN_LABEL_PATH = "resources/data/train-labels.idx1-ubyte";
  public static final String CONTROL_IMAGE_PATH = "resources/data/t10k-images.idx3-ubyte";
  public static final String CONTROL_LABEL_PATH = "resources/data/t10k-labels.idx1-ubyte";

  public static void main(String[] args) {

    System.out.printf("""
                                            
                      \t\t\tNN    NN  NN    NN  jjjjjj
                      \t\t\tNNNN  NN  NNNN  NN      jj
                      \t\t\tNN  NNNN  NN  NNNN  jj  jj
                      \t\t\tNN    NN  NN    NN   jjj
                                            
                      \t\t\tNeuronales Netz in Java V1
                                            
                      """);

    try {

      boolean cleanRun = false;
      double incrementLearning = 0.0;
      int hidden = 2;
      short gen = 1;
      double lr = 0.03;
      for (int i = 0; i < args.length; i++) {
        var s = args[i];
        switch (s) {
          case "-c" -> cleanRun = true;
          case "-g" -> gen = Short.parseShort(args[i + 1]);
          case "-h" -> help(0);
          case "-l", "--learning-rate" -> lr = (double) (Double.parseDouble(args[i + 1]) / 100.);
          case "-i", "--increase-learning-rate" ->
              incrementLearning = (1 - lr) * 100 / (gen * 60_000);
        }
      }

      var weights = fetchWeigths(2, cleanRun);
      double[][] fst = weights.get(0);
      double[][] sec = weights.get(1);
      double[][] thrd = weights.get(2);
      var net = new neural.network.NeuralNet(fst, sec, thrd);
      net.train(gen, lr, incrementLearning);
      assessRandom();

      // save the weights to file.
      saveWeights(net.getFstHiddenLayerWeigths(),
                  net.getSecondHiddenLayerWeigths(),
                  net.getFinalHiddenLayerWeigths());
    } catch (Exception e) {
      help(1);
    }

  }

  private static void help(int exitcode) {

    String help = """
                  Neural Network to asses handwritten numbers.
                                    
                    -h                                      opens the help
                    -c                                      clean run, ignores existing weights and creates new weights matrix.
                    -g <int>                                the generations to run for.
                    -l | --learning-rate <double>           the learning rate in percent.
                    -i | --increase-learning-rate           Increases the learning rate with each element until a learning rate of 100 % is reached. 
                    --hidden                                the hidden layers generated, should be used with option [-c]
                  How to use:
                      java -jar neuronalesJavaNetz.jar -g 5 -c --lr 3 -i 7.5 
                  """;

    System.out.println(help);
    System.exit(exitcode);
  }

  public static void assessRandom() {

    var weights = fetchWeigths(2, false);
    double[][] fst = weights.get(0);
    double[][] sec = weights.get(1);
    double[][] thrd = weights.get(2);
    var net = new neural.network.NeuralNet(fst, sec, thrd);

    Random rand = new Random();
    int r = rand.nextInt(0, 60000);
    double[] img = getImage(r);
    short l = getLable(r);

    double[] out = net.assess(img);

    printResult(l, out);
  }

  public static void assess(double[] inp) {

    var weights = fetchWeigths(2, false);
    double[][] fst = weights.get(0);
    double[][] sec = weights.get(1);
    double[][] thrd = weights.get(2);
    var net = new neural.network.NeuralNet(fst, sec, thrd);

    double[] out = net.assess(inp);

    printResult((short) -1, out);
  }

  private static void printResult(short l, double[] out) {

    String br = "=".repeat(20);
    System.out.printf("%s\nInput was %s, results:\n", br, l);

    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < out.length; i++) {
      String s = String.format("%s: %3.1f", i, out[i]);
      sb.append("\t").append(s).append("\n");
    }
    System.out.println(sb);

  }

  private static short getLable(int r) {


    short lbl = 0;
    short offset = 8;
    try (var chan = Files.newByteChannel(Path.of(
        CONTROL_LABEL_PATH))) {
      chan.position(8);
      byte[] arr = new byte[60000];
      ByteBuffer buff = ByteBuffer.wrap(arr);
      chan.read(buff);
      lbl = arr[r];
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return lbl;
  }

  private static double[] getImage(int r) {

//    Path path = Path.of("src/resources/data/t10k-images.idx3-ubyte");
    Path path = Path.of(TRAIN_IMAGE_PATH);

    short size = 28 * 28;
    short offset = 16;
    double[] arr = new double[size];
    try (var channel = Files.newByteChannel(path)) {
      int start = r * size;

      channel.position(offset + start);
      byte[] b = new byte[size];
      ByteBuffer buff = ByteBuffer.wrap(b);
      channel.read(buff);

      for (int i = 0; i < b.length; i++) {
        arr[i] = (b[i] & 0xff) / 255.;
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return arr;
  }

  /**
   * Fetches weights from file or generates weights for the number of hidden layers supplied. The
   * amount of weights generated is equal to the number of hidden layers +1.
   *
   * @param hiddenLayers amount of hidden layers.
   * @return a list of weigths
   */
  private static List<double[][]> fetchWeigths(int hiddenLayers, boolean clean) {

    var gson = new Gson();
    int prevSize = 784;


    List<double[][]> weights = new ArrayList<>();
    for (int i = 0; i <= hiddenLayers; i++) {
      int outSize = i == hiddenLayers ? 10 : (int) Math.sqrt(prevSize * 10);

      final String path = String.format(BASEPATH, i);
      try {
        File f = new File(path);
        if (clean || f.createNewFile() || "".equals(Files.readString(Path.of(path)))) {
          weights.add(generateRandomWeigths(i, prevSize, outSize));
        } else {
          weights.add(gson.fromJson(Files.readString(Path.of(path)), double[][].class));
        }
      } catch (IOException e) {
        weights.add(generateRandomWeigths(i, prevSize, outSize));
      }

      prevSize = outSize;
    }
    return weights;
  }

  private static double[][] generateRandomWeigths(int index, int prevSize, int outSize) {

    double[][] arr = new double[outSize][prevSize];

    for (int i = 0; i < arr.length; i++) {
      for (int j = 0; j < arr[i].length; j++) {
        arr[i][j] = Math.random() * .1;
      }
    }
    return arr;
  }

  private static void saveWeights(double[][]... weights) {

    var gson = new Gson();
    for (int i = 0; i < weights.length; i++) {
      final double[][] weight = weights[i];
      String json = gson.toJson(weight);
      String filePath = String.format(BASEPATH, i);
      try {
        Files.write(Path.of(filePath), json.getBytes());
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

}
