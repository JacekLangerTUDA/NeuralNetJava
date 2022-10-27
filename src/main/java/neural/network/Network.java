package neural.network;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import neural.network.math.MathUtils;
import neural.network.models.Layer;
import neural.network.models.NodesList;
import neural.network.reader.MnistReader;
import neural.network.utils.Validate;


/**
 * The Neural Network to train on the MNIST data.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 20.09.2022</p>
 */
public class Network extends NeuralLogging {

  public static final int INPUT_NODES = 784;

  public static final double CORRECTION_RATE = .1;

  private static final Path WEIGHTS_PATH = Path.of("src/main/resources/weights.json");
  /**
   * unsigmoided layer values.
   */
  private static final List<double[]> LAYERS = new ArrayList<>();
  public static List<double[][]> WEIGHTS = new ArrayList<>();

  public static void main(String[] args) {

    train(1);
    MnistReader reader = new MnistReader();
    assess(reader.readControlImage(78), reader.readControlLabel(78));
  }

  public static void assess(double[] img, int expected) {

    fetchWeightsFromJson();
    double[] out = processLayers(img);
    StringBuilder sb = new StringBuilder();

    sb.append("-----------------------")
      .append("\n\tExpected: ").append(expected).append("\n\n");
    for (int i = 0; i < 10; i++) {
      sb.append("\t\t").append(i).append(":\t").append(out[i]).append("\n");
    }
    sb
        .append("-----------------------");

    info(sb.toString());
  }

  public static void train(int generations) {

    double[] pixelArray = null;
    short label = 0;
    MnistReader reader = new MnistReader();
    fetchWeightsFromJson();

    for (int g = 0; g < generations; g++) {

      for (int n = 0; n < 100; n++) {
        var rand = (int) (Math.random() * 100);
        pixelArray = reader.readTrainImage(rand);
        label = reader.readTrainLabel(rand);
        double[] out = processLayers(pixelArray);
        double[] errors = getErrorFromOutput(out, label);
        int layers = WEIGHTS.size() - 1;
        for (int i = layers; i >= 0; i--) {
          errors = backpropagation(errors, i);
        }
      }
    }

    updateWeightsFile();
  }

  /**
   * Calculate all the errors of the hidden layers and correct the weights accordingly.
   *
   * @param errors errors from last layer
   * @param i      the index of the current layer
   * @return
   */
  private static double[] backpropagation(double[] errors, int i) {

// TODO: 27.10.2022 needs fixing
    final double[][] weights = (WEIGHTS.get(i));
    final double[] layer = LAYERS.get(i);
    double[] errorTemp = new double[layer.length];
    for (int k = 0; k < errors.length; k++) {
      int length = weights[k].length;
      for (int j = 0; j < length; j++) {

        double oj = layer[j];
        double sig = MathUtils.sigmoid(MathUtils.scalar(weights[k], layer));
        double err = -1 * errors[k] * sig * (1 - sig) * oj;

        errorTemp[j] = err;

        WEIGHTS.get(i)[k][j] += err; //* CORRECTION_RATE;
      }
    }
    return errorTemp;
  }

  private static double[] processLayers(double[] nodes) {

    // a Layer is a vector of nodes, input is a matrix of inputs
    // each node has a vector of inputs therefore the input matrix is of size MxN where
    // M is the number of inputs and N is the number of nodes in the Layer.

    int i = 0;
    double[] tempLayer = nodes;
    int layerSize = INPUT_NODES;    // 784 nodes
    double[][] iW = getWeightsForLayer(i, layerSize);   // initial matrix 392x784
    LAYERS.add(nodes);
    while (layerSize >> 1 > 10) {
      // half the size of the layer,
      // but make sure we have at least 10  nodes
      layerSize = Math.max((layerSize >> 1), 10);
      // use sigmoid function on value.

      ExecutorService executorService = Executors.newCachedThreadPool();

      tempLayer = MathUtils.sigmoid(MathUtils.mMult(iW, tempLayer));
      LAYERS.add(tempLayer);
      // get next layers weights
      iW = getWeightsForLayer(++i, layerSize);    // get the next layer weights
    }
    tempLayer = MathUtils.sigmoid(MathUtils.mMult(iW, tempLayer));
    LAYERS.add(tempLayer);

    return tempLayer;
  }


  /**
   * Gets or generates the weights for the current layer.
   *
   * @param i  the index of the layer
   * @param in size of the input layer
   * @return weights matrix for the current layer
   */
  private static double[][] getWeightsForLayer(int i, int in) {

    int out = Math.max(in >> 1, 10);
    // no weights.json in list for the current Layer. check file for weights.json.
    if (WEIGHTS.size() - 1 < i) {   // includes empty
      WEIGHTS.add(generateRandomWeights(in, out));
    }

    return WEIGHTS.get(i);
  }

  private static double[][] generateRandomWeights(int in, int out) {

    double[][] temp = new double[out][in];
    for (int n = 0; n < out; n++) {
      for (int m = 0; m < in; m++) {
        temp[n][m] = Math.random();
      }
    }

    return temp;
  }

  private static void updateWeightsFile() {

    var gson = new Gson();

    NodesList nodesList = new NodesList();

    for (double[][] weight : WEIGHTS) {
      nodesList.add(new Layer(weight));
    }

    String json = gson.toJson(nodesList);

    try (var writer = Files.newBufferedWriter(WEIGHTS_PATH)) {
      writer.write(json);
    } catch (IOException e) {
      error(e.getMessage());
    }

  }

  private static void fetchWeightsFromJson() {

    Gson gson = new Gson();
    try {
      String json = Files.readString(WEIGHTS_PATH);
      if (Validate.isNotBlank(json)) {

        NodesList data = gson.fromJson(json, NodesList.class);

        for (Layer layer : data.getLayers()) {
          WEIGHTS.add(layer.getWeightsMatrix());
        }
      }

    } catch (IOException | ClassCastException e) {
      error(e.getMessage());
    }

  }

  private static double[] getErrorFromOutput(double[] output, short label) {

    double[] errors = new double[10];
    for (int i = 0; i < errors.length; i++) {
      if (i == label) {
        errors[i] = (1 - output[i]) * (1 - output[i]);
      } else {
        errors[i] = (0 - output[i]) * (0 - output[i]);
      }
    }
    return errors;
  }

}
