package neural.network;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
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

  private static final Path WEIGHTS_PATH = Path.of("src/main/resources/weights.json");
  public static List<double[][]> WEIGHTS = new ArrayList<>();

  private static double CORRECTION_RATE = .03;

  public static void main(String[] args) {

    train(Integer.MAX_VALUE);
  }

  private static void train(int generations) {

    double[] pixelArray = null;
    short label = 0;
    int generation = 0;
    MnistReader reader = new MnistReader();
    fetchWeightsFromJson();
    while (generation++ < 2) {
      var rand = (int) (Math.random() * 100);
      pixelArray = reader.readTrainImage(rand);
      label = reader.readTrainLabel(rand);
      double[] out = processLayers(pixelArray);
      double[] errors = MathUtils.mult(CORRECTION_RATE, getErrorFromOutput(out, label));
      correctWeights(errors);
    }
    updateWeightsFile();
  }

  private static void correctWeights(double[] errors) {


  }

  private static double[] processLayers(double[] nodes) {

    // a Layer is a vector of nodes, input is a matrix of inputs
    // each node has a vector of inputs therefore the input matrix is of size MxN where
    // M is the number of inputs and N is the number of nodes in the Layer.

    int i = 0;
    double[] tempLayer = nodes;
    int layerSize = INPUT_NODES;    // 784 nodes
    double[][] iW = getWeightsForLayer(i, layerSize);   // initial matrix 392x784

    while (layerSize >> 1 > 10) {
      // half the size of the layer,
      // but make sure we have at least 10  nodes
      layerSize = Math.max((layerSize >> 1), 10);

      tempLayer = MathUtils.sigmoid(MathUtils.mMult(iW, tempLayer));

      iW = getWeightsForLayer(++i, layerSize);    // get the next layer weights
    }
    iW = getWeightsForLayer(++i, layerSize);
    tempLayer = MathUtils.sigmoid(MathUtils.mMult(iW, tempLayer));

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
        errors[i] = 1 - output[i];
      } else {
        errors[i] = 0 - output[i];
      }
    }
    return errors;
  }

}
