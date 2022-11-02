package neural.network;

import static java.lang.Math.E;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import neural.network.linearalgebra.Matrix;

/**
 * Neural network for to process handwritten data from MNIST datasets.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 01.11.2022</p>
 */
public class NeuralNet {

  private static final int OUTPUT_SIZE = 10;
  private static final List<double[][]> WEIGHTS = new ArrayList<>();
  private static final List<double[]> HIDDEN_LAYERS_OUT = new ArrayList<>();
  private static final int MAX_IMAGE_COUNT = 60_000;
  private double[][] fstHiddenLayerWeigths;
  private double[][] secondHiddenLayerWeigths;
  private double[][] finalHiddenLayerWeigths;
  private double[] fstHLayer;
  private double[] scndHlayer;

  public double[][] getFstHiddenLayerWeigths() {

    return fstHiddenLayerWeigths;
  }

  public double[][] getSecondHiddenLayerWeigths() {

    return secondHiddenLayerWeigths;
  }

  public double[][] getFinalHiddenLayerWeigths() {

    return finalHiddenLayerWeigths;
  }

  /**
   * Creates a new network with provided data
   */
  public NeuralNet(double[][] fstWeights,
                   double[][] scndWeights,
                   double[][] fnlLayerWeights) {

    this.fstHiddenLayerWeigths = fstWeights;
    this.secondHiddenLayerWeigths = scndWeights;
    this.finalHiddenLayerWeigths = fnlLayerWeights;
    Collections.addAll(WEIGHTS, fstWeights, scndWeights, fnlLayerWeights);
  }

  public void train(int generations, double lr) {


    short tOffset = 16;
    short lOffset = 8;
    short imageSize = 28 * 28;
    Path train = Path.of("src/main/resources/data/train-images.idx3-ubyte");
    Path lbl = Path.of("src/main/resources/data/train-labels.idx1-ubyte");
    // open a channel to the file data.
    try (var trainChannel = Files.newByteChannel(train);
         var lblChannel = Files.newByteChannel(lbl)) {
      trainChannel.position(tOffset);
      lblChannel.position(lOffset);
      byte[] trainBytes = new byte[MAX_IMAGE_COUNT * imageSize];
      byte[] lblBytes = new byte[MAX_IMAGE_COUNT];
      var tBuff = ByteBuffer.wrap(trainBytes);
      var lBuff = ByteBuffer.wrap(lblBytes);
      trainChannel.read(tBuff);
      lblChannel.read(lBuff);

      double[] trainDouble = new double[MAX_IMAGE_COUNT * imageSize];
      for (int i = 0; i < tBuff.array().length; i++) {
        byte b = tBuff.array()[i];
        trainDouble[i] = (b & 0xff) / 255.;
      }
      Random rand = new Random();

      long start = System.currentTimeMillis();
      long total = (long) MAX_IMAGE_COUNT * generations;
      for (int i = 0; i < generations; i++) {
        for (int j = 0; j < MAX_IMAGE_COUNT; j++) {
          long current = System.currentTimeMillis();
          progressBar((i + 1) * j, total, current - start);
          // get a random image
          int r = rand.nextInt(0, MAX_IMAGE_COUNT);
          double[] img = Arrays.copyOfRange(trainDouble, r * imageSize,
                                            r * imageSize + imageSize);
          short lable = lBuff.array()[r];

          processAndCorrect(lr, img, lable);
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

  private void progressBar(int prog, long total, long dur) {

    double p = ((prog + 1) / (double) total) * 100.;
    String bars = "=".repeat((int) ((prog + 1) * 20 / total)) + ">";

    int hours = (int) (dur / 60000 * 60);
    int min = (int) (dur / 60000);
    float sec = (float) (dur / 1000.);
    System.out.printf("%5.2f [%-21s] %s/%s,\tduration: %2s:%2s:%5.2f\r", p, bars, prog + 1, total,
                      hours, min, sec);
  }

  /**
   * Applies the sigmoid function to every element of an array.
   *
   * @param arr the array of doubles
   * @return the sigmoid corrected array
   */
  private double[] sigmoid(double[] arr) {

    double[] temp = new double[arr.length];
    for (int i = 0; i < arr.length; i++) {
      double x = -1 * arr[i];
      temp[i] = (1 / (1 + Math.pow(E, x)));
    }
    return temp;
  }

  /**
   * Assess image data and return the array of possibilities. In a well-trained network the output
   * should converge to 1 for the only a single node and to 0 for all others.
   *
   * @param input the image data
   * @return the array of output nodes
   */
  public double[] assess(double[] input) {

    double[] fstHidden = Matrix.mult(fstHiddenLayerWeigths, input);
    fstHLayer = sigmoid(fstHidden);
    double[] sndHidden = Matrix.mult(secondHiddenLayerWeigths, fstHLayer);
    scndHlayer = sigmoid(sndHidden);
    return sigmoid(Matrix.mult(finalHiddenLayerWeigths, scndHlayer));
  }

  /**
   * Starts the training of the network for the given amount of generations.
   *
   * @param learingrate the learning rate at which the network learns
   * @param lbl         the lable of for the current entry
   */
  public void processAndCorrect(double learingrate, double[] input, short lbl) {

    // input to hidden
    double[] fstHidden = Matrix.mult(fstHiddenLayerWeigths, input);
    fstHLayer = sigmoid(fstHidden);
    // first hidden to second hidden
    double[] sndHidden = Matrix.mult(secondHiddenLayerWeigths, fstHLayer);
    scndHlayer = sigmoid(sndHidden);
    // second hidden to out
    double[] out = sigmoid(Matrix.mult(finalHiddenLayerWeigths, scndHlayer));

    /*
     * correction function
     * */
    // set target for error calculation
    double[] target = new double[out.length];
    target[lbl] = 1;

    double[] err = Matrix.sub(target, out);
    // calc consecutive errors for the hidden layers
    // second hidden layer error is the error times the weights used for calculating the final
    // outputs.
    // => the Weighted sum of all errors used for the final output
    double[] errSndHid = Matrix.multTrans(finalHiddenLayerWeigths, err);
    double[] errFstHid = Matrix.multTrans(secondHiddenLayerWeigths, errSndHid);

       /*
       calc the absolute errors and correct by the learning rate
       the absolute error is the differential of the sigmoid func for the given layer.
       Using the output functions of each layer we can omit another use of the sigmoid function
       as we already adjusted all values. As a result we need to multiply the error with the
       output of the layer and 1 - the output as well as the uncorrected output.
       => f'(x) = -(target - err) * sig(out) * (1-sig(out)) *out
       <=> f'(x) = -(target - err) * out * (1 - out) * outHid.T
       this value will be subtracted from the current weigths after
       */

    finalHiddenLayerWeigths = Matrix.add(finalHiddenLayerWeigths,
                                         correctErrors(err,
                                                       out,
                                                       scndHlayer,
                                                       learingrate));
    // second layer
    secondHiddenLayerWeigths = Matrix.add(secondHiddenLayerWeigths,
                                          correctErrors(errSndHid,
                                                        scndHlayer,
                                                        fstHLayer,
                                                        learingrate));
    // first layer
    fstHiddenLayerWeigths = Matrix.add(fstHiddenLayerWeigths,
                                       correctErrors(errFstHid,
                                                     fstHLayer,
                                                     input,
                                                     learingrate));
  }

  /**
   * Calculate the error of the given output and creates a weigthed error matrix for the weights
   * leading into the output layer.
   *
   * @param err          the error from the current output
   * @param curOut       the current output
   * @param prevHidden   the previous hidden layer
   * @param learningRate the learning rate at which changes are applied to the weights
   * @return the errors for the previous hidden layer
   */
  private double[][] correctErrors(double[] err,
                                   double[] curOut,
                                   double[] prevHidden,
                                   double learningRate) {

    double[] errHid = Matrix.multVertices(err, curOut, Matrix.sub(curOut, 1));
    return Matrix.mult(learningRate, Matrix.multTrans(errHid, prevHidden));
  }

}
