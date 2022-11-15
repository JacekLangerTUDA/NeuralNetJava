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
import java.util.concurrent.atomic.AtomicInteger;
import neural.network.linearalgebra.Matrix;

/**
 * Neural network for to process handwritten data from MNIST datasets.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 01.11.2022</p>
 */
public class NeuralNet {

  private static final List<double[][]> WEIGHTS = new ArrayList<>();
  private static final int MAX_IMAGE_COUNT = 60_000;
  private double[][] fstHiddenLayerWeigths;
  private double[][] secondHiddenLayerWeigths;
  private double[][] finalHiddenLayerWeigths;
  private double[] fstHLayer;
  private double[] scndHlayer;
  private List<Boolean> hits;

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

  /**
   * Runs the training session of the network.
   *
   * @param generations number of generations to run
   * @param lr          the learningrate to begin with
   * @param ilr         the increment of the learningrate with each generation
   */
  public void train(int generations, double lr, double ilr) {

    System.out.println("\t\t\tTRAINING");
    hits = new ArrayList();
    short tOffset = 16;
    short lOffset = 8;
    short imageSize = 28 * 28;
    Path train = Path.of(Main.TRAIN_IMAGE_PATH);
    Path lbl = Path.of(Main.TRAIN_LABEL_PATH);
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

      AtomicInteger counter = new AtomicInteger();

      for (int i = 0; i < generations; i++) {

        for (int j = 0; j < MAX_IMAGE_COUNT; j++) {
          long current = System.currentTimeMillis();
          progressBar(counter.getAndIncrement(), total, current - start);
          // get a random image
          int r = rand.nextInt(0, MAX_IMAGE_COUNT);
          double[] img = Arrays.copyOfRange(trainDouble, r * imageSize,
                                            r * imageSize + imageSize);
          short lable = lBuff.array()[r];

          hits.add(processAndCorrect(lr, img, lable));
          lr += ilr;
        }
      }


    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }


  }

  private double hitrate() {

    double hitcount = 0f;
    if (hits.size() == 0) {
      return 0;
    }
    for (Boolean hit : hits) {
      if (hit) {
        hitcount++;
      }
    }
    return (hitcount / (double) hits.size()) * 100f;
  }

  private void progressBar(int prog, long total, long dur) {

    double p = ((prog + 1) / (double) total) * 100f;
    String bars = "=".repeat((int) ((prog + 1) * 20 / total)) + ">";

    int min = (int) (dur / 60000);
    double sec = dur / 1000f - (min * 60);
    System.out.printf("%05.2f [%-21s] %9s/%s dur: %02d:%05.2f acc: %7.2f \r",
                      p,
                      bars, prog + 1, total,
                      min, sec, hitrate());
    System.out.flush();
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
  public boolean processAndCorrect(double learingrate, double[] input, short lbl) {

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

    return isHit(lbl, out);
  }

  private boolean isHit(short lbl, double[] out) {

    int index = -1;
    double max = Arrays.stream(out).max().getAsDouble();
    for (int i = 0; i < out.length; i++) {
      if (out[i] == max) {
        index = i;
        break;
      }

    }
    return index == lbl;
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
