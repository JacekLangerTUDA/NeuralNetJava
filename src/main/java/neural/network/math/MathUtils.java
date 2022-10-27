package neural.network.math;

import static java.lang.Math.E;
import static java.lang.Math.pow;

import neural.network.math.exception.IllegalMathOperationException;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 21.09.2022</p>
 */
public class MathUtils {

  public static double sigmoid(double x) {

    return 1 / (1 + (pow(E, -1 * x)));
  }

  /**
   * Applies the sigmoid function to every element of an array.
   *
   * @param arr the array of doubles
   * @return the sigmoid corrected array
   */
  public static double[] sigmoid(double[] arr) {

    double[] temp = new double[arr.length];
    for (int i = 0; i < arr.length; i++) {
      temp[i] = sigmoid(arr[i]);
    }
    return temp;
  }


  public static double sum(double... vals) {

    double sum = 0;

    for (double val : vals) {
      sum += val;
    }
    return sum;
  }


  /**
   * Multiplies two matrices. returns a new matrix of the size MxN where m is the number of rows in
   * the first and n the number of columns in the second matrix.
   *
   * @param first  first matrix
   * @param second second matrix
   * @return new matrix MxN
   */
  public static double[][] mMult(double[][] first, double[][] second) {

    int m, n;
    m = first.length;
    n = second[0].length;

    if (m != n) {
      throw new IllegalMathOperationException("Matrizen incompatibel");
    }

    double[][] temp = new double[first.length][second[0].length];

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < second.length; k++) {
          temp[i][j] += first[i][k] * second[k][j];
        }
      }
    }
    return temp;
  }

  /**
   * Multiplies two matrices of which the input matrix is a Nx1 matrix and the weights are of MxN
   * the result is a matrix with dimensions of Mx1.
   *
   * @param weights the weights matrix
   * @param input   the input matrix
   * @return new input layer
   */
  public static double[] mMult(final double[][] weights, final double[] input) {

    if (weights[0].length != input.length) {
      throw new IllegalMathOperationException("Invalid matrix operation");
    }

    double[] temp = new double[weights.length];

    for (int i = 0; i < weights.length; i++) {
      for (int k = 0; k < input.length; k++) {
        temp[i] += weights[i][k] * input[k];
      }
    }

    return temp;
  }

  /**
   * Multiplies a vector with a given alpha
   *
   * @param alpha the alpha to multiply with
   * @param arr   the array
   * @return new array with modified values
   */
  public static double[] mult(double alpha, double[] arr) {

    var temp = new double[arr.length];
    for (int i = 0; i < arr.length; i++) {
      temp[i] = arr[i] * alpha;
    }
    return temp;
  }

}
