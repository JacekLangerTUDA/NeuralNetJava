package neural.network.math;

import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.util.stream.IntStream;
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
   * the first and n the number of columns in the second matrix. Parallel execution.
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

    IntStream.range(0, m).parallel().forEach(i -> {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < second.length; k++) {
          temp[i][j] += first[i][k] * second[k][j];
        }
      }
    });

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
    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int k = 0; k < input.length; k++) {
        temp[i] += weights[i][k] * input[k];
      }
    });

    return temp;
  }

  /**
   * Multiplies two matrices of which the first matrix is a Nx1 matrix and the second will be
   * handled as a 1xM matrix. the result is a matrix with dimensions of Mx1.
   *
   * @param fst the weights matrix
   * @param scd the input matrix
   * @return new input layer
   */
  public static double[][] mMult(final double[] fst, final double[] scd) {


    double[][] temp = new double[fst.length][scd.length];

    IntStream.range(0, fst.length).parallel().forEach(i -> {
      for (int k = 0; k < scd.length; k++) {
        temp[i][k] += fst[i] * scd[k];
      }
    });

    return temp;
  }


  /**
   * Multiplies two matrices of which the first matrix is a Nx1 matrix and the second will be
   * handled as a 1xM matrix. the result is a matrix with dimensions of Mx1.
   *
   * @param fst the weights matrix
   * @param scd the input matrix
   * @return new input layer
   */
  public static double[][] mMultWeighted(final double[] fst, final double[] scd, double weight) {


    double[][] temp = mMult(fst, scd);

    return mult(weight, temp);
  }

  public static double[][] mAdd(double[][] fst, double[][] scd) {

    if (fst.length != scd.length || fst[0].length != scd[0].length) {
      throw new IllegalMathOperationException(
          "Invalid matrix size. Can not add matrix of different sizes");
    }

    double[][] temp = new double[fst.length][fst[0].length];

    IntStream.range(0, fst.length).parallel().forEach(i -> {
      for (int j = 0; j < fst[i].length; j++) {
        temp[i][j] = fst[i][j] + scd[i][j];
      }
    });
    return temp;
  }


  public static double scalar(double[] fst, double[] sec) {

    double val = 0;

    for (int i = 0; i < fst.length; i++) {
      val += fst[i] * sec[i];
    }

    return val;
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

  /**
   * Multiplies a vector with a given alpha
   *
   * @param alpha the alpha to multiply with
   * @param arr   the array
   * @return new array with modified values
   */
  public static double[][] mult(double alpha, double[][] arr) {

    var temp = new double[arr.length][arr[0].length];
    IntStream.range(0, arr.length).parallel().forEach(i -> {

      for (int j = 0; j < arr[i].length; j++) {
        temp[i][j] = arr[i][j] * alpha;
      }
    });
    return temp;
  }

  /**
   * Transponse of a matrix.
   *
   * @param matrix matrix to transponse
   * @return a transponse of the original matrix m<pow>T</pow>
   */
  public static double[][] trans(double[][] matrix) {

    double[][] temp = new double[matrix[0].length][matrix.length];

    IntStream.range(0, matrix.length).parallel().forEach(
        i -> {
          for (int j = 0; j < matrix[i].length; j++) {
            temp[j][i] = matrix[i][j];
          }
        }
    );

    return temp;
  }

}
