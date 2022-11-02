package neural.network.linearalgebra;

import java.util.stream.IntStream;
import neural.network.linearalgebra.exception.IllegalMathOperationException;
import org.jetbrains.annotations.NotNull;

/**
 * Linear Algebra Math library.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 01.11.2022</p>
 */
public class Matrix {


  /**
   * Multiplies two matrices of which the input matrix is a Nx1 matrix and the weights are of MxN
   * the result is a matrix with dimensions of Mx1.
   *
   * @param weights the weights matrix
   * @param input   the input matrix
   * @return new input layer
   */
  public static double[] mult(final double[] input, final double[][] weights) {

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
   * Multiplies two matrices of which the input matrix is a Nx1 matrix and the weights are of MxN
   * the result is a matrix with dimensions of Mx1.
   *
   * @param matrix the weights matrix
   * @param vector the input matrix
   * @return new input layer
   */
  public static double[] mult(final double[][] matrix, final double[] vector) {

    if (matrix[0].length != vector.length) {
      throw new IllegalMathOperationException("Invalid matrix operation");
    }

    double[] temp = new double[matrix.length];
    IntStream.range(0, matrix.length).parallel().forEach(i -> {
      for (int k = 0; k < vector.length; k++) {
        temp[i] += matrix[i][k] * vector[k];
      }
    });

    return temp;
  }

  /**
   * add Second matrix to first and override values in first matrix to recive the result after
   * addition.
   *
   * @param fst first matrix
   * @param scd second matrix
   */
  public static double[][] add(double[][] fst, double[][] scd) {

    double[][] tmp = new double[fst.length][fst[0].length];
    if (fst.length != scd.length || fst[0].length != scd[0].length) {
      throw new IllegalMathOperationException(
          "Invalid matrix size. Can not add matrix of different sizes");
    }

    IntStream.range(0, fst.length).parallel().forEach(i -> {
      for (int j = 0; j < fst[i].length; j++) {
        tmp[i][j] = fst[i][j] + scd[i][j];
      }
    });
    return tmp;
  }

  /**
   * Multiplies two matrices. returns a new matrix of the size MxN where m is the number of rows in
   * the first and n the number of columns in the second matrix. Parallel execution.
   *
   * @param first  first matrix
   * @param second second matrix
   * @return new matrix MxN
   */
  public static double[][] mult(double[][] first, double[][] second) {

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
   * Multiplies a matrix with a given alpha
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
   * Transpose of a matrix.
   *
   * @param matrix matrix to transpose
   * @return a transpose of the original matrix m<pow>T</pow>
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

  /**
   * Subtracts two vectors from one another.
   *
   * @param target the first vector
   * @param out    the second vector
   * @return first vector minus second vector
   */
  public static double[] sub(double[] target, double[] out) {

    double[] temp = new double[target.length];
    for (int i = 0; i < target.length; i++) {

      temp[i] = target[i] - out[i];
    }
    return temp;
  }

  /**
   * Subtract n from each instance in the target vector.
   *
   * @param target the target vector
   * @param n      the amount to subtract of each value
   * @return target vector reduced by n
   */
  public static double[] sub(double[] target, double n) {

    double[] temp = new double[target.length];
    for (int i = 0; i < target.length; i++) {

      temp[i] = n - target[i];
    }
    return temp;
  }

  /**
   * Multiplies the values of multiple vertices and return a single vector with the product values
   * for each row. this is a shorthand method to generate a product for each row in multiple
   * vertices of the same size and return a new vector.
   *
   * @param verts the vertices to multiply
   * @return a new vector with the product of each vector row for each row.
   */
  public static double[] multVertices(@NotNull double[]... verts) {

    double[] tmp = verts[0];
    for (int i = 1; i < verts.length; i++) {
      if (verts[i].length != tmp.length) {
        throw new IllegalMathOperationException(
            "illegal operation on vertices of different length");
      }
      for (int j = 0; j < tmp.length; j++) {
        tmp[j] *= verts[i][j];
      }
    }

    return tmp;
  }

  /**
   * Transpose the matrix and multiply with the vector. This is a shorthand method for first
   * transposing a MxN matrix and before multiplying with an M-dimensional vector.
   *
   * @param matrix the matrix
   * @param vector the vector
   * @return return a new vector of N dimensions
   */
  public static double[] multTrans(double[][] matrix, double[] vector) {

    var t = trans(matrix);
    return mult(t, vector);
  }

  /**
   * Multiplies two Matrices Nx1/Vertices of which the second is treated as a transpose matrix.
   *
   * @param fst  the first vector / array
   * @param scnd the second vector / array
   * @return a MxN matrix of which M is the length of the first vector and N the length of the
   *     second.
   */
  public static double[][] multTrans(double[] fst, double[] scnd) {

    double[][] tmp = new double[fst.length][scnd.length];

    IntStream.range(0, tmp.length).parallel().forEach(i -> {
      for (int j = 0; j < scnd.length; j++) {
        tmp[i][j] += fst[i] * scnd[j];
      }
    });

    return tmp;
  }

  /**
   * Calculate the dotproduct of two vectors.
   *
   * @param fst first vector
   * @param snd second vector
   * @return a scalar.
   */
  public double[] dot(double[] fst, double[] snd) {

    if (fst.length != snd.length) {
      throw new IllegalMathOperationException(
          "Unable to calculate the dot product of two vectors of size %sx%s",
          fst.length, snd.length);
    }

    double[] tmp = new double[fst.length];
    IntStream.range(0, fst.length).parallel().forEach(i -> {
      tmp[i] += fst[i] * snd[i];
    });

    return tmp;
  }

}
