package neural.network.models;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 27.10.2022</p>
 */
public class Layer {

  // TODO(Jack): 27.10.2022 Implement.
  double[][] weightsMatrix;

  public double[][] getWeightsMatrix() {

    return weightsMatrix;
  }

  public Layer(double[][] matrix) {

    this.weightsMatrix = matrix;
  }

}
