package neural.network.reader;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 21.09.2022</p>
 */
public interface MnistReadOnly {

  /**
   * Returns the image with the given index from the MNIST train data file.
   *
   * @param index the index of the image
   * @return byte array of image data 28x28 pixel -> 784 byte
   */
  double[] readTrainImage(int index);

  /**
   * Returns the image with the given index from the MNIST train data file.
   *
   * @param index the index of the label
   * @return label of the image
   */
  short readTrainLabel(int index);

  /**
   * Returns the image with the given index from the MNIST control data file.
   *
   * @param index the index of the image
   * @return byte array of image data 28x28 pixel -> 784 byte
   */
  double[] readControlImage(int index);

  /**
   * Returns the image with the given index from the MNIST control data file.
   *
   * @param index the index of the label
   * @return label of the image
   */
  short readControlLabel(int index);

}
