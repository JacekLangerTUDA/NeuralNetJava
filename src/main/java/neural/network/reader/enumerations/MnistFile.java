package neural.network.reader.enumerations;

import java.io.File;
import java.nio.file.Path;

/**
 * Enumeration of the MNIST file types.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 20.09.2022</p>
 */
public enum MnistFile {
  /**
   * The path to the training's data.
   */
  TRAIN_IMAGE("src/main/resources/data/train-images.idx3-ubyte", 16),
  /**
   * Path to the labels of the training's data.
   */
  TRAIN_LABEL("src/main/resources/data/train-labels.idx1-ubyte", 8),
  /**
   * Path to the control images.
   */
  CONTROL_IMAGE("src/main/resources/data/t10k-images.idx3-ubyte", 16),
  /**
   * Path to the control labels.
   */
  CONTROL_LABEL("src/main/resources/data/t10k-labels.idx1-ubyte", 8),
  /**
   * Path to weights.json file.
   */
  WEIGTHS("src/main/resources/weights.json", 0);

  /**
   * The path to the file.
   */
  final String path;
  /**
   * Start point where reading begins.
   */
  final int start;

  /**
   * Returns the Path of the file.
   *
   * @return {@link Path} of the file
   */
  public Path getPath() {

    return Path.of(path);
  }

  /**
   * Returns the index of the first entry.
   *
   * @return index of the first entry
   */
  public int getStart() {

    return start;
  }

  /**
   * Returns the file associated.
   *
   * @return {@link File}
   */
  public File getFile() {

    return new File(path);
  }

  /**
   * Creates a new Mnist File
   *
   * @param path  path to the file
   * @param start index of the first entry.
   */
  MnistFile(String path, int start) {

    this.path = path;
    this.start = start;
  }
}
