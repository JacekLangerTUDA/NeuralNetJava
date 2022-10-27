package neural.network.reader;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import neural.network.reader.enumerations.MnistFile;

/**
 * Reader class to process MNIST files.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 21.09.2022</p>
 */
public class MnistReader extends MnistReaderBase {

  @Override
  public double[] readTrainImage(int index) {

    return readImage(index, MnistFile.TRAIN_IMAGE);
  }

  @Override
  public short readTrainLabel(int index) {

    return readLabel(index, MnistFile.TRAIN_LABEL);
  }

  @Override
  public double[] readControlImage(int index) {

    return readImage(index, MnistFile.CONTROL_IMAGE);
  }

  @Override
  public short readControlLabel(int index) {

    return readLabel(index, MnistFile.CONTROL_LABEL);
  }

  /**
   * Fetch weights.json for the corresponding Layer. this function will create an array of random
   * weights.json if there are no weights.json present.
   *
   * @param layerIndex index of the Layer being processed
   * @param inputs     the size of the Layer
   * @return weights.json for the Layer.
   */
  public double[][] fetchWeights(int layerIndex, int inputs, int outputs) {

    String weightsString = null;
    Path weightsFile = MnistFile.WEIGTHS.getPath();
    try {
      File file = weightsFile.toFile();
      if (weightsFile.toFile().exists()) {
        weightsString = Files.readString(weightsFile).split("\n")[layerIndex];
        if (isEmptyString(weightsString)) {
          weightsString = createWeightsData(file, inputs, outputs);
        }
      } else {
        file.createNewFile();
        weightsString = createWeightsData(file, inputs, outputs);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (ArrayIndexOutOfBoundsException e) {
      try {
        weightsString = createWeightsData(weightsFile.toFile(), inputs, outputs);
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    }

    var nums = weightsString.split(",");

    double[][] weights = new double[inputs][outputs];
    for (int w = 0; w < inputs; w++) {
      for (int h = 0; h < outputs; h++) {
        weights[h][w] = Double.valueOf(nums[h + h]);
      }
    }

    return weights;
  }

  private boolean isEmptyString(String str) {

    return "".equals(str);
  }

  private String createWeightsData(File file, int inputs, int outputs) throws IOException {

    var arr = new double[inputs * outputs];
    StringBuilder sb = new StringBuilder(arr.length);

    for (int i = 0; i < arr.length; i++) {
      arr[i] = Math.random();
      // do not append to the last instance
      if (i < arr.length) {
        sb.append(arr[i]).append(",");
      }
    }

    Files.write(file.toPath(), sb.toString().getBytes());

    return sb.toString();
  }

}
