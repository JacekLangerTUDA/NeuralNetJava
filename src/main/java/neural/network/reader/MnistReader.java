package neural.network.reader;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
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
