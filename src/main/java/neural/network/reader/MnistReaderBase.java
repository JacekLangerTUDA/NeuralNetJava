package neural.network.reader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import neural.network.reader.enumerations.MnistFile;

/**
 * Base class of the MnistReader.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 10.10.2022</p>
 */
public abstract class MnistReaderBase implements MnistReadOnly {

  /**
   * Base class to read image file data.
   *
   * @param index the index where to begin reading
   * @return an array of byte data representing the image
   */
  protected double[] readImage(int index, MnistFile mnistFile) {

    int start = mnistFile.getStart();
    short size = 28 * 28;
    int offset = index * size + start;
    double[] buffer = new double[size];
    File file = mnistFile.getPath().toFile();

    if (file.exists()) {
      try (var stream = Files.newInputStream(MnistFile.TRAIN_IMAGE.getPath())) {

        var bytes = stream.readAllBytes();

        int j = 0;
        for (int i = offset; i < size + offset; i++) {
          buffer[j++] = (double) (bytes[i] & 0xff) / 254; // convert to unsigned byte and normalize
        }

      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    return buffer;
  }


  protected byte readLabel(int index, MnistFile mnistFile) {

    var start = mnistFile.getStart();

    var offset = start + index;
    byte label = 0;
    try (var stream = Files.newInputStream(mnistFile.getPath())) {
      label = stream.readAllBytes()[offset];
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return label;
  }

}
