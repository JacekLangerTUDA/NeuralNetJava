package neural.network.reader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MnistReaderTest {

  private MnistReader reader = new MnistReader();

  /**
   * Test for {@link MnistReader#readTrainImage(int)}.
   **/
  @Test
  /* default */ void readTrainImage() {

    var actual = reader.readTrainImage(0);
    assertTrue(actual.length == 784);
  }

  /**
   * test for {@link MnistReader#readTrainLabel(int)}.
   **/
  @Test
  /* default */ void readTrainLabel() {

    var label = reader.readTrainLabel(0);
    assertEquals(label, 7);
  }


  /**
   * test for {@link MnistReader#fetchWeights(int, int, int)}.
   **/
  @Test
  /* default */ void fetchWeights() {

    var data = reader.fetchWeights(0, 784, 392);

    assertEquals(784, data.length);
    assertEquals(392, data[0].length);
  }

}