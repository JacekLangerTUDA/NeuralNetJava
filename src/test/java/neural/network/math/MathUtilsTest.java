package neural.network.math;

import static java.lang.Math.E;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import neural.network.models.NodesList;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

class MathUtilsTest {

  static double[][] bigmatrix = new double[784][392];

  @BeforeAll
  public static void getString() {

    try {
      String json = Files.readString(Path.of("src/test/resources/weights.json"));
      bigmatrix = new Gson().fromJson(json, NodesList.class).get(0);

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * ${END}.
   **/
  @Test
  /* default */ void sigmoid() {

    assertEquals(0.5, Math.round(MathUtils.sigmoid(0) * 10) / 10.0);
    assertEquals(1 / (1 + E), MathUtils.sigmoid(-1));
  }

  /**
   * ${END}.
   **/
  @Test
  /* default */ void sum() {

    double[] arr = { 1, 2, 3, 4, 5, 6 };
    assertEquals(21, MathUtils.sum(arr));
  }


  /**
   * Test for {@link MathUtils#mMult(double[][], double[][])}.
   **/
  @Test
  /*default*/ void testMMult() {

    double[][] first = { { 1, 2, 3 }, { 2, 3, 1 } };
    double[][] second = { { 1, 3 }, { 2, 4 }, { 5, 1 } };
    double[][] expected = { { 20, 14 }, { 13, 19 } };

    assertArrayEquals(expected, MathUtils.mMult(first, second));
  }

  /**
   * Test for {@link MathUtils#mMult(double[][], double[][])}.
   **/
  @Test
  /*default*/ void testMMultInp() {

    double[] input = { 1, 2, 3 };
    double[][] weights = { { 1, 2, 5 }, { 3, 4, 1 } };
    double[] expected = { 20, 14 };

    assertArrayEquals(expected, MathUtils.mMult(weights, input));
  }


  /**
   * Test for {@link MathUtils#mMult(double[], double[])}.
   **/
  @Test
  /*default*/ void testMMultArr() {

    double[] fst = { 1, 2, 3 };
    double[] scnd = { 1, 2, 3 };
    double[][] expected = { { 1, 2, 3 }, { 2, 4, 6 }, { 3, 6, 9 } };

    assertArrayEquals(expected, MathUtils.mMult(fst, scnd));
  }


  /**
   * Test for {@link MathUtils#trans(double[][])}.
   **/
  @Test
  /*default*/ void testTrans() {

    double[][] orig = { { 1, 2, 5 }, { 3, 4, 1 } };
    double[][] expected = { { 1, 3 }, { 2, 4 }, { 5, 1 } };

    assertArrayEquals(expected, MathUtils.trans(orig));
  }

  /**
   * Test for {@link MathUtils#trans(double[][])}.
   **/
  @Test
  /*default*/ void testAdd() {

    double[][] fst = { { 1, 2, 3 }, { 2, 4, 6 }, { 3, 6, 9 } };
    double[][] scd = { { 1, 3, 0 }, { 2, 4, 0 }, { 5, 1, 3 } };
    double[][] expected = { { 2, 5, 3 }, { 4, 8, 6 }, { 8, 7, 12 } };

    assertArrayEquals(expected, MathUtils.mAdd(fst, scd));
  }

  /**
   * Test for {@link MathUtils#trans(double[][])}.
   **/
  @Test
  @RepeatedTest(10)
  /*default*/ void stressDuration() {

    long avg = 0;
    for (int i = 0; i < 10; i++) {
      var start = System.nanoTime();
      MathUtils.mAdd(bigmatrix, bigmatrix);
      var end = System.nanoTime();
      long diff = end - start;
      System.out.println(String.format("time add: %s ns", diff));
      avg += diff;
    }
    System.out.println(String.format("avg. time add: %s ns", avg / 10.0));

    avg = 0;
    for (int i = 0; i < 10; i++) {

      var bt = MathUtils.trans(bigmatrix);
      long start = System.currentTimeMillis();
      MathUtils.mMult(bigmatrix, bt);
      long end = System.currentTimeMillis();

      long diff = end - start;
      System.out.println(String.format("time mult: %s ms", diff));

      avg += diff;
    }

    System.out.println(String.format("avg. time mult: %s ms", avg / 10.0));
  }


}