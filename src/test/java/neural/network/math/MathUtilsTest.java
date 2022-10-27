package neural.network.math;

import static java.lang.Math.E;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class MathUtilsTest {

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
   * Test for {@link MathUtils#trans(double[][])}.
   **/
  @Test
  /*default*/ void testTrans() {

    double[][] orig = { { 1, 2, 5 }, { 3, 4, 1 } };
    double[][] expected = { { 1, 3 }, { 2, 4 }, { 5, 1 } };

    assertArrayEquals(expected, MathUtils.trans(orig));
  }

}