package neural.network.math;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 31.10.2022</p>
 */
public class MathBenchmark extends Benchmark {

  @Test
  public void benchmarkAddition() {

    var a = 25;
    var b = 26;

    start();
    add(a, add(a, (add(a, (add(a, add(a, add(a, (add(a, (add(a, add(a, add(a, (add(a, (add(a, add(a,
                                                                                                  add(a,
                                                                                                      (add(
                                                                                                          a,
                                                                                                          (add(
                                                                                                              a,
                                                                                                              b))))))))))))))))))))))));
    stop();

    System.out.println(getTimeInNs());

    start();
    bitwiseAdd(a, bitwiseAdd(a, (bitwiseAdd(a, (bitwiseAdd(a, bitwiseAdd(a, bitwiseAdd(a,
                                                                                       (bitwiseAdd(
                                                                                           a,
                                                                                           (bitwiseAdd(
                                                                                               a,
                                                                                               bitwiseAdd(
                                                                                                   a,
                                                                                                   bitwiseAdd(
                                                                                                       a,
                                                                                                       (bitwiseAdd(
                                                                                                           a,
                                                                                                           (bitwiseAdd(
                                                                                                               a,
                                                                                                               bitwiseAdd(
                                                                                                                   a,
                                                                                                                   bitwiseAdd(
                                                                                                                       a,
                                                                                                                       (bitwiseAdd(
                                                                                                                           a,
                                                                                                                           (bitwiseAdd(
                                                                                                                               a,
                                                                                                                               b))))))))))))))))))))))));
    stop();

    System.out.println(getTimeInNs());
  }


  /**
   * Test for {@link }.
   **/
  @Test
  /*default*/ void testMethod() {

    assertEquals(39, bitwiseAdd(25, 14));
  }

  int bitwiseAdd(int a, int b) {

    if ((a & b) == 0) {
      return a ^ b;
    }

    int c = a & b;
    a = a ^ b;
    b = c << 1;

    return bitwiseAdd(a, b);
  }

  int add(int a, int b) {

    return a + b;
  }

}
