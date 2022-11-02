package neural.network.math;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 31.10.2022</p>
 */
public abstract class Benchmark {

  private long start;
  private long stop;

  protected long getTimeInNs() {

    return stop - start;
  }

  protected void start() {

    this.start = System.nanoTime();
  }

  protected void stop() {

    this.stop = System.nanoTime();
  }

}
