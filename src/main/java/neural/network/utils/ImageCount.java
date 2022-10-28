package neural.network.utils;

/**
 * Count Enum to pass image count to train function. used for unit testing..
 *
 * <p>Created by: Jack</p>
 * <p>Date: 28.10.2022</p>
 */
public enum ImageCount {
  ALL(60000),
  ONE(1);

  private int count;

  public int getCount() {

    return count;
  }

  ImageCount(int count) {

    this.count = count;
  }
}
