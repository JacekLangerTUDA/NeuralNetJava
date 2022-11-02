package neural.network.linearalgebra.exception;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 27.10.2022</p>
 */
public class IllegalMathOperationException extends IllegalArgumentException {

  public IllegalMathOperationException(String s, Object... args) {

    super(String.format(s, args));
  }

}
