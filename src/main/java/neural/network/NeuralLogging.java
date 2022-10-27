package neural.network;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * .
 *
 * <p>Created by: Jack</p>
 * <p>Date: 27.10.2022</p>
 */
public abstract class NeuralLogging {

  private static final Logger LOGGER = Logger.getLogger(NeuralLogging.class.getSimpleName());

  public static void error(String msg, Object... args) {

    LOGGER.log(Level.SEVERE, String.format(msg, args));
  }


  public static void debug(String msg, Object... args) {

    LOGGER.log(Level.WARNING, String.format(msg, args));
  }

  public static void info(String msg, Object... args) {

    LOGGER.log(Level.INFO, String.format(msg, args));
  }

}
