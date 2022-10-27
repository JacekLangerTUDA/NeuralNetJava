package neural.network.models;

import java.util.ArrayList;
import java.util.List;


/**
 * Datastructure for saving weights.
 *
 * <p>Created by: Jack</p>
 * <p>Date: 27.10.2022</p>
 */
public class NodesList {

  List<Layer> layers;

  public List<Layer> getLayers() {

    return layers;
  }

  public void add(Layer layer) {

    if (layers == null) {
      layers = new ArrayList<>();
    }
    layers.add(layer);
  }

  public double[][] get(int i) {

    return layers.get(i).weightsMatrix;
  }

}
