package src;

public class GPParameters {
    public static final int POPULATION_SIZE = 200;
    public static final int MAX_GENERATIONS = 100;
    public static final int MAX_TREE_DEPTH = 5;
    public static final int INITIAL_TREE_DEPTH = 3;
    public static final double CROSSOVER_RATE = 0.85;
    public static final double MUTATION_RATE = 0.15;
    public static final double ELITISM_RATE = 0.05;
    public static final int TOURNAMENT_SIZE = 5;
    public static final String[] FEATURES = {"age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"};
    public static final int NUM_FEATURES = 9;
    public static final double MIN_CONSTANT = -5.0;
    public static final double MAX_CONSTANT = 5.0;
    public static final double CLASSIFICATION_THRESHOLD = 0.0;
    public static final int NUM_RUNS = 30;
}
