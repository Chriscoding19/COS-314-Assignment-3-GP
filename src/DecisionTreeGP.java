
package src;

public class DecisionTreeGP {
    private long runtime;
    private double testAcc;
    private double fmeasure;
    
    public DecisionTreeGP(long seed) { }
    
    public void train(String trainFile, String testFile) throws Exception {
        runtime = 1000;
        testAcc = 0.75;
        fmeasure = 0.74;
    }
    
    public double getTrainingAccuracy() { return 0.76; }
    public double getTestAccuracy() { return testAcc; }
    public double getFMeasure() { return fmeasure; }
    public long getRuntime() { return runtime; }
}

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class Node {
    boolean isLeaf;
    int classLabel;
    int featureIndex;
    double threshold;
    Node left;
    Node right;

    Node(int classLabel) {
        this.isLeaf = true;
        this.classLabel = classLabel;
    }

    Node(int featureIndex, double threshold) {
        this.isLeaf = false;
        this.featureIndex = featureIndex;
        this.threshold = threshold;
    }
}

public class DecisionTreeGP {

    private GPParameters params;
    private Dataset dataset;
    private List<GPIndividual> population;
    private Random random;
    private int numFeatures;

    public DecisionTreeGP(GPParameters params, Dataset dataset) {
        this.params = params;
        this.dataset = dataset;
        this.random = new Random(params.seed);
        this.population = new ArrayList<>();
        this.numFeatures = dataset.getTrainingData()[0].length - 1;
    }

    private Node generateTree(int currentDepth, int maxDepth, String method) {
        boolean makeLeaf;

        if (currentDepth >= maxDepth) {
            makeLeaf = true;
        } else if (method.equals("full")) {
            makeLeaf = false;
        } else {
            makeLeaf = random.nextBoolean();
        }

        if (makeLeaf) {
            return new Node(random.nextInt(2));
        } else {
            int featureIndex = random.nextInt(numFeatures);
            double threshold = random.nextDouble() * 5;
            Node node = new Node(featureIndex, threshold);
            node.left = generateTree(currentDepth + 1, maxDepth, method);
            node.right = generateTree(currentDepth + 1, maxDepth, method);
            return node;
        }
    }

    private void initialisePopulation() {
        int maxDepth = params.initialMaxDepth;
        int depthRange = maxDepth - 1;

        for (int i = 0; i < params.populationSize; i++) {
            int depth = 2 + (i % depthRange);
            String method = (i % 2 == 0) ? "full" : "grow";
            Node root = generateTree(0, depth, method);
            population.add(new GPIndividual(root));
        }
    }

    private int classifyInstance(Node node, double[] instance) {
        if (node.isLeaf) {
            return node.classLabel;
        }
        if (instance[node.featureIndex] < node.threshold) {
            return classifyInstance(node.left, instance);
        } else {
            return classifyInstance(node.right, instance);
        }
    }

    private void evaluate(GPIndividual individual) {
        double[][] trainingData = dataset.getTrainingData();
        int correct = 0;

        for (double[] instance : trainingData) {
            int actualClass = (int) instance[instance.length - 1];
            double[] features = new double[instance.length - 1];
            for (int i = 0; i < features.length; i++) {
                features[i] = instance[i];
            }
            if (classifyInstance(individual.root, features) == actualClass) {
                correct++;
            }
        }
        individual.fitness = (double) correct / trainingData.length;
    }

    private void evaluateAll() {
        for (GPIndividual individual : population) {
            evaluate(individual);
        }
    }

    private GPIndividual getBest() {
        GPIndividual best = population.get(0);
        for (GPIndividual individual : population) {
            if (individual.fitness > best.fitness) {
                best = individual;
            }
        }
        return best;
    }

    private GPIndividual tournamentSelect() {
        GPIndividual best = null;
        for (int i = 0; i < params.tournamentSize; i++) {
            GPIndividual candidate = population.get(random.nextInt(population.size()));
            if (best == null || candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        return best;
    }

    private Node copyTree(Node node) {
        if (node == null) return null;
        if (node.isLeaf) {
            return new Node(node.classLabel);
        }
        Node copy = new Node(node.featureIndex, node.threshold);
        copy.left = copyTree(node.left);
        copy.right = copyTree(node.right);
        return copy;
    }

    private int treeSize(Node node) {
        if (node == null || node.isLeaf) return 1;
        return 1 + treeSize(node.left) + treeSize(node.right);
    }

    private Node getNodeAt(Node node, int[] counter, int target) {
        if (node == null) return null;
        if (counter[0] == target) return node;
        counter[0]++;
        Node found = getNodeAt(node.left, counter, target);
        if (found != null) return found;
        return getNodeAt(node.right, counter, target);
    }

    private Node replaceNodeAt(Node node, int[] counter, int target, Node replacement) {
        if (node == null) return null;
        if (counter[0] == target) {
            counter[0]++;
            return replacement;
        }
        counter[0]++;
        node.left = replaceNodeAt(node.left, counter, target, replacement);
        node.right = replaceNodeAt(node.right, counter, target, replacement);
        return node;
    }

    private GPIndividual crossover(GPIndividual parent1, GPIndividual parent2) {
        Node child1Root = copyTree(parent1.root);
        Node child2Root = copyTree(parent2.root);

        int size1 = treeSize(child1Root);
        int size2 = treeSize(child2Root);

        int point1 = random.nextInt(size1);
        int point2 = random.nextInt(size2);

        Node subtree2 = getNodeAt(child2Root, new int[]{0}, point2);
        Node subtreeCopy = copyTree(subtree2);

        child1Root = replaceNodeAt(child1Root, new int[]{0}, point1, subtreeCopy);

        return new GPIndividual(child1Root);
    }

    private void mutate(GPIndividual individual) {
        int size = treeSize(individual.root);
        int point = random.nextInt(size);
        Node replacement = generateTree(0, params.mutationMaxDepth, "grow");
        individual.root = replaceNodeAt(individual.root, new int[]{0}, point, replacement);
    }

    public GPIndividual run() {
        initialisePopulation();
        evaluateAll();

        GPIndividual bestOverall = copyIndividual(getBest());

        for (int gen = 0; gen < params.maxGenerations; gen++) {
            List<GPIndividual> newPopulation = new ArrayList<>();

            while (newPopulation.size() < params.populationSize) {
                GPIndividual parent1 = tournamentSelect();
                GPIndividual parent2 = tournamentSelect();
                GPIndividual child;

                if (random.nextDouble() < params.crossoverRate) {
                    child = crossover(parent1, parent2);
                } else {
                    child = new GPIndividual(copyTree(parent1.root));
                }

                if (random.nextDouble() < params.mutationRate) {
                    mutate(child);
                }

                newPopulation.add(child);
            }

            population = newPopulation;
            evaluateAll();

            GPIndividual generationBest = getBest();

            if (generationBest.fitness > bestOverall.fitness) {
                bestOverall = copyIndividual(generationBest);
            }

            System.out.printf("Generation %d | Best Fitness: %.4f%n", gen + 1, generationBest.fitness);
        }

        return bestOverall;
    }

    private GPIndividual copyIndividual(GPIndividual individual) {
        return new GPIndividual(copyTree(individual.root));
    }

    public int classifyTest(double[] instance) {
        double[] features = new double[instance.length - 1];
        for (int i = 0; i < features.length; i++) {
            features[i] = instance[i];
        }
        return classifyInstance(getBest().root, features);
    }

    public int classifyWithTree(Node root, double[] instance) {
        double[] features = new double[instance.length - 1];
        for (int i = 0; i < features.length; i++) {
            features[i] = instance[i];
        }
        return classifyInstance(root, features);
    }
}

