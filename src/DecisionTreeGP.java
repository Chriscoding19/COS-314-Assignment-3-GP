package src;

import java.util.*;

public class DecisionTreeGP {
    private Random random;
    private long seed;
    private Node bestTree;
    private Dataset trainData;
    private Dataset testData;
    private long runtime;
    private double trainAcc;
    private double testAcc;
    private double fmeasure;
    
    private List<GPIndividual> population;
    
    private int populationSize = 200;
    private int maxGenerations = 100;
    private int maxTreeDepth = 5;
    private int initialMaxDepth = 3;
    private double crossoverRate = 0.85;
    private double mutationRate = 0.15;
    private double elitismRate = 0.05;
    private int tournamentSize = 5;
    private int numFeatures = 9;
    
    public DecisionTreeGP(long seed) {
        this.seed = seed;
        this.random = new Random(seed);
        this.population = new ArrayList<>();
    }
    
    public void train(String trainFile, String testFile) throws Exception {
        long startTime = System.currentTimeMillis();
        
        trainData = new Dataset();
        testData = new Dataset();
        trainData.load(trainFile);
        testData.load(testFile);
        
        initializePopulation();
        
        for (int generation = 0; generation < maxGenerations; generation++) {
            evaluatePopulation();
            Collections.sort(population);
            
            GPIndividual currentBest = population.get(0);
            if (bestTree == null || currentBest.fitness > getFitness(bestTree, trainData)) {
                bestTree = copyTree(currentBest.root);
            }
            
            List<GPIndividual> newPopulation = new ArrayList<>();
            
            int eliteCount = (int)(populationSize * elitismRate);
            for (int i = 0; i < eliteCount && i < population.size(); i++) {
                newPopulation.add(new GPIndividual(copyTree(population.get(i).root)));
            }
            
            while (newPopulation.size() < populationSize) {
                GPIndividual parent1 = tournamentSelect();
                GPIndividual parent2 = tournamentSelect();
                
                GPIndividual offspring;
                if (random.nextDouble() < crossoverRate) {
                    offspring = crossover(parent1, parent2);
                } else {
                    offspring = new GPIndividual(copyTree(parent1.root));
                }
                
                mutate(offspring);
                newPopulation.add(offspring);
            }
            
            population = newPopulation;
        }
        
        trainAcc = evaluateAccuracy(bestTree, trainData);
        testAcc = evaluateAccuracy(bestTree, testData);
        fmeasure = evaluateFMeasure(bestTree, testData);
        
        runtime = System.currentTimeMillis() - startTime;
    }
    
    public double getTrainingAccuracy() { return trainAcc; }
    public double getTestAccuracy() { return testAcc; }
    public double getFMeasure() { return fmeasure; }
    public long getRuntime() { return runtime; }
    
    private void initializePopulation() {
        population.clear();
        int depthRange = initialMaxDepth - 1;
        
        for (int i = 0; i < populationSize; i++) {
            int depth = 2 + (i % depthRange);
            String method = (i % 2 == 0) ? "full" : "grow";
            Node tree = generateTree(0, depth, method);
            population.add(new GPIndividual(tree));
        }
    }
    
    private Node generateTree(int currentDepth, int maxDepth, String method) {
        boolean makeLeaf;
        
        if (currentDepth >= maxDepth) {
            makeLeaf = true;
        } else if (method.equals("full")) {
            makeLeaf = false;
        } else {
            makeLeaf = random.nextDouble() < 0.4;
        }
        
        if (makeLeaf) {
            return new Node(random.nextInt(2));
        } else {
            int featureIndex = random.nextInt(numFeatures);
            double threshold = getRandomThresholdForFeature(featureIndex);
            Node node = new Node(featureIndex, threshold);
            node.left = generateTree(currentDepth + 1, maxDepth, method);
            node.right = generateTree(currentDepth + 1, maxDepth, method);
            return node;
        }
    }
    
    private double getRandomThresholdForFeature(int featureIndex) {
        double[][] ranges = {
            {0, 5}, {0, 2}, {0, 10}, {0, 30}, {0, 2}, {0, 3}, {0, 1}, {0, 5}, {0, 1}
        };
        
        if (featureIndex < ranges.length) {
            return ranges[featureIndex][0] + random.nextDouble() * (ranges[featureIndex][1] - ranges[featureIndex][0]);
        }
        return random.nextDouble() * 5.0;
    }
    
    private int classifyInstance(Node node, double[] features) {
        if (node.isLeaf) {
            return node.classLabel;
        }
        if (features[node.featureIndex] <= node.threshold) {
            return classifyInstance(node.left, features);
        } else {
            return classifyInstance(node.right, features);
        }
    }
    
    private void evaluatePopulation() {
        for (GPIndividual individual : population) {
            double[] predictions = getPredictions(individual.root, trainData);
            double acc = Metrics.accuracy(predictions, trainData.getLabelArray());
            double f1 = Metrics.fMeasure(predictions, trainData.getLabelArray());
            individual.fitness = (acc + f1) / 2.0;
        }
    }
    
    private double getFitness(Node tree, Dataset data) {
        double[] predictions = getPredictions(tree, data);
        double acc = Metrics.accuracy(predictions, data.getLabelArray());
        double f1 = Metrics.fMeasure(predictions, data.getLabelArray());
        return (acc + f1) / 2.0;
    }
    
    private double[] getPredictions(Node tree, Dataset data) {
        double[][] features = data.getFeatureMatrix();
        double[] predictions = new double[features.length];
        
        for (int i = 0; i < features.length; i++) {
            predictions[i] = classifyInstance(tree, features[i]);
        }
        return predictions;
    }
    
    private double evaluateAccuracy(Node tree, Dataset data) {
        double[] predictions = getPredictions(tree, data);
        return Metrics.accuracy(predictions, data.getLabelArray());
    }
    
    private double evaluateFMeasure(Node tree, Dataset data) {
        double[] predictions = getPredictions(tree, data);
        return Metrics.fMeasure(predictions, data.getLabelArray());
    }
    
    private GPIndividual tournamentSelect() {
        GPIndividual best = null;
        for (int i = 0; i < tournamentSize; i++) {
            int idx = random.nextInt(population.size());
            GPIndividual contender = population.get(idx);
            if (best == null || contender.fitness > best.fitness) {
                best = contender;
            }
        }
        return new GPIndividual(copyTree(best.root));
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
    
    private int getTreeNodeCount(Node node) {
        if (node == null) return 0;
        if (node.isLeaf) return 1;
        return 1 + getTreeNodeCount(node.left) + getTreeNodeCount(node.right);
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
        Node childRoot = copyTree(parent1.root);
        Node parent2Root = copyTree(parent2.root);
        
        int size1 = getTreeNodeCount(childRoot);
        int size2 = getTreeNodeCount(parent2Root);
        
        if (size1 <= 1 || size2 <= 1) {
            return new GPIndividual(copyTree(parent1.root));
        }
        
        int point1 = random.nextInt(size1);
        int point2 = random.nextInt(size2);
        
        Node subtree = getNodeAt(parent2Root, new int[]{0}, point2);
        Node subtreeCopy = copyTree(subtree);
        
        childRoot = replaceNodeAt(childRoot, new int[]{0}, point1, subtreeCopy);
        
        return new GPIndividual(childRoot);
    }
    
    private void mutate(GPIndividual individual) {
        if (random.nextDouble() >= mutationRate) return;
        
        int size = getTreeNodeCount(individual.root);
        if (size <= 1) return;
        
        int point = random.nextInt(size);
        int newDepth = random.nextInt(maxTreeDepth) + 1;
        Node replacement = generateTree(0, newDepth, "grow");
        individual.root = replaceNodeAt(individual.root, new int[]{0}, point, replacement);
    }
    
    private static class Node {
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
    
    private static class GPIndividual implements Comparable<GPIndividual> {
        Node root;
        double fitness;
        
        GPIndividual(Node root) {
            this.root = root;
            this.fitness = 0.0;
        }
        
        @Override
        public int compareTo(GPIndividual other) {
            return Double.compare(other.fitness, this.fitness);
        }
    }
}