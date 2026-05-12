package src;

import java.util.*;

public class ArithmeticGP {
    private Random random;
    private GPIndividual best;
    private Dataset trainData, testData;
    private long runtime;
    private double trainAcc, testAcc, fmeasure;
    
    private static final String[] FUNCTIONS = {"+", "-", "*", "/"};
    
    public ArithmeticGP(long seed) { 
        this.random = new Random(seed);
    }
    
    public void train(String trainFile, String testFile) throws Exception {
        long start = System.currentTimeMillis();
        
        // Load data
        trainData = new Dataset();
        testData = new Dataset();
        trainData.load(trainFile);
        testData.load(testFile);
        
        System.out.println("Training data size: " + trainData.size());
        System.out.println("Test data size: " + testData.size());
        System.out.println();
        
        // Initialize population
        List<GPIndividual> population = new ArrayList<>();
        for (int i = 0; i < GPParameters.POPULATION_SIZE; i++) {
            GPIndividual ind = new GPIndividual(growTree(GPParameters.INITIAL_TREE_DEPTH));
            population.add(ind);
        }
        
        best = null;
        
        // Evolution loop
        for (int gen = 0; gen < GPParameters.MAX_GENERATIONS; gen++) {
            // Evaluate fitness for all individuals
            for (GPIndividual ind : population) {
                double[] predictions = evaluate(ind, trainData);
                double acc = Metrics.accuracy(predictions, trainData.getLabelArray());
                double fm = Metrics.fMeasure(predictions, trainData.getLabelArray());
                double fitness = (acc + fm) / 2.0;  // Average of accuracy and F-measure
                ind.setFitness(fitness);
                ind.setPredictions(predictions);
            }
            
            // Sort by fitness (best first)
            Collections.sort(population);
            
            // Update best individual
            GPIndividual currentBest = population.get(0);
            if (best == null || currentBest.getFitness() > best.getFitness()) {
                best = currentBest.copy();
            }
            
            // Print progress every 10 generations
            if (gen % 10 == 0) {
                System.out.printf("Gen %d: Best Fitness = %.4f (Train Acc: %.4f, F1: %.4f)%n",
                    gen, best.getFitness(), 
                    Metrics.accuracy(evaluate(best, trainData), trainData.getLabelArray()),
                    Metrics.fMeasure(evaluate(best, trainData), trainData.getLabelArray()));
            }
            
            // Create new population
            List<GPIndividual> newPopulation = new ArrayList<>();
            
            // Elitism - keep best individuals
            int eliteCount = (int)(GPParameters.POPULATION_SIZE * GPParameters.ELITISM_RATE);
            for (int i = 0; i < eliteCount; i++) {
                newPopulation.add(population.get(i).copy());
            }
            
            // Fill rest with offspring
            while (newPopulation.size() < GPParameters.POPULATION_SIZE) {
                GPIndividual parent1 = tournamentSelect(population);
                GPIndividual parent2 = tournamentSelect(population);
                
                GPIndividual offspring;
                if (random.nextDouble() < GPParameters.CROSSOVER_RATE) {
                    offspring = crossover(parent1, parent2);
                } else {
                    offspring = parent1.copy();
                }
                
                mutate(offspring);
                newPopulation.add(offspring);
            }
            
            population = newPopulation;
        }
        
        // Final evaluation
        double[] finalTrainPred = evaluate(best, trainData);
        double[] finalTestPred = evaluate(best, testData);
        
        trainAcc = Metrics.accuracy(finalTrainPred, trainData.getLabelArray());
        testAcc = Metrics.accuracy(finalTestPred, testData.getLabelArray());
        fmeasure = Metrics.fMeasure(finalTestPred, testData.getLabelArray());
        
        runtime = System.currentTimeMillis() - start;
        
        // Print final best individual
        System.out.println("\n=== BEST EVOLVED EXPRESSION ===");
        System.out.println(best.getRoot().toString());
        System.out.println();
    }
    
    private GPIndividual.Node growTree(int maxDepth) {
        if (maxDepth <= 1 || random.nextDouble() < 0.3) {
            // Terminal node (70% feature, 30% constant)
            if (random.nextDouble() < 0.7) {
                int featureIdx = random.nextInt(GPParameters.NUM_FEATURES);
                return new GPIndividual.TerminalNode(featureIdx);
            } else {
                double constant = GPParameters.MIN_CONSTANT + 
                    random.nextDouble() * (GPParameters.MAX_CONSTANT - GPParameters.MIN_CONSTANT);
                return new GPIndividual.TerminalNode(constant);
            }
        } else {
            // Function node
            String op = FUNCTIONS[random.nextInt(FUNCTIONS.length)];
            GPIndividual.Node left = growTree(maxDepth - 1);
            GPIndividual.Node right = growTree(maxDepth - 1);
            return new GPIndividual.FunctionNode(op, left, right);
        }
    }
    
    private double[] evaluate(GPIndividual ind, Dataset data) {
        double[][] features = data.getFeatureMatrix();
        double[] predictions = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            try {
                double val = ind.getRoot().evaluate(features[i]);
                predictions[i] = val;
            } catch (Exception e) {
                predictions[i] = 0;
            }
        }
        return predictions;
    }
    
    private GPIndividual tournamentSelect(List<GPIndividual> population) {
        GPIndividual bestInd = null;
        for (int i = 0; i < GPParameters.TOURNAMENT_SIZE; i++) {
            int idx = random.nextInt(population.size());
            GPIndividual contender = population.get(idx);
            if (bestInd == null || contender.getFitness() > bestInd.getFitness()) {
                bestInd = contender;
            }
        }
        return bestInd.copy();
    }
    
    private GPIndividual crossover(GPIndividual parent1, GPIndividual parent2) {
        GPIndividual offspring = parent1.copy();
        GPIndividual.Node subtree = getRandomSubtree(parent2.getRoot());
        replaceRandomSubtree(offspring.getRoot(), subtree);
        return offspring;
    }
    
    private GPIndividual.Node getRandomSubtree(GPIndividual.Node node) {
        List<GPIndividual.Node> nodes = new ArrayList<>();
        collectNodes(node, nodes);
        int idx = random.nextInt(nodes.size());
        return nodes.get(idx).copy();
    }
    
    private void collectNodes(GPIndividual.Node node, List<GPIndividual.Node> nodes) {
        nodes.add(node);
        if (node instanceof GPIndividual.FunctionNode) {
            GPIndividual.FunctionNode fn = (GPIndividual.FunctionNode) node;
            collectNodes(fn.left, nodes);
            if (fn.right != null) {
                collectNodes(fn.right, nodes);
            }
        }
    }
    
    private void replaceRandomSubtree(GPIndividual.Node root, GPIndividual.Node subtree) {
        List<GPIndividual.Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        if (nodes.isEmpty()) return;
        
        int idx = random.nextInt(nodes.size());
        GPIndividual.Node target = nodes.get(idx);
        
        if (target instanceof GPIndividual.FunctionNode) {
            GPIndividual.FunctionNode fn = (GPIndividual.FunctionNode) target;
            if (random.nextBoolean()) {
                fn.left = subtree;
            } else if (fn.right != null) {
                fn.right = subtree;
            }
        }
    }
    
    private void mutate(GPIndividual individual) {
        if (random.nextDouble() < GPParameters.MUTATION_RATE) {
            GPIndividual.Node newSubtree = growTree(GPParameters.MAX_TREE_DEPTH / 2);
            replaceRandomSubtree(individual.getRoot(), newSubtree);
        }
    }
    
    public double getTrainingAccuracy() { return trainAcc; }
    public double getTestAccuracy() { return testAcc; }
    public double getFMeasure() { return fmeasure; }
    public long getRuntime() { return runtime; }
}
