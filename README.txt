================================================================================
  GPClassifier - Genetic Programming for Breast Cancer Classification
  COS314 Assignment 3 | University of Pretoria
================================================================================

GROUP MEMBERS
  - Gift        : DecisionTreeGP.java
  - Christopher : ArithmeticGP.java, GPParameters.java, GPIndividual.java,
                  Dataset.java, Metrics.java, WilcoxonTest.java, Main.java
  - Tiego : manifest.txt, README.txt, report.pdf, GPClassifier.jar

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
This project implements two Genetic Programming (GP) approaches for classifying
the Breast Cancer Wisconsin dataset:

  1. Arithmetic GP  - Evolves symbolic arithmetic expressions (e.g., +, -, *, /)
                      that compute a score; instances are classified based on
                      whether the score exceeds a threshold.

  2. Decision Tree GP - Evolves binary decision trees where each internal node
                        splits on a feature threshold and leaves predict a class.

Both algorithms are evaluated over 30 independent runs. A Wilcoxon signed-rank
test is used to determine statistical significance between the two approaches.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
  - Java Runtime Environment (JRE) 11 or later
  - No external libraries required (self-contained JAR)

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------
  GPClassifier.jar          Executable JAR (main entry point: src.Main)
  README.txt                This file
  report.pdf                Full assignment report
  manifest.txt              JAR manifest (Main-Class: src.Main)
  data/
    Breast_train.csv        Training dataset (183 instances)
    Breast_test.csv         Test dataset (86 instances)
  src/
    Main.java               Entry point - runs 30 independent runs of both GPs
    ArithmeticGP.java       Symbolic/Arithmetic GP implementation
    DecisionTreeGP.java     Decision Tree GP implementation
    GPIndividual.java       Individual representation (tree nodes)
    GPParameters.java       Shared hyperparameter constants
    Dataset.java            CSV data loader
    Metrics.java            Accuracy and F-measure calculations
    WilcoxonTest.java       Wilcoxon signed-rank statistical test
    TestArithmeticGP.java   Standalone test runner for Arithmetic GP only

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

  OPTION 1 - Run both GP algorithms (30 runs each, with Wilcoxon test):
  -----------------------------------------------------------------------
  Place Breast_train.csv and Breast_test.csv inside a folder named "data/"
  in the same directory as GPClassifier.jar, then run:

      java -jar GPClassifier.jar

  You will be prompted:
      Enter seed value: <type an integer, e.g. 42>

  Output: Per-run accuracy for both GPs, mean accuracies, and Wilcoxon result.

  OPTION 2 - Run Arithmetic GP only (single run, verbose output):
  -----------------------------------------------------------------------
  Compile and run TestArithmeticGP directly, or extract and run:

      javac -d . src/*.java
      java src.TestArithmeticGP

  You will be prompted for a seed value. This mode prints the best evolved
  expression and generation-by-generation fitness progress.

  OPTION 3 - Compile from source:
  -----------------------------------------------------------------------
      javac -d . src/*.java
      java src.Main

--------------------------------------------------------------------------------
REPLICATING REPORTED RESULTS
--------------------------------------------------------------------------------
  Seed used in the report: 12345

  To replicate:
      java -jar GPClassifier.jar
      Enter seed value: 12345

  Each of the 30 runs uses seeds 12345, 12346, 12347, ... 12374 sequentially.

--------------------------------------------------------------------------------
GP PARAMETERS (as configured in GPParameters.java)
--------------------------------------------------------------------------------
  Population size       : 200
  Max generations       : 100
  Initial tree depth    : 3
  Max tree depth        : 5
  Crossover rate        : 85%
  Mutation rate         : 15%
  Elitism rate          : 5%
  Tournament size       : 5
  Selection method      : Tournament selection
  Mutation type         : Point mutation (subtree replacement)
  Fitness function      : (Accuracy + F-measure) / 2

--------------------------------------------------------------------------------
DATASET
--------------------------------------------------------------------------------
  Source : Breast Cancer Wisconsin (Diagnostic) Dataset
  Train  : 183 instances
  Test   : 86 instances
  Class  : 0 = no-recurrence-events, 1 = recurrence-events
  Features (9):
    age, menopause, tumor_size, inv_nodes, node_caps,
    deg_malig, breast, breast_quad, irradiat

  All features are pre-encoded as integers (see Appendix in report.pdf).

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
  - The JAR expects data files at the relative path: data/Breast_train.csv
    and data/Breast_test.csv. Ensure this folder exists alongside the JAR.
  - All random operations are seeded for full reproducibility.
  - Runtime will vary by machine; reported runtimes were measured on a
    standard laptop (Intel Core i5, 8GB RAM).

================================================================================