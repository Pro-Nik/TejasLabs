/* =====================================================
   COURSE-SPECIFIC ACADEMIC CHATBOT
   Frontend Only | GitHub Pages Safe
===================================================== */

let selectedCourse = null;

const chatWindow = document.getElementById("chatWindow");
const chatArea = document.getElementById("chatArea");
const courseTitle = document.getElementById("courseTitle");

/* ===============================
   UI MESSAGE HANDLER
================================ */

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `msg ${sender}`;
  div.innerText = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/* ===============================
   COURSE SELECTION
================================ */

function selectCourse(course) {
  selectedCourse = course;

  // Hide course cards
  document.getElementById("courseSelect").style.display = "none";

  // Show chat area
  chatArea.style.display = "block";

  // Set course title
  const courseNames = {
    ML: "📘 Machine Learning",
    DBMS: "📘 Database Management Systems",
    DL: "📘 Deep Learning",
    AI: "📘 Artificial Intelligence"
  };

  courseTitle.innerText = courseNames[course];

  addMessage(
    `You have selected ${courseNames[course]}.
Ask conceptual, algorithmic, or exam-oriented questions.`,
    "bot"
  );
}

/* ===============================
   QUESTION BANK
================================ */

const courseQA = {

  /* =====================================================
     MACHINE LEARNING (ML) – 20+ QUESTIONS
  ===================================================== */
 ML: [

/* ===================== BASIC CONCEPTS ===================== */

{
  keywords: ["machine learning","ml", "ml definition", "learning from data"],
  answer: `Machine Learning is a branch of Artificial Intelligence that enables systems to automatically learn patterns from data and improve performance without explicit programming.

ML algorithms build mathematical models using historical data and optimize them to make predictions or decisions. It plays a vital role in data-driven applications such as recommendation systems, fraud detection, and medical diagnosis.

📌 Suggestions:
• Strengthen math fundamentals
• Practice real-world datasets`
},

{
  keywords: ["types of machine learning", "supervised unsupervised", "learning paradigms"],
  answer: `Machine Learning is broadly classified into Supervised Learning, Unsupervised Learning, and Reinforcement Learning.

Supervised learning uses labeled data, unsupervised learning works with unlabeled data, while reinforcement learning learns via rewards and penalties.

📌 Suggestions:
• Compare use cases
• Understand data requirements`
},

{
  keywords: ["supervised learning", "labeled data", "classification regression"],
  answer: `Supervised learning is a paradigm where the training dataset contains input-output pairs. The model learns a mapping from inputs to known labels.

Common tasks include classification and regression. Algorithms include Linear Regression, Logistic Regression, SVM, and Decision Trees.

📌 Suggestions:
• Practice evaluation metrics
• Apply cross-validation`
},

{
  keywords: ["unsupervised learning", "unlabeled data", "clustering"],
  answer: `Unsupervised learning deals with datasets that do not have labeled outputs. The goal is to identify hidden patterns or structures in data.

Clustering and dimensionality reduction are key tasks. Algorithms include K-Means, DBSCAN, PCA, and Autoencoders.

📌 Suggestions:
• Visualize clusters
• Use distance metrics`
},

{
  keywords: ["reinforcement learning", "reward", "agent environment"],
  answer: `Reinforcement Learning is a learning paradigm where an agent interacts with an environment and learns optimal behavior through rewards and penalties.

It is widely used in robotics, gaming, and autonomous systems.

📌 Suggestions:
• Understand Markov Decision Processes
• Learn Q-learning`
},

/* ===================== MODEL BEHAVIOR ===================== */

{
  keywords: ["overfitting", "generalization error", "complex model"],
  answer: `Overfitting occurs when a model performs very well on training data but poorly on unseen data. It happens due to excessive model complexity or limited data.

Techniques such as regularization, pruning, dropout, and early stopping help reduce overfitting.

📌 Suggestions:
• Use validation sets
• Reduce model complexity`
},

{
  keywords: ["underfitting", "high bias", "simple model"],
  answer: `Underfitting occurs when a model is too simple to capture underlying patterns in the data, leading to poor performance on both training and test data.

📌 Suggestions:
• Increase model complexity
• Add relevant features`
},

{
  keywords: ["bias variance tradeoff", "bias variance", "model complexity"],
  answer: `The bias–variance tradeoff represents the balance between underfitting and overfitting.

High bias leads to underfitting, while high variance causes overfitting. A good model balances both.

📌 Suggestions:
• Use ensemble methods
• Tune hyperparameters`
},

/* ===================== REGRESSION ===================== */

{
  keywords: ["linear regression", "regression model", "continuous output"],
  answer: `Linear Regression models the relationship between a dependent variable and independent variables using a linear equation.

It assumes linearity, independence, and homoscedasticity.

📌 Suggestions:
• Analyze residuals
• Check assumptions`
},

{
  keywords: ["multiple linear regression", "multivariate regression", "many features"],
  answer: `Multiple Linear Regression extends linear regression by using multiple independent variables to predict a continuous target variable.

📌 Suggestions:
• Handle multicollinearity
• Normalize features`
},

{
  keywords: ["logistic regression", "binary classification", "sigmoid"],
  answer: `Logistic Regression is a classification algorithm that uses a sigmoid function to output probabilities for binary classification.

📌 Suggestions:
• Tune threshold
• Analyze ROC curve`
},

/* ===================== CLASSIFICATION ===================== */

{
  keywords: ["knn", "k nearest neighbors", "distance based"],
  answer: `K-Nearest Neighbors is a lazy learning algorithm that classifies a data point based on the majority class of its nearest neighbors.

📌 Suggestions:
• Choose optimal K
• Normalize data`
},

{
  keywords: ["decision tree", "tree based model", "entropy gini"],
  answer: `Decision Trees use hierarchical rules to split data based on feature values. They are interpretable but prone to overfitting.

📌 Suggestions:
• Prune trees
• Use ensemble models`
},

{
  keywords: ["random forest", "ensemble learning", "bagging"],
  answer: `Random Forest builds multiple decision trees using bootstrapped data and aggregates predictions to improve accuracy and reduce variance.

📌 Suggestions:
• Tune number of trees
• Analyze feature importance`
},

{
  keywords: ["naive bayes", "probabilistic classifier", "bayes theorem"],
  answer: `Naive Bayes is a probabilistic classifier based on Bayes’ theorem with an assumption of feature independence.

📌 Suggestions:
• Useful for text classification`
},

{
  keywords: ["svm", "support vector machine", "margin hyperplane"],
  answer: `Support Vector Machines classify data by finding an optimal hyperplane that maximizes margin between classes.

📌 Suggestions:
• Use kernel functions
• Tune C and gamma`
},

/* ===================== CLUSTERING ===================== */

{
  keywords: ["k means clustering", "centroid", "unsupervised clustering"],
  answer: `K-Means clusters data into K groups by minimizing intra-cluster variance.

📌 Suggestions:
• Use elbow method
• Normalize data`
},

{
  keywords: ["hierarchical clustering", "agglomerative", "dendrogram"],
  answer: `Hierarchical clustering builds nested clusters using either agglomerative or divisive approaches.

📌 Suggestions:
• Analyze dendrogram`
},

{
  keywords: ["dbscan", "density based clustering", "outlier detection"],
  answer: `DBSCAN groups points based on density and can detect outliers effectively.

📌 Suggestions:
• Tune eps and minPts`
},

/* ===================== FEATURE ENGINEERING ===================== */

{
  keywords: ["feature scaling", "normalization", "standardization"],
  answer: `Feature scaling ensures that numerical features contribute equally to model training.

📌 Suggestions:
• Use StandardScaler
• Normalize for distance-based models`
},

{
  keywords: ["feature selection", "dimensionality reduction", "important features"],
  answer: `Feature selection improves model performance by removing irrelevant or redundant features.

📌 Suggestions:
• Use correlation analysis
• Apply L1 regularization`
},

{
  keywords: ["pca", "principal component analysis", "dimension reduction"],
  answer: `PCA reduces dimensionality by transforming features into orthogonal components that preserve maximum variance.

📌 Suggestions:
• Analyze explained variance`
},

/* ===================== MODEL EVALUATION ===================== */

{
  keywords: ["accuracy", "classification metric", "evaluation"],
  answer: `Accuracy measures the proportion of correctly classified instances but can be misleading for imbalanced datasets.

📌 Suggestions:
• Use precision & recall`
},

{
  keywords: ["precision recall", "f1 score", "imbalanced data"],
  answer: `Precision measures correctness of positive predictions, while recall measures coverage of actual positives.

📌 Suggestions:
• Use F1-score for imbalance`
},

{
  keywords: ["confusion matrix", "tp fp fn tn", "classification analysis"],
  answer: `A confusion matrix summarizes classification results using true positives, false positives, false negatives, and true negatives.

📌 Suggestions:
• Analyze misclassification`
},

{
  keywords: ["roc curve", "auc", "probability threshold"],
  answer: `ROC curve plots TPR vs FPR, while AUC measures overall classifier performance.

📌 Suggestions:
• Compare models using AUC`
},

/* ===================== ADVANCED ===================== */

{
  keywords: ["ensemble learning", "bagging boosting", "model combination"],
  answer: `Ensemble learning combines multiple models to improve predictive performance.

📌 Suggestions:
• Study Random Forest & AdaBoost`
},

{
  keywords: ["boosting", "adaboost", "gradient boosting"],
  answer: `Boosting sequentially trains weak learners to correct previous errors.

📌 Suggestions:
• Learn XGBoost`
},

{
  keywords: ["hyperparameter tuning", "grid search", "random search"],
  answer: `Hyperparameter tuning optimizes model performance by searching optimal parameter values.

📌 Suggestions:
• Use cross-validation`
},

{
  keywords: ["cross validation", "k fold", "model validation"],
  answer: `Cross-validation evaluates model performance by dividing data into training and validation folds.

📌 Suggestions:
• Prefer k-fold CV`
},

{
  keywords: ["ml pipeline", "end to end model", "production ml"],
  answer: `An ML pipeline automates preprocessing, training, evaluation, and deployment steps.

📌 Suggestions:
• Use sklearn pipelines`
},

{
  keywords: ["ethical machine learning", "bias fairness", "responsible ai"],
  answer: `Ethical ML focuses on fairness, transparency, accountability, and bias reduction.

📌 Suggestions:
• Study responsible AI practices`
}

],


  /* =====================================================
     DBMS – 20+ QUESTIONS
  ===================================================== */
  DBMS: [

/* ===================== BASICS ===================== */

{
  keywords: ["dbms", "database management system", "data management"],
  answer: `A Database Management System (DBMS) is software that enables efficient storage, retrieval, and management of structured data.

It provides data abstraction, security, concurrency control, and integrity constraints. DBMS separates physical data storage from user applications, making systems scalable and reliable.

📌 Suggestions:
• Practice SQL queries
• Understand schema design`
},

{
  keywords: ["database", "data vs information", "structured data"],
  answer: `A database is an organized collection of related data stored electronically.

Data represents raw facts, while information is processed, meaningful data. DBMS helps convert data into information efficiently.

📌 Suggestions:
• Study real database schemas`
},

{
  keywords: ["dbms advantages", "file system vs dbms", "data redundancy"],
  answer: `DBMS overcomes limitations of file systems by reducing redundancy, improving consistency, supporting concurrency, and enforcing security.

File systems lack centralized control and efficient query mechanisms.

📌 Suggestions:
• Compare use cases`
},

/* ===================== ARCHITECTURE ===================== */

{
  keywords: ["three level architecture", "dbms architecture", "data abstraction"],
  answer: `The three-level architecture includes internal, conceptual, and external levels.

It ensures data independence and separates physical storage from logical representation.

📌 Suggestions:
• Understand logical vs physical independence`
},

{
  keywords: ["schema", "instance", "database schema"],
  answer: `A schema defines the logical structure of a database, while an instance represents the actual data at a given time.

📌 Suggestions:
• Study schema evolution`
},

{
  keywords: ["data independence", "logical physical independence", "dbms feature"],
  answer: `Data independence allows changes at one level of database architecture without affecting higher levels.

📌 Suggestions:
• Relate to real DBMS examples`
},

/* ===================== DATA MODELS ===================== */

{
  keywords: ["data models", "relational hierarchical network", "dbms models"],
  answer: `Data models define how data is structured and related.

Common models include hierarchical, network, relational, object-oriented, and NoSQL models.

📌 Suggestions:
• Focus on relational model`
},

{
  keywords: ["relational model", "tables tuples attributes", "edgar codd"],
  answer: `The relational model represents data using tables (relations) consisting of rows (tuples) and columns (attributes).

📌 Suggestions:
• Understand keys and constraints`
},

{
  keywords: ["er model", "entity relationship", "er diagram"],
  answer: `The ER model uses entities, attributes, and relationships to design databases conceptually.

📌 Suggestions:
• Practice ER diagrams`
},

{
  keywords: ["weak entity", "strong entity", "er diagram"],
  answer: `A weak entity depends on a strong entity for its existence and does not have a primary key.

📌 Suggestions:
• Identify partial keys`
},

/* ===================== KEYS ===================== */

{
  keywords: ["primary key", "candidate key", "unique identifier"],
  answer: `A primary key uniquely identifies each record in a table and cannot contain NULL values.

📌 Suggestions:
• Choose minimal attributes`
},

{
  keywords: ["foreign key", "referential integrity", "table relationship"],
  answer: `A foreign key establishes relationships between tables and enforces referential integrity.

📌 Suggestions:
• Use cascading actions carefully`
},

{
  keywords: ["super key", "candidate key", "key types"],
  answer: `A super key uniquely identifies tuples, while a candidate key is a minimal super key.

📌 Suggestions:
• Identify minimality`
},

{
  keywords: ["composite key", "multiple attributes", "primary key"],
  answer: `A composite key consists of two or more attributes that together uniquely identify a record.

📌 Suggestions:
• Avoid unnecessary complexity`
},

/* ===================== NORMALIZATION ===================== */

{
  keywords: ["normalization", "database design", "redundancy removal"],
  answer: `Normalization organizes data to reduce redundancy and improve data integrity.

It is based on functional dependencies and normal forms.

📌 Suggestions:
• Balance normalization and performance`
},

{
  keywords: ["first normal form", "1nf", "atomic values"],
  answer: `1NF ensures that each attribute contains atomic values and no repeating groups.

📌 Suggestions:
• Eliminate multi-valued attributes`
},

{
  keywords: ["second normal form", "2nf", "partial dependency"],
  answer: `2NF removes partial dependencies in tables with composite primary keys.

📌 Suggestions:
• Identify full functional dependencies`
},

{
  keywords: ["third normal form", "3nf", "transitive dependency"],
  answer: `3NF eliminates transitive dependencies to improve data consistency.

📌 Suggestions:
• Normalize step-by-step`
},

{
  keywords: ["bcnf", "boyce codd normal form", "advanced normalization"],
  answer: `BCNF is a stronger version of 3NF where every determinant is a candidate key.

📌 Suggestions:
• Apply in critical systems`
},

/* ===================== SQL ===================== */

{
  keywords: ["sql", "structured query language", "database queries"],
  answer: `SQL is a standard language used to define, manipulate, and control relational databases.

📌 Suggestions:
• Practice complex queries`
},

{
  keywords: ["ddl", "data definition language", "create alter drop"],
  answer: `DDL defines database structure using commands like CREATE, ALTER, and DROP.

📌 Suggestions:
• Understand schema changes`
},

{
  keywords: ["dml", "data manipulation language", "insert update delete"],
  answer: `DML is used to insert, update, delete, and retrieve data from tables.

📌 Suggestions:
• Practice transaction safety`
},

{
  keywords: ["dcl", "data control language", "grant revoke"],
  answer: `DCL controls database access using GRANT and REVOKE commands.

📌 Suggestions:
• Follow principle of least privilege`
},

{
  keywords: ["tcl", "transaction control", "commit rollback"],
  answer: `TCL manages transactions using COMMIT, ROLLBACK, and SAVEPOINT.

📌 Suggestions:
• Understand transaction boundaries`
},

/* ===================== TRANSACTIONS ===================== */

{
  keywords: ["transaction", "acid properties", "database consistency"],
  answer: `A transaction is a sequence of operations performed as a single logical unit.

ACID properties ensure reliability.

📌 Suggestions:
• Understand real-world examples`
},

{
  keywords: ["acid", "atomicity consistency isolation durability", "transactions"],
  answer: `ACID properties guarantee correct transaction processing.

📌 Suggestions:
• Map ACID to banking systems`
},

{
  keywords: ["concurrency control", "simultaneous transactions", "dbms locking"],
  answer: `Concurrency control ensures database correctness when multiple transactions execute simultaneously.

📌 Suggestions:
• Learn locking protocols`
},

{
  keywords: ["deadlock", "deadlock prevention", "dbms problem"],
  answer: `Deadlock occurs when transactions wait indefinitely for each other.

📌 Suggestions:
• Use detection and prevention techniques`
},

{
  keywords: ["serializability", "schedule", "transaction order"],
  answer: `Serializability ensures that concurrent transactions produce the same result as serial execution.

📌 Suggestions:
• Study conflict serializability`
},

/* ===================== STORAGE & INDEXING ===================== */

{
  keywords: ["indexing", "database index", "search optimization"],
  answer: `Indexing improves data retrieval speed by reducing disk access.

📌 Suggestions:
• Use indexes selectively`
},

{
  keywords: ["b tree", "b+ tree", "index structure"],
  answer: `B+ trees are balanced tree structures commonly used for database indexing.

📌 Suggestions:
• Compare with hash indexing`
},

{
  keywords: ["hashing", "hash index", "direct access"],
  answer: `Hashing provides fast data access using hash functions.

📌 Suggestions:
• Understand collision handling`
},

{
  keywords: ["file organization", "heap sequential", "storage structure"],
  answer: `File organization determines how data is stored physically on disk.

📌 Suggestions:
• Choose based on workload`
},

/* ===================== ADVANCED ===================== */

{
  keywords: ["view", "virtual table", "sql view"],
  answer: `A view is a virtual table created using SQL queries to simplify complex operations.

📌 Suggestions:
• Use for security`
},

{
  keywords: ["stored procedure", "database procedure", "sql automation"],
  answer: `Stored procedures are precompiled SQL programs stored in the database.

📌 Suggestions:
• Improve performance`
},

{
  keywords: ["trigger", "event based action", "dbms trigger"],
  answer: `Triggers automatically execute actions in response to database events.

📌 Suggestions:
• Use carefully`
},

{
  keywords: ["nosql", "non relational database", "big data"],
  answer: `NoSQL databases handle unstructured data and support horizontal scalability.

📌 Suggestions:
• Compare with RDBMS`
},

{
  keywords: ["distributed database", "replication", "data fragmentation"],
  answer: `Distributed databases store data across multiple locations for scalability and reliability.

📌 Suggestions:
• Study CAP theorem`
}

]
,

  /* =====================================================
     DEEP LEARNING – 20+ QUESTIONS
  ===================================================== */
 DL: [

/* ===================== BASICS ===================== */

{
  keywords: ["deep learning", "neural networks", "representation learning"],
  answer: `Deep Learning is a subfield of Machine Learning that focuses on learning hierarchical representations of data using multi-layer neural networks.

Unlike traditional ML, deep learning automatically extracts features from raw data, making it highly effective for complex tasks such as image recognition, speech processing, and natural language understanding.

📌 Suggestions:
• Learn neural network fundamentals
• Study real-world DL applications`
},

{
  keywords: ["machine learning vs deep learning", "difference", "comparison"],
  answer: `Deep Learning differs from traditional Machine Learning in its ability to automatically learn features using multiple hidden layers.

While ML often requires manual feature engineering, deep learning models learn representations directly from data.

📌 Suggestions:
• Compare use cases
• Understand data requirements`
},

{
  keywords: ["artificial neural network", "ann", "biological inspiration"],
  answer: `An Artificial Neural Network (ANN) is inspired by the human brain and consists of interconnected neurons organized in layers.

Each neuron applies a weighted sum and activation function to its inputs.

📌 Suggestions:
• Study neuron math
• Practice simple ANN models`
},

/* ===================== NETWORK STRUCTURE ===================== */

{
  keywords: ["input layer", "hidden layer", "output layer"],
  answer: `Neural networks consist of an input layer, one or more hidden layers, and an output layer.

Hidden layers enable the network to learn complex non-linear relationships.

📌 Suggestions:
• Visualize network architectures`
},

{
  keywords: ["weights", "bias", "neural parameters"],
  answer: `Weights and biases are trainable parameters that control the influence of inputs on neuron outputs.

They are adjusted during training to minimize error.

📌 Suggestions:
• Understand gradient updates`
},

{
  keywords: ["activation function", "relu sigmoid tanh", "non-linearity"],
  answer: `Activation functions introduce non-linearity into neural networks.

Common functions include ReLU, Sigmoid, Tanh, and Softmax.

📌 Suggestions:
• Learn when to use each function`
},

{
  keywords: ["relu", "vanishing gradient", "deep networks"],
  answer: `ReLU (Rectified Linear Unit) is widely used because it reduces vanishing gradient problems and improves training speed.

📌 Suggestions:
• Study ReLU variants`
},

/* ===================== TRAINING ===================== */

{
  keywords: ["loss function", "objective function", "error minimization"],
  answer: `A loss function measures the difference between predicted output and actual target.

Training aims to minimize this loss.

📌 Suggestions:
• Match loss function with task`
},

{
  keywords: ["backpropagation", "gradient descent", "weight update"],
  answer: `Backpropagation computes gradients of the loss with respect to weights using the chain rule.

It enables efficient training of deep networks.

📌 Suggestions:
• Understand math behind gradients`
},

{
  keywords: ["gradient descent", "optimization", "learning rate"],
  answer: `Gradient Descent is an optimization algorithm used to minimize loss by updating parameters iteratively.

📌 Suggestions:
• Tune learning rate carefully`
},

{
  keywords: ["stochastic gradient descent", "sgd", "mini batch"],
  answer: `SGD updates model parameters using a subset of training data, improving efficiency and generalization.

📌 Suggestions:
• Compare with batch GD`
},

{
  keywords: ["adam optimizer", "rmsprop", "adaptive learning"],
  answer: `Adam combines momentum and adaptive learning rates, making it a popular optimizer for deep learning.

📌 Suggestions:
• Use Adam as default`
},

/* ===================== OVERFITTING ===================== */

{
  keywords: ["overfitting", "generalization", "training vs testing"],
  answer: `Overfitting occurs when a model learns training data too well but performs poorly on unseen data.

📌 Suggestions:
• Use regularization`
},

{
  keywords: ["underfitting", "high bias", "model simplicity"],
  answer: `Underfitting happens when a model is too simple to capture underlying patterns.

📌 Suggestions:
• Increase model capacity`
},

{
  keywords: ["regularization", "l1 l2", "weight penalty"],
  answer: `Regularization techniques penalize large weights to prevent overfitting.

📌 Suggestions:
• Apply L2 regularization`
},

{
  keywords: ["dropout", "neuron deactivation", "overfitting control"],
  answer: `Dropout randomly deactivates neurons during training to improve generalization.

📌 Suggestions:
• Use in hidden layers`
},

{
  keywords: ["early stopping", "training control", "validation loss"],
  answer: `Early stopping halts training when validation performance stops improving.

📌 Suggestions:
• Monitor validation curves`
},

/* ===================== CNN ===================== */

{
  keywords: ["cnn", "convolutional neural network", "image processing"],
  answer: `CNNs are specialized neural networks designed for image and spatial data.

They use convolution and pooling layers to extract features.

📌 Suggestions:
• Study CNN architecture`
},

{
  keywords: ["convolution layer", "filters kernels", "feature extraction"],
  answer: `Convolution layers apply filters to input data to detect features such as edges and textures.

📌 Suggestions:
• Visualize feature maps`
},

{
  keywords: ["pooling", "max pooling", "dimensionality reduction"],
  answer: `Pooling layers reduce spatial dimensions, improving efficiency and robustness.

📌 Suggestions:
• Compare max vs average pooling`
},

/* ===================== RNN ===================== */

{
  keywords: ["rnn", "recurrent neural network", "sequence data"],
  answer: `RNNs are designed to handle sequential data by maintaining hidden states.

📌 Suggestions:
• Apply to time-series data`
},

{
  keywords: ["vanishing gradient", "rnn problem", "long sequences"],
  answer: `RNNs suffer from vanishing gradients when learning long-term dependencies.

📌 Suggestions:
• Use LSTM or GRU`
},

{
  keywords: ["lstm", "long short term memory", "sequence learning"],
  answer: `LSTM networks address vanishing gradient issues using gated memory cells.

📌 Suggestions:
• Study gate mechanisms`
},

{
  keywords: ["gru", "gated recurrent unit", "rnn variant"],
  answer: `GRU is a simplified version of LSTM with fewer gates and parameters.

📌 Suggestions:
• Compare with LSTM`
},

/* ===================== TRANSFORMERS ===================== */

{
  keywords: ["transformer", "attention mechanism", "nlp"],
  answer: `Transformers use self-attention to model relationships between all tokens in a sequence simultaneously.

📌 Suggestions:
• Understand attention math`
},

{
  keywords: ["self attention", "query key value", "transformer core"],
  answer: `Self-attention computes weighted relationships between tokens using query, key, and value vectors.

📌 Suggestions:
• Visualize attention scores`
},

{
  keywords: ["bert", "gpt", "pretrained models"],
  answer: `BERT and GPT are transformer-based models pretrained on large datasets.

📌 Suggestions:
• Learn fine-tuning`
},

/* ===================== PRACTICAL ===================== */

{
  keywords: ["transfer learning", "pretrained models", "fine tuning"],
  answer: `Transfer learning reuses pretrained models for new tasks, reducing training time.

📌 Suggestions:
• Use ImageNet weights`
},

{
  keywords: ["fine tuning", "model adaptation", "pretrained"],
  answer: `Fine-tuning adjusts pretrained weights for a specific downstream task.

📌 Suggestions:
• Freeze initial layers`
},

{
  keywords: ["batch normalization", "training stability", "normalization"],
  answer: `Batch normalization stabilizes and accelerates training by normalizing activations.

📌 Suggestions:
• Place before activation`
},

{
  keywords: ["hyperparameters", "model tuning", "learning rate"],
  answer: `Hyperparameters control model behavior and must be tuned carefully.

📌 Suggestions:
• Use grid search`
},

/* ===================== TOOLS ===================== */

{
  keywords: ["tensorflow", "keras", "deep learning framework"],
  answer: `TensorFlow and Keras provide high-level APIs for building deep learning models.

📌 Suggestions:
• Start with Keras`
},

{
  keywords: ["pytorch", "dynamic graph", "deep learning"],
  answer: `PyTorch offers dynamic computation graphs and is popular in research.

📌 Suggestions:
• Practice PyTorch basics`
},

{
  keywords: ["gpu", "cuda", "training acceleration"],
  answer: `GPUs accelerate deep learning by parallelizing matrix operations.

📌 Suggestions:
• Use cloud GPUs`
},

/* ===================== ETHICS ===================== */

{
  keywords: ["ethical ai", "bias", "fairness"],
  answer: `Ethical considerations in deep learning include bias, fairness, and transparency.

📌 Suggestions:
• Follow responsible AI practices`
}

]
,

  /* =====================================================
     ARTIFICIAL INTELLIGENCE – 20+ QUESTIONS
  ===================================================== */
  AI: [

/* ===================== AI BASICS ===================== */

{
  keywords: ["artificial intelligence", "ai definition", "intelligent systems"],
  answer: `Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks requiring human intelligence, such as reasoning, learning, perception, and decision-making.

AI systems can be rule-based or learning-based and are widely applied in automation, robotics, healthcare, and intelligent software.

📌 Suggestions:
• Understand AI history
• Study real-world AI systems`
},

{
  keywords: ["weak ai", "strong ai", "types of ai"],
  answer: `Weak AI, also known as Narrow AI, is designed to perform specific tasks, while Strong AI aims to replicate human-level intelligence across domains.

Currently, most AI systems are Narrow AI.

📌 Suggestions:
• Compare AI capabilities`
},

{
  keywords: ["history of ai", "turing test", "ai evolution"],
  answer: `The history of AI dates back to the 1950s, beginning with Alan Turing and the Turing Test, which evaluates a machine’s ability to exhibit human-like intelligence.

📌 Suggestions:
• Study AI milestones`
},

/* ===================== INTELLIGENT AGENTS ===================== */

{
  keywords: ["intelligent agents", "agent environment", "ai agents"],
  answer: `An intelligent agent perceives its environment through sensors and acts upon it using actuators to achieve specific goals.

📌 Suggestions:
• Learn agent architectures`
},

{
  keywords: ["rational agent", "performance measure", "agent behavior"],
  answer: `A rational agent selects actions that maximize expected performance based on its knowledge and percepts.

📌 Suggestions:
• Understand rationality in AI`
},

{
  keywords: ["environment types", "fully observable", "partially observable"],
  answer: `AI environments can be fully observable or partially observable, deterministic or stochastic, episodic or sequential.

📌 Suggestions:
• Classify environments correctly`
},

/* ===================== SEARCH ALGORITHMS ===================== */

{
  keywords: ["search algorithms", "state space", "problem solving"],
  answer: `Search algorithms explore a state space to find a path from an initial state to a goal state.

📌 Suggestions:
• Practice problem formulation`
},

{
  keywords: ["uninformed search", "bfs dfs", "blind search"],
  answer: `Uninformed search algorithms such as BFS and DFS do not use heuristic information and explore systematically.

📌 Suggestions:
• Compare time and space complexity`
},

{
  keywords: ["informed search", "heuristic", "a star"],
  answer: `Informed search algorithms use heuristics to guide the search efficiently. A* is a popular informed search algorithm.

📌 Suggestions:
• Design admissible heuristics`
},

{
  keywords: ["a star algorithm", "heuristic function", "optimal search"],
  answer: `The A* algorithm combines path cost and heuristic estimates to find optimal solutions efficiently.

📌 Suggestions:
• Analyze heuristic properties`
},

/* ===================== KNOWLEDGE REPRESENTATION ===================== */

{
  keywords: ["knowledge representation", "ai reasoning", "facts rules"],
  answer: `Knowledge representation involves encoding information in a form that an AI system can reason with.

📌 Suggestions:
• Learn logical representations`
},

{
  keywords: ["propositional logic", "logical reasoning", "ai logic"],
  answer: `Propositional logic represents knowledge using true or false statements and logical operators.

📌 Suggestions:
• Practice logical inference`
},

{
  keywords: ["first order logic", "predicate logic", "quantifiers"],
  answer: `First Order Logic extends propositional logic by introducing predicates, variables, and quantifiers.

📌 Suggestions:
• Model real-world facts`
},

{
  keywords: ["semantic networks", "knowledge graph", "ai structure"],
  answer: `Semantic networks represent knowledge as graphs with nodes and relationships.

📌 Suggestions:
• Explore knowledge graphs`
},

/* ===================== REASONING ===================== */

{
  keywords: ["inference", "logical reasoning", "ai deduction"],
  answer: `Inference is the process of deriving new knowledge from existing facts using logical rules.

📌 Suggestions:
• Understand forward and backward chaining`
},

{
  keywords: ["forward chaining", "rule based", "ai inference"],
  answer: `Forward chaining starts with known facts and applies rules to derive new conclusions.

📌 Suggestions:
• Use in expert systems`
},

{
  keywords: ["backward chaining", "goal driven", "ai reasoning"],
  answer: `Backward chaining starts with a goal and works backward to find supporting facts.

📌 Suggestions:
• Apply in diagnosis systems`
},

/* ===================== PLANNING ===================== */

{
  keywords: ["ai planning", "goal based", "state transition"],
  answer: `Planning involves selecting a sequence of actions that transforms the initial state into a goal state.

📌 Suggestions:
• Study STRIPS representation`
},

{
  keywords: ["planning algorithms", "means ends analysis", "ai planning"],
  answer: `Means-ends analysis reduces differences between current and goal states to plan actions.

📌 Suggestions:
• Practice planning problems`
},

/* ===================== UNCERTAINTY ===================== */

{
  keywords: ["uncertainty", "probabilistic reasoning", "ai uncertainty"],
  answer: `AI systems handle uncertainty using probabilistic models when information is incomplete or noisy.

📌 Suggestions:
• Learn probability theory`
},

{
  keywords: ["bayesian networks", "probabilistic models", "ai"],
  answer: `Bayesian networks represent probabilistic relationships among variables using directed graphs.

📌 Suggestions:
• Interpret conditional probability`
},

{
  keywords: ["markov models", "hidden markov model", "sequence data"],
  answer: `Markov models represent systems where future states depend only on the current state.

📌 Suggestions:
• Apply to speech recognition`
},

/* ===================== MACHINE LEARNING IN AI ===================== */

{
  keywords: ["machine learning", "ai learning", "data driven ai"],
  answer: `Machine Learning enables AI systems to improve performance by learning from data rather than explicit programming.

📌 Suggestions:
• Integrate ML with AI`
},

{
  keywords: ["supervised learning", "ai training", "labeled data"],
  answer: `Supervised learning uses labeled datasets to train predictive AI models.

📌 Suggestions:
• Understand learning paradigms`
},

{
  keywords: ["unsupervised learning", "clustering", "pattern discovery"],
  answer: `Unsupervised learning identifies patterns and structures in unlabeled data.

📌 Suggestions:
• Apply clustering algorithms`
},

/* ===================== NLP ===================== */

{
  keywords: ["natural language processing", "nlp", "human language"],
  answer: `NLP enables AI systems to understand, interpret, and generate human language.

📌 Suggestions:
• Study language models`
},

{
  keywords: ["nlp applications", "chatbots", "language understanding"],
  answer: `NLP applications include chatbots, translation systems, and sentiment analysis.

📌 Suggestions:
• Build simple chatbots`
},

/* ===================== VISION ===================== */

{
  keywords: ["computer vision", "image processing", "ai vision"],
  answer: `Computer Vision enables machines to interpret and analyze visual data.

📌 Suggestions:
• Learn CNN fundamentals`
},

/* ===================== ROBOTICS ===================== */

{
  keywords: ["robotics", "ai robotics", "autonomous systems"],
  answer: `AI in robotics enables autonomous perception, decision-making, and control.

📌 Suggestions:
• Study robot architectures`
},

/* ===================== ETHICS ===================== */

{
  keywords: ["ai ethics", "bias", "fairness"],
  answer: `Ethical AI focuses on fairness, transparency, accountability, and responsible deployment of AI systems.

📌 Suggestions:
• Follow ethical guidelines`
},

{
  keywords: ["explainable ai", "xai", "model transparency"],
  answer: `Explainable AI aims to make AI decisions understandable to humans.

📌 Suggestions:
• Learn interpretable models`
},

/* ===================== FUTURE ===================== */

{
  keywords: ["future of ai", "ai trends", "emerging ai"],
  answer: `The future of AI includes advancements in general intelligence, autonomous systems, and responsible AI development.

📌 Suggestions:
• Stay updated with research`
}

]

};


/* ===============================
   CONFIDENCE-BASED MATCHING
================================ */

function generateAnswer(question) {
  question = question.toLowerCase();
  let bestMatch = null;
  let maxScore = 0;

  const qaList = courseQA[selectedCourse] || [];

  qaList.forEach(item => {
    let score = 0;
    item.keywords.forEach(keyword => {
      if (question.includes(keyword)) score++;
    });
    if (score > maxScore) {
      maxScore = score;
      bestMatch = item;
    }
  });

  if (bestMatch) return bestMatch.answer;

  return `This is a relevant academic question.
Please refine your query or ask about core concepts, algorithms, or applications related to this course.

📌 Suggested Topics:
• Definitions
• Algorithms
• Examples
• Exam-oriented answers`;
}

/* ===============================
   SEND MESSAGE
================================ */

function sendMessage() {
  const input = document.getElementById("userInput");
  const question = input.value.trim();

  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  setTimeout(() => {
    addMessage(generateAnswer(question), "bot");
  }, 500);
}
