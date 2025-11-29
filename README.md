ML EXPEIMENTAL LEARNING :GROUP 22
NAME : SOLANKI OM NARENDRA (24BTRCL190)
	JAYNSH JAIN (24BTRCL088)
	Priyal Dobariya(24BTLCL004)
	Bhavya(24BTRCL110)
Project Report: Dimensionality Reduction of CTI Event Space

1. Research Objectives and Project Aim
The core purpose of this study was to identify the most effective and resource-efficient method for compressing complex attack embeddings while retaining their predictive intelligence.
•	Core Research Question: Which dimensionality reduction techniques: Linear Discriminant Analysis (LDA) or Autoencoder (AE) best preserves the semantic structure of CTI event embeddings?
•	Deepest Aim: To identify the fundamental, low-dimensional axes that linearly define the difference between MITRE ATT&CK Tactics within the high-dimensional CTI embedding space, thereby maximizing model efficiency and interpretability for security analysts.
•	identify the fundamental, low-dimensional axes that efficiently capture the maximum discriminative information within the high-dimensional CTI event embeddings.
•	move beyond mere data compression and determine the optimal mathematical framework (linear via LDA, or non-linear via Autoencoder) for representing complex attack semantics.
•	prioritize interpretability, ensuring the reduced components are maximally useful for security analysts who need clear insights into the characteristics driving the Tactic classifications.
•	 establish a validated, minimal-size vector space (e.g., 2D, 10D) that can be used for fast, resource-efficient clustering and downstream Tactic prediction in real-time systems.
2. Procedure and Work Done by Notebook Cell
Phase I: Prerequisite—Data Generation (Cells 1-5)
This phase establishes the input data (embeddings, $X$) and labels ($y$) required for the dimensionality reduction study.
1.	CELL 1: Setup and Dependencies
o	Action: Installed and imported all required Python libraries, including torch, sentence-transformers, scikit-learn, and utility functions like compute_class_weight.
o	Concept: This prepared the entire software environment needed to handle everything from binary log file parsing to training the complex Autoencoder model.
2.	CELL 2: Load Data and Initial Parse
o	Action: Used files.upload() and the Evtx library to read your raw Sysmon .evtx file and the xmltodict library to convert the binary XML records into the initial df_raw DataFrame.
o	Concept: This is the Data Ingestion step. It transforms the raw system telemetry into a format Python can process, making the raw CTI data accessible.
3.	CELL 3: Data Flattening and Feature Extraction
o	Action: Extracted the critical security fields (CommandLine, Image, ParentImage, EventID) from the nested log structure and cleaned them by handling missing values.
o	Concept: This isolates the specific features required for contextual analysis, preparing the text input for the Transformer model.
4.	CELL 4: MITRE Tactic Labeling and Context Construction
o	Action: Applied rule-based mapping (keywords like mimikatz, schtasks, and EventID types) to assign a specific MITRE Tactic (e.g., 'Execution', 'Discovery') label to each event.
o	Action: Combined the extracted fields into a single, rich Contextual Text feature.
o	Concept: This step creates the supervised ground truth ($\mathbf{y}$) required by LDA, and generates the single, contextual input string for the embedding model.
5.	CELL 5: Embedding Generation
o	Action: Loaded the MiniLM Transformer model and converted the context_text strings into 384-dimensional numerical vectors (embeddings). The data was then scaled using StandardScaler.
o	Concept: This creates the high-dimensional CTI event embeddings ($\mathbf{X}$) that are the subject of your entire dimensionality reduction project. Scaling the data is vital for the Autoencoder to train efficiently.
Phase II: Dimensionality Reduction and Comparative Analysis (Cells 6-10)
This phase executes the core comparison tasks (LDA vs. Autoencoder) to answer your research question.
6.	CELL 6: Dimensionality Reduction Setup
o	Action: Defined the reduction targets (2D, 10D, 50D), determined the maximum components LDA can use, and split the data into train/test sets for downstream evaluation.
o	Action: Calculated Class Weights to ensure the downstream classifier is robust against the rarity of attack events.
o	Concept: This sets up the control parameters for the experiment, ensuring that both LDA and the Autoencoder are evaluated fairly on the same metrics.
7.	CELL 7: LDA (Linear Discriminant Analysis)
o	Action: Applied the supervised, linear LDA model using the Tactic Labels ($\mathbf{y}$) to reduce the data. It successfully reduced the embeddings to 2D.
o	Action: Calculated the Variance Explained by the LDA components.
o	Concept: This model finds the best linear projection to maximize the separation between the Tactic classes. The Variance Explained metric measures how much discriminative power is retained.
8.	CELL 8: Autoencoder (AE) Setup and Training
o	Action: Defined and trained a PyTorch Autoencoder (a neural network) for unsupervised, non-linear reduction, applying it to 2D, 10D, and 50D.
o	Action: Calculated the Reconstruction Error for each AE variant.
o	Concept: This model finds the best non-linear projection to preserve the overall semantic structure of the data. The Reconstruction Error is the measure of how much information was lost in compression.
9.	CELL 9: Downstream Task Evaluation
o	Action: Trained and tested a Logistic Regression classifier 6 times: once on the Original 384D embeddings (baseline) and once on each of the reduced datasets (LDA-2D, AE-2D/10D/50D).
o	Action: Calculated the Macro F1-Score for each classifier.
o	Concept: This is the most important preservation metric (Step 5). It directly answers the question: How well does the reduced data still support the prediction of the MITRE Tactic?
10.	CELL 10: Visualization and Final Metrics
o	Action: Compiled all results (F1 Scores, Reconstruction Errors, Variance Explained) into a single final metrics table and exported it to final_dr_metrics_table.csv.
o	Action: Generated the side-by-side 2D scatter plots comparing the LDA and AE projections.
o	Concept: This fulfills the final Deliverables, allowing you to visually and numerically compare the two techniques and form the conclusion for your Interpretability Report.



3. Procedure and Methodology Breakdown:
A. Data Generation (Phase I: Cells 1–5)
This phase created the high-dimensional input data ($\mathbf{X}$) and the supervision labels ($\mathbf{y}$).
1.	Data Ingestion & Preparation (Cells 1, 2, 3):
o	Action: The raw, binary Sysmon log file (.evtx) was read, parsed, and flattened into a structured DataFrame. Critical features (CommandLine, ParentImage) were extracted and cleaned.
o	Concept: Established the foundation by making the raw CTI telemetry accessible and isolating the key textual signals from the noise.
2.	Feature Labeling & Contextualization (Cell 4):
o	Action: MITRE Tactic Labels ($\mathbf{y}$) were assigned to each log event using rule-based mapping (EventIDs and command keywords). The core input string (Contextual Text) was constructed by combining these features.
o	Concept: Created the ground truth required for the supervised LDA model and the rich text feature necessary for the Transformer to capture semantic context.
3.	Embedding Generation (Cell 5):
o	Action: The MiniLM Transformer model was used to convert the contextual text into 384-dimensional numerical vectors (Embeddings). The data was then scaled using StandardScaler.
o	Concept: Transformed the raw, sequential text into a dense, high-dimensional vector space ($\mathbf{X}$), which is the necessary input for all dimensionality reduction models.


B. Comparative Analysis (Phase II: Cells 6–10)
This phase executed the core comparison of the reduction techniques.
Step	Code Cell	Action Taken (Procedure)	Conceptual Explanation
DR Setup	CELL 6	Defined the dimensional targets (2D, 10D, 50D), split the data for train/test evaluation, and calculated Class Weights for the downstream classifier.	Prepares the parameters for the experiment and ensures fair evaluation despite data imbalance.
Apply LDA	CELL 7	Applied the supervised, linear LDA model to the embeddings, successfully reducing them to the maximum feasible dimension of 2D. Calculated the Variance Explained.	LDA finds the axes that maximize separation between the Tactic classes. The Variance Explained shows how much of that discriminative power is retained.
Apply Autoencoder (AE)	CELL 8	Defined and trained the PyTorch Autoencoder (unsupervised, non-linear), applying it to 2D, 10D, and 50D. Calculated the Reconstruction Error.	The AE seeks to preserve the overall semantic structure of the data. The Reconstruction Error is the metric for how much information was lost in compression (minimal loss = high structural preservation).
Downstream Evaluation	CELL 9	Trained a Logistic Regression classifier on all reduced datasets (LDA-2D, AE-2D/10D/50D) and the original baseline. Calculated the Macro F1-Score.	This is the core preservation metric (Step 5). The F1 Score measures how well the reduced data still supports the prediction of the MITRE Tactic.
Visualization & Report	CELL 10	Generated the final Metrics Table (compiling F1, Error, Variance) and the 2D Scatter Plots comparing LDA vs. AE.	Compiles all deliverables and provides the visual and numerical evidence to answer the research question.

4. Final Result and Conclusion
A. Performance Summary
Metric	Original (384D)	LDA (2D)	Autoencoder (50D)
Macro F1 Score (Preservation)	$0.7959$	$0.7959$	$0.7600$
Reconstruction Error (Structural Loss)	$0.0$	N/A	$0.1295$
Variance Explained (LDA)	$1.0$	$0.9831$	N/A

B. Conclusion Based on Research Question
The data provides a clear answer based on the two comparison metrics:
1.	Preservation for Classification (F1 Score):
o	LDA is superior. It reduced the data from 384 dimensions to just 2 dimensions while perfectly preserving the Tactic Prediction performance (F1 Score remains at $0.7959$).
o	The Autoencoder required 50 dimensions ($76.00\%$) to get close to LDA's 2D performance, showing that non-linear compression is less efficient for this specific classification boundary.
2.	Interpretability (Answer to the Question):
o	The success of LDA proves that the semantic structure required to distinguish attack tactics is overwhelmingly linear. The differences between attack types (Execution vs. Discovery) are easily isolated along simple axes.
o	Since LDA provides these axes and is the most efficient method, LDA is the preferred technique for this project, offering maximum performance and the highest interpretability (the 2D components directly represent the most discriminative linear features of the original embedding).
5.Generated Files and Data Breakdown (Cells 1–10)

Your project creates three main categories of output data that feed into the final report: the input data variables, the model artifacts, and the final result files.

I. Data Generation (Phase I: Cells 1–5)
No persistent files are saved in these cells, but they create the core Python variables that define the problem.
Cell	Variable Created	Description of Data
CELL 4	logs_df['tactic_label']	The ground truth ($\mathbf{y}$): String labels for the 5 MITRE Tactic classes (e.g., 'Execution', 'Discovery') plus 'benign'.
CELL 5	X_full (embeddings)	The raw input data: A NumPy array of shape $(N, 384)$. Each row is the contextual embedding of a Sysmon event, generated by the Transformer model.
CELL 5	X_scaled	The final input matrix used for DR models. It is X_full after being normalized by the StandardScaler.
CELL 5	y_encoded	The numerical version of the Tactic labels (e.g., $0, 1, 2, 3...$) used for model training and LDA.



II. Reduction and Metrics (Phase II: Cells 7–10)
These cells generate the comparison metrics and the final deliverables.

CELL 7: LDA (Linear Discriminant Analysis)
Variable/Data Generated	Contents	Purpose/Used For
lda_results[k]['X_reduced'] (for $k=2$)	The reduced 2D embedding space.	Visualization (plotting the 2D graph) and Downstream Evaluation (training the classifier in Cell 9).
lda_results[k]['variance_explained']	A scalar value (e.g., 0.9831).	Measures the percentage of discriminative power captured by the reduced LDA components. Deliverable

CELL 8: Autoencoder (AE) Setup and Training
Variable/Data Generated	Contents	Purpose/Used For
ae_results[k]['X_reduced'] (for $k=2, 10, 50$)	The reduced 2D, 10D, and 50D non-linear embeddings.	Visualization (2D plot) and Downstream Evaluation (Cell 9).
ae_results[k]['reconstruction_error']	A scalar value (e.g., 0.1295).	Measures the average loss of information (semantic structure) when compressing and decompressing the embeddings. Deliverable
CELL 10: Final Deliverables (Files Exported)
This cell compiles all the metrics from Cells 7, 8, and 9 into the final report files.
1.	final_dr_metrics_table.csv
o	Data: The central table of your project. It contains the Macro F1 Score for all 7 runs (Original, LDA-2D, AE-2D/10D/50D), the Reconstruction Error (for AE), and the Variance Explained (for LDA).
o	Purpose: Fulfills the Preservation Metrics Table and Downstream Task Performance deliverables.
2.	2d_reduction_comparison.png
o	Data: A graphic showing two scatter plots side-by-side: the LDA 2D reduction and the Autoencoder 2D reduction, with points colored by their Tactic label.
o	Purpose: Fulfills the 2D Visualization Comparisons deliverable, providing the visual evidence for the interpretability report.

3.	No Model File (.pkl):
o	Note: Since the project goal is comparison and interpretability of the reduced features, we do not save the final classifier model (LogisticRegression), but we do save the resulting metrics.

