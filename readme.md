README: [Datasets present at the same location as the python scripts]


1. Run the perceptron.py python file :

	Syntax:​ python perceptron.py --dataset <dataset> --mode <mode>

 	-- Dataset:​ linearly-separable-dataset.csv / Breast_cancer_data.csv
	-- Mode:​ erm / cv (erm - Empirical Risk Minimization, cv - Cross-validation), for cross validation, number of folds ‘k’ = 10 by default)

  To run the code:​ 

	python perceptron.py --dataset linearly-separable-dataset.csv --mode erm 				
	python perceptron.py --dataset Breast_cancer_data.csv --mode cv


2. Run the adaBoost.py python file.

	Syntax:​ python adaBoost.py  --dataset <dataset> --mode <mode>
	
	-- Dataset:​ Breast_cancer_data.csv
	-- Mode:​ erm / cv (For cross validation, number of folds = 10 by default) / plot (for plotting graph between ERM and Validation error vs Rounds)

  To run the code:​ 
 
	(i)   python adaBoost.py --dataset Breast_cancer_data.csv --mode erm 					
	(ii)  python adaboost-final.py --dataset Breast_cancer_data.csv --mode cv 					
	(iii) python adaboost-final.py --dataset Breast_cancer_data.csv --mode plot    (for plotting graph)