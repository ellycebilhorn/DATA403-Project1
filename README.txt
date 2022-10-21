DATA 403 FALL 2022 
Project 1
=======
TEAM BEAUMARIS:
Ellyce Bilhorn (ebilhorn@calpoly.edu)
Edward Du (eddu@calpoly.edu)
Kaanan Kharwa (kkharwa@calpoly.edu)
=======
Programming Language: Python
Deliverables: gen_alc_dataset.py gen_df.py eda.py predict_models.py pre_interp_coefs.py interp_models.py
NOTES: 
=======
How-to Run:
1. Make sure that /datasets/Iowa_Liquor_Sales.csv exists.
2. Run gen_alc_dataset.py
    This file creates the joined, cleaned datasets and stores it in alc_dataset.csv.
3. Run gen_df.py
    This file creates the grouped and aggregated dataset (by Month) and stores it in grouped_df.csv.
4. Run eda.py
    This file does the exploratory data analysis.
5. Run predict_models.py
    This file runs all 4 predictive models and cross-validation for each model. 
    NOTE: This file takes a long time to run (grid search + all-but-one-cross-validation). 
6. Run pre_interp_coefs.py
	This file generates the coefficient weightings (why we chose to include certain variables in our interpretive model).
7. Run interp_models.py
	This file runs all 4 interpretive models.
