# Student PG Rent Analysis and Prediction
**Experiment 12 — Data Science Mini Project**

## Problem statement
Students moving to a college town need to compare paying-guest (PG) accommodation options. This project explores how monthly rent relates to distance from college, room-sharing capacity, facilities, and accommodation type, and demonstrates rent prediction and affordability classification.

## Dataset notice
`pg_rent_dataset.csv` contains **synthetic demonstration data** (80 illustrative listings), not verified live listings or a claim that these properties exist. Replace it with ethically collected, accurate local listings before using results for real accommodation decisions.

## Features
- Exploratory analysis using Pandas and NumPy
- Three graphs generated with Matplotlib
- Random Forest Regression to estimate monthly rent per person
- Logistic Regression to classify listings as Affordable or Expensive
- Interactive terminal input for one prediction

## Files
- `main.py` — single Python entry point
- `pg_rent_dataset.csv` — dataset
- `graphs/` — three example graphs, regenerated when the script runs
- `Project_Report.docx` — project report
- `README.md` — instructions and project overview

## Requirements
Python 3.9 or later recommended.

```bash
pip install pandas numpy matplotlib scikit-learn
python main.py
```

Run the commands from this folder. The script expects `pg_rent_dataset.csv` beside `main.py` and writes plots into `graphs/`.

## CSV columns
| Column | Meaning |
|---|---|
| `PG_ID` | Demonstration listing ID |
| `Location` | Approximate area label |
| `Monthly_Rent_Rs` | Monthly rent per person in Indian rupees |
| `Distance_km` | Distance from college in kilometres |
| `Sharing_Capacity` | Number of people sharing the room |
| `Current_Residents` | Current residents |
| `Accommodation_Type` | Boys, Girls, or Co-ed |
| `WiFi`, `Laundry`, `Meals_Included`, `AC`, `Security` | 1 = available/included; 0 = not available |

## Method
1. Load and validate the CSV.
2. Calculate descriptive statistics and the median rent.
3. Define Affordable as rent at or below the dataset median; otherwise Expensive.
4. Split data into training and test sets.
5. Train Random Forest Regression and report MAE, RMSE, and R².
6. Train Logistic Regression and report accuracy and classification metrics.
7. Save the three plots and optionally predict rent for a user-entered listing.

## Interpretation and limitations
The affordability boundary is relative to the dataset median, not an official market definition or a student's personal budget. Small synthetic datasets can produce unstable metrics and do not establish real-world predictive accuracy. A real study should collect more listings, verify rent units and facilities, avoid duplicate records, document collection dates, and validate on unseen real listings. Rent can also depend on deposits, food quality, electricity, occupancy rules, and seasonal demand, which are not fully represented here.
