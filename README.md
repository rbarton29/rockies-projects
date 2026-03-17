# rockies-projects
Analysis of 2025 Rockies vs rest of MLB in a few metrics, and Batting Lineup Optimizer with 2026 Projections

Instructions to Run Code Files

Before running the files, you can either install all of needed the libraries (for all 3 projects) in the terminal directly with this command:

pip install pybaseball pandas numpy tqdm matplotlib seaborn scipy

OR, you can create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate 
pip install pybaseball pandas numpy tqdm matplotlib seaborn scipy

Then, in order to run the COL vs MLB analysis, you just run this command in the terminal from the folder containing the file: 
python rockies_2025_analysis.py

If you want to run the batting lineup optimizer, you navigate to inside the unzipped folder and run: 
python main.py
