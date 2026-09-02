# data_quality_audit

This is a Data Quality Exception Report of university student data that includes name, ID, department, issues, tuition, GPA and enroll date. I introduced errors and then created this program to run validation checks and then output 2 files to display the errors found. The files are:

data_quality_exception_log.csv - A CSV log of every error
data_quality_exception_memo.html - A business ready styled HTML file that displays error by severity. Where Critical is something missing several foreign keys and would have to be escalated to a senior. Warning is anything that I could fix myself. 

To run open terminal, navigate to your folder and run:

python data_audit.py
