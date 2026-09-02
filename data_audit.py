import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. GENERATE A SYNTHETIC "DIRTY" DATASET (Simulates SIS + HR)
# ------------------------------------------------------------
np.random.seed(42)  # for reproducibility
n = 500

# Simulate student/faculty records
student_ids = [f'S{str(i).zfill(5)}' for i in range(1, n+1)]
names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']

data = {
    'student_id': student_ids,
    'first_name': np.random.choice(first_names, n),
    'last_name': np.random.choice(names, n),
    'department': np.random.choice(['Math', 'English', 'History', 'Biology', 'CS', None], n, p=[0.2,0.2,0.15,0.15,0.2,0.1]),
    'tuition_paid': np.round(np.random.uniform(5000, 25000, n), 2),
    'enrollment_date': pd.date_range(start='2020-01-01', periods=n).strftime('%Y-%m-%d'),
    'gpa': np.round(np.random.uniform(1.5, 4.0, n), 2),
    'faculty_advisor_id': [f'F{str(np.random.randint(1, 100)).zfill(3)}' for _ in range(n)]
}

df = pd.DataFrame(data)

# INTENTIONALLY INJECT ERRORS (Simulating real-world messiness)

# Error 1: Negative tuition (Critical)
df.loc[10, 'tuition_paid'] = -500.00
df.loc[55, 'tuition_paid'] = -1200.00

# Error 2: Future enrollment date (Critical - impossible)
df.loc[22, 'enrollment_date'] = '2027-12-01'
df.loc[88, 'enrollment_date'] = '2028-05-15'

# Error 3: GPA outliers > 4.0 or < 0.0 (Warning - extreme outliers)
df.loc[34, 'gpa'] = 5.2
df.loc[76, 'gpa'] = -1.5
df.loc[120, 'gpa'] = 4.8

# Error 4: Duplicate student IDs (Warning)
# Duplicate student S00002
duplicate_rows = df[df['student_id'] == 'S00002'].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Error 5: Faculty Advisor ID format - invalid (not starting with F or wrong length) - Critical
df.loc[150, 'faculty_advisor_id'] = 'XYZ99'
df.loc[200, 'faculty_advisor_id'] = 'F12'  # missing digit (should be Fxxx)
df.loc[250, 'faculty_advisor_id'] = 'F9999' # too long

# Error 6: Tuition paid ridiculously high (Warning - statistical outlier, > 3 std devs)
df.loc[300, 'tuition_paid'] = 950000.00

# Shuffle the DataFrame to hide errors
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Mock dataset generated: {len(df)} total records.")
print("⚠️  Injected errors: negative tuition, future dates, missing depts, outliers, duplicates, invalid IDs.\n")

# ------------------------------------------------------------
# 2. DEFINE THE VALIDATION RULES & FLAGGING LOGIC
# ------------------------------------------------------------

# Create empty lists to collect flagged issues
flags = []

# Convert enrollment_date to datetime for comparisons
df['enrollment_date_parsed'] = pd.to_datetime(df['enrollment_date'], errors='coerce')
today = datetime.now()

for idx, row in df.iterrows():
    errors = []
    severity = 'Warning'  # default
    
    # --- CRITICAL CHECKS (Escalate to Senior) ---
    
    # C1: Negative tuition
    if row['tuition_paid'] < 0:
        errors.append(f"Negative tuition: ${row['tuition_paid']:,.2f}")
        severity = 'Critical'
    
    # C2: Future enrollment date (after today)
    if pd.notna(row['enrollment_date_parsed']) and row['enrollment_date_parsed'] > today:
        errors.append(f"Future enrollment date: {row['enrollment_date']}")
        severity = 'Critical'
    
    # C3: Invalid Faculty Advisor ID (must match pattern F[0-9][0-9][0-9])
    if not (isinstance(row['faculty_advisor_id'], str) and 
            len(row['faculty_advisor_id']) == 4 and 
            row['faculty_advisor_id'].startswith('F') and 
            row['faculty_advisor_id'][1:].isdigit()):
        errors.append(f"Invalid advisor ID: {row['faculty_advisor_id']}")
        severity = 'Critical'  # Override to Critical if this is caught with others
    
    # --- WARNING CHECKS (Fix yourself / investigate) ---
    
    # W1: Missing Department
    if pd.isna(row['department']):
        errors.append("Missing department")
        # Keep severity as Warning unless already Critical
    
    # W2: GPA out of realistic range (0-4.5)
    if row['gpa'] > 4.5 or row['gpa'] < 0:
        errors.append(f"GPA out of range (0-4.5): {row['gpa']}")
        if severity != 'Critical': severity = 'Warning'  # keep Critical if already flagged
    
    # W3: Extreme tuition outlier (> 3 standard deviations from mean - we'll calculate after)
    # We'll handle this separately with a global calculation to avoid iterating twice.
    
    # If there are errors, add to flags list
    if errors:
        flags.append({
            'row_index': idx,
            'student_id': row['student_id'],
            'name': f"{row['first_name']} {row['last_name']}",
            'department': row['department'] if pd.notna(row['department']) else 'NULL',
            'severity': severity,
            'issues': ' | '.join(errors),
            'tuition_paid': row['tuition_paid'],
            'gpa': row['gpa'],
            'enrollment_date': row['enrollment_date'],
            'advisor': row['faculty_advisor_id']
        })

# --- GLOBAL STATISTICAL OUTLIER CHECK (Tuition > 3 std devs) ---
mean_tuition = df['tuition_paid'].mean()
std_tuition = df['tuition_paid'].std()
upper_bound = mean_tuition + (3 * std_tuition)

# Re-iterate to add outlier flags, but only if not already Critical
for idx, row in df.iterrows():
    if row['tuition_paid'] > upper_bound:
        # Check if this row already has a flag
        existing = next((f for f in flags if f['row_index'] == idx), None)
        if existing:
            # Add to existing issues, but keep Critical if already critical
            if 'tuition outlier' not in existing['issues']:
                existing['issues'] += f" | Extreme tuition outlier (>3 std devs): ${row['tuition_paid']:,.2f}"
                if existing['severity'] != 'Critical':
                    existing['severity'] = 'Warning'  # upgrade to Warning if not Critical
        else:
            # New flag
            flags.append({
                'row_index': idx,
                'student_id': row['student_id'],
                'name': f"{row['first_name']} {row['last_name']}",
                'department': row['department'] if pd.notna(row['department']) else 'NULL',
                'severity': 'Warning',
                'issues': f"Extreme tuition outlier (>3 std devs): ${row['tuition_paid']:,.2f}",
                'tuition_paid': row['tuition_paid'],
                'gpa': row['gpa'],
                'enrollment_date': row['enrollment_date'],
                'advisor': row['faculty_advisor_id']
            })

# --- DUPLICATE STUDENT ID CHECK ---
duplicate_ids = df[df.duplicated('student_id', keep=False)]['student_id'].unique()
if len(duplicate_ids) > 0:
    dup_rows = df[df['student_id'].isin(duplicate_ids)]
    for idx, row in dup_rows.iterrows():
        # Check if already flagged
        existing = next((f for f in flags if f['row_index'] == idx), None)
        if existing:
            if 'Duplicate student ID' not in existing['issues']:
                existing['issues'] += f" | Duplicate student ID (shared with {len(dup_rows[dup_rows['student_id']==row['student_id']])-1} others)"
                if existing['severity'] != 'Critical':
                    existing['severity'] = 'Warning'
        else:
            flags.append({
                'row_index': idx,
                'student_id': row['student_id'],
                'name': f"{row['first_name']} {row['last_name']}",
                'department': row['department'] if pd.notna(row['department']) else 'NULL',
                'severity': 'Warning',
                'issues': f"Duplicate student ID (shared with {len(dup_rows[dup_rows['student_id']==row['student_id']])-1} others)",
                'tuition_paid': row['tuition_paid'],
                'gpa': row['gpa'],
                'enrollment_date': row['enrollment_date'],
                'advisor': row['faculty_advisor_id']
            })

# Convert to DataFrame
flags_df = pd.DataFrame(flags)

# ------------------------------------------------------------
# 3. GENERATE OUTPUTS
# ------------------------------------------------------------

# ---- 3A: CSV DETAILED LOG ----
flags_df.to_csv('data_quality_exception_log.csv', index=False)
print(f"📊 Detailed CSV log saved: {len(flags_df)} issues flagged.")

# ---- 3B: HTML EXCEPTION MEMO (Business-Ready) ----
critical_count = len(flags_df[flags_df['severity'] == 'Critical'])
warning_count = len(flags_df[flags_df['severity'] == 'Warning'])

# Generate HTML with styling
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Quality Exception Memo</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8f9fa; }}
        .memo-box {{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #b31b1b; padding-bottom: 10px; }}
        .header-info {{ background: #f1f3f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .summary {{ display: flex; gap: 30px; margin: 25px 0; }}
        .summary-box {{ flex: 1; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; }}
        .critical-box {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .warning-box {{ background: #fff3cd; color: #856404; border: 1px solid #ffc107; }}
        .escalate-note {{ background: #cce5ff; padding: 15px; border-left: 5px solid #004085; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
        .tag-critical {{ background: #dc3545; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .tag-warning {{ background: #ffc107; color: #212529; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .footer {{ margin-top: 30px; font-size: 13px; color: #6c757d; border-top: 1px solid #dee2e6; padding-top: 15px; }}
        .action-needed {{ background: #fcf8e3; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
<div class="memo-box">
    <h1> Data Quality Exception Memo</h1>
    <div class="header-info">
        <strong>To:</strong> Senior Team, Institutional Reporting<br>
        <strong>From:</strong> Junior Data Analyst (Portfolio Project)<br>
        <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
        <strong>Subject:</strong> Critical Data Discrepancies Identified in Monthly Extract
    </div>

    <div class="escalate-note">
        <strong> Executive Summary:</strong> 
        During routine validation of the {len(df)}-record dataset, I identified <strong>{len(flags_df)} total exceptions</strong>. 
        <strong>{critical_count} Critical</strong> issues require immediate escalation and <strong>{warning_count} Warnings</strong> require investigative follow-up.
    </div>

    <div class="summary">
        <div class="summary-box critical-box">
             CRITICAL (Escalate): {critical_count}
        </div>
        <div class="summary-box warning-box">
             WARNINGS (Investigate): {warning_count}
        </div>
    </div>

    <h3>Full Exception List</h3>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Student ID</th>
                <th>Name</th>
                <th>Department</th>
                <th>Issues</th>
                <th>Tuition</th>
                <th>GPA</th>
                <th>Enroll Date</th>
            </tr>
        </thead>
        <tbody>
"""

# Add rows to the HTML table
for _, row in flags_df.iterrows():
    severity_tag = f'<span class="tag-critical">CRITICAL</span>' if row['severity'] == 'Critical' else f'<span class="tag-warning">WARNING</span>'
    tuition_str = f"${row['tuition_paid']:,.2f}" if pd.notna(row['tuition_paid']) else "N/A"
    gpa_str = f"{row['gpa']:.2f}" if pd.notna(row['gpa']) else "N/A"
    enroll_str = row['enrollment_date'] if pd.notna(row['enrollment_date']) else "N/A"
    
    html_content += f"""
        <tr>
            <td>{severity_tag}</td>
            <td>{row['student_id']}</td>
            <td>{row['name']}</td>
            <td>{row['department']}</td>
            <td style="max-width: 250px;">{row['issues']}</td>
            <td>{tuition_str}</td>
            <td>{gpa_str}</td>
            <td>{enroll_str}</td>
        </tr>
    """

html_content += f"""
        </tbody>
    </table>

    <div class="action-needed">
        <strong> Recommended Next Steps:</strong>
        <ul>
            <li><strong>Critical (Escalate):</strong> Review rows with negative tuition, future dates, and invalid advisor IDs. These will break downstream financial aid and federal reporting.</li>
            <li><strong>Warnings (Investigate):</strong> Reach out to department admin for missing department codes. Verify GPA outliers with Registrar's office.</li>
            <li><strong>Duplicate IDs:</strong> Confirm with IT whether these are system replication errors or true duplicates.</li>
        </ul>
        <p style="margin-top:10px;"><em>I have saved the detailed CSV log for your reference. I can assist in correcting the Warning-level issues, but Critical items require your approval before we proceed with final reporting.</em></p>
    </div>

    <div class="footer">
        This is an automated exception report generated by the Data Quality Audit script. 
        For full details, see <code>data_quality_exception_log.csv</code>.
    </div>
</div>
</body>
</html>
"""

# Write the HTML file
with open('data_quality_exception_memo.html', 'w') as f:
    f.write(html_content)

print("📄 HTML Exception Memo saved: data_quality_exception_memo.html")

# ------------------------------------------------------------
# 4. PRINT SUMMARY TO CONSOLE
# ------------------------------------------------------------
print("\n" + "="*60)
print("AUDIT COMPLETE")
print("="*60)
print(f"Total records scanned: {len(df)}")
print(f"Total exceptions found: {len(flags_df)}")
print(f"  🔴 Critical (Escalate): {critical_count}")
print(f"  🟡 Warnings (Investigate): {warning_count}")
print("\n📁 Files generated:")
print("  1. data_quality_exception_log.csv  (for detailed filtering)")
print("  2. data_quality_exception_memo.html (open in browser for executive summary)")
print("\n✅ Portfolio project ready! Open the HTML file to see your business-facing memo.")