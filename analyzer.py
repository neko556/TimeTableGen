import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_student_clustering(student_registrations, optimal_k=3):
    """
    Analyzes student registration data to find clusters of students.
    """
    if not student_registrations:
        return {}

    # 1. Prepare Data
    all_courses = sorted(list(set(course for courses in student_registrations.values() for course in courses)))
    df = pd.DataFrame(columns=all_courses, index=student_registrations.keys())
    for student, courses in student_registrations.items():
        df.loc[student, courses] = 1
        
    # FIX for FutureWarning:
    df = df.fillna(0).infer_objects(copy=False)

    print("--- Student-Course Matrix (Corrected) ---")
    print(df)

    # 2. Run K-Means (adaptively)
    num_students = len(df)
    if num_students < 2:
        df['cluster'] = range(num_students)
    else:
        k_to_use = min(optimal_k, num_students)
        kmeans = KMeans(n_clusters=k_to_use, n_init='auto', random_state=42)
        df['cluster'] = kmeans.fit_predict(df)

    # 3. Analyze and Format Output
    dynamic_group_courses = {}
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        if not cluster_df.empty:
            # CRITICAL FIX: Create a view with only course columns first
            courses_only_df = cluster_df.drop('cluster', axis=1)
            # Then create the filter and apply it to the course columns
            courses_in_cluster = courses_only_df.columns[courses_only_df.sum() > 0].tolist()
            dynamic_group_courses[f"Cluster_{cluster_id}"] = courses_in_cluster
            
    print("\n--- Dynamically Generated Student Groups ---")
    print(dynamic_group_courses)
    
    return dynamic_group_courses