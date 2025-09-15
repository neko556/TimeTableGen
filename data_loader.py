import pandas as pd
import json
import ast

def load_university_data():
    """
    Reads all university data from CSV files and prepares it for the solvers.
    """
    try:
       
        # Load the raw data from CSV files
        courses_df = pd.read_csv("courses.csv")
        faculty_df = pd.read_csv("faculty.csv")
        rooms_df = pd.read_csv("rooms.csv")
        enrollments_df = pd.read_csv("student_enrollments.csv")
        programs_df = pd.read_csv("programs.csv")
        prefs_df = pd.read_csv("preferences.csv")
        print("CSV files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all required CSV files are in the directory.")
        return None

    # --- Process Data for Solvers ---

    # 1. Get a list of all courses that need to be scheduled in a timeslot
    # We exclude internships, fieldwork, etc., as they are handled differently.
    scheduled_courses = courses_df[~courses_df['type'].isin(['Internship', 'Fieldwork'])]

    # 1. Create a map from program_id to a list of subject_codes
    student_registrations = {}
    kmeans_needed = False
    program_map={}

    if 'program_id' in enrollments_df.columns:
        print("Info: 'program_id' found. Using pre-defined groups (K-Means will be bypassed).")
        kmeans_needed = False
        program_map = {row['program_id']: row['course_codes'].split(',') 
                       for _, row in programs_df.iterrows()}
        for _, row in enrollments_df.iterrows():
            student_id, program_id = row['student_id'], row['program_id']
            if program_id in program_map:
                student_registrations[student_id] = program_map[program_id]
    else:
        print("Info: 'program_id' not found. Data will be clustered with K-Means.")
        kmeans_needed = True
        # Group by student_id and create a list of their subjects
        student_registrations = enrollments_df.groupby('student_id')['course_code'].apply(list).to_dict()
    
    for index, row in enrollments_df.iterrows():
        student_id = row['student_id']
        program_id = row['program_id']
        if program_id in program_map:
            student_registrations[student_id] = program_map[program_id]

    
    # 3. Get course details like type and enrollment count
    course_enrollments = enrollments_df['program_id'].value_counts().to_dict()
    course_info = courses_df.set_index('course_code').to_dict('index')

    course_enrollments = {}
    for student, subjects in student_registrations.items():
        for subject in subjects:
            course_enrollments[subject] = course_enrollments.get(subject, 0) + 1
    

    # 4. Get faculty details like expertise and availability
    faculty_info = faculty_df.set_index('faculty_id').to_dict('index')
    for fid in faculty_info:
        # Convert expertise string to a list
        faculty_info[fid]['expertise'] = str(faculty_info[fid]['expertise']).split(',')
        
        # CRITICAL FIX: Use ast.literal_eval for robust parsing of the availability string
        availability_str = faculty_info[fid]['availability']
        if isinstance(availability_str, str) and availability_str:
            try:
                # Safely evaluate the string as a Python dictionary
                faculty_info[fid]['availability'] = ast.literal_eval(availability_str)
            except (ValueError, SyntaxError):
                # If the string is malformed, default to an empty dictionary
                print(f"Warning: Could not parse availability for {fid}. Defaulting to empty.")
                faculty_info[fid]['availability'] = {}
        else:
            # If the cell is empty, default to an empty dictionary
            faculty_info[fid]['availability'] = {}
            
            
    # 5. Get room details
    room_info = rooms_df.set_index('room_id').to_dict('index')
    # --- Process Preferences Data ---
    processed_preferences = {"professors": {}, "courses": {}}
    for _, row in prefs_df.iterrows():
        target_id = row['target_id']
        rule_type = row['rule_type']
        value = row['value']
        target_key = f"{row['target_type']}s"

        if target_id not in processed_preferences[target_key]:
            processed_preferences[target_key][target_id] = {}
        if rule_type not in processed_preferences[target_key][target_id]:
            processed_preferences[target_key][target_id][rule_type] = []
        
        processed_preferences[target_key][target_id][rule_type].append(value)

    # Bundle everything into a single dictionary
    university_data = {
        "scheduled_courses": scheduled_courses.to_dict('records'),
        "all_courses": course_info,
        "student_registrations": student_registrations,
        "course_enrollments": course_enrollments,
        "faculty": faculty_info,
        "rooms": room_info,
        "preferences": processed_preferences
    }
    
    print("Data processed for solvers.")
    return university_data,kmeans_needed,programs_df

# Example of how to run this from main.py
if __name__ == '__main__':
    data = load_university_data()
    # You can print parts of the data to verify it's loaded correctly
    if data:
        print("\nSample Processed Data:")
        print("Course Enrollments:", data['course_enrollments'])
        print("Faculty Expertise for prof_ada:", data['faculty']['prof_ada']['expertise'])