import argparse
import time
from data_loader import load_university_data
from analyzer import run_student_clustering
from ga_solver import solve_with_ga
from sat_solver import solve_with_or_tools

 
def print_formatted_schedule(schedule_list):
    if not schedule_list:
        print("  - No classes scheduled.")
        return
        
    day_order = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}
    sorted_schedule = sorted(schedule_list, key=lambda s: (day_order.get(s[3].split('_')[0], 99), s[3]))
    last_day = ""
    for session in sorted_schedule:
        course_code, prof_id, room_id, timeslot = session
        day = timeslot.split('_')[0]
        if day != last_day:
            print(f"  --- {day} ---")
            last_day = day
        print(f"    {timeslot}: {course_code} by {prof_id} in {room_id}")

def main():
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="AI-Based Timetable Generation System")
    parser.add_argument('--solver', required=True, choices=['ga', 'sat', 'hybrid'], 
                        help="Specify the solver to use: 'ga', 'sat', or 'hybrid'.")
    args = parser.parse_args()

    # --- 2. Load and Process Real Data ---
    print("--- Loading University Data from CSV files ---")
    university_data= load_university_data()
    
    # Exit if data loading failed (e.g., files not found)
    if not university_data:
        return

    # --- 3. Analyze Data to Find Cohorts ---
    # The analyzer uses the 'student_registrations' part of our loaded data
    university_data, kmeans_needed,programs_df = load_university_data()
    
    if not university_data:
        return

    # --- 3. Conditionally Analyze Data ---
    if kmeans_needed:
        print("\n--- Running K-Means Analyzer ---")
        dynamic_groups = run_student_clustering(university_data["student_registrations"])
    else:
        print("\n--- Using Pre-defined Student Groups (K-Means Bypassed) ---")
        # If data is clean, the "groups" are just the student registrations
        dynamic_groups = university_data["student_registrations"]
    
    # --- 4. Run the Selected Solver ---
    print(f"\n--- Running Timetable Generation with {args.solver.upper()} Solver ---")
    final_timetable = None
    solution_package = None
   
    start_time = time.time()

    if args.solver == 'ga':
        final_timetable = solve_with_ga(university_data, dynamic_groups) 
    elif args.solver == 'sat':
        final_timetable = solve_with_or_tools(university_data, dynamic_groups, time_limit=30)
    elif args.solver == "hybrid":
        seed = solve_with_or_tools(university_data, dynamic_groups, time_limit=20)
        if seed:
            final_timetable = solve_with_ga(university_data, dynamic_groups, seed_solution=seed)
        else:
            final_timetable = solve_with_ga(university_data, dynamic_groups, seed_solution=None)
    
    end_time = time.time()
    print(f"\n--- Solver finished in {end_time - start_time:.2f} seconds ---")

    # --- 5. Generate and Print Final Views ---
    solution_package = final_timetable
    if final_timetable:
        session_map = {session[0]: session for session in final_timetable}
        print("\n\n--- Professor Timetables ---")
        # Access the professor timetables from the solution package
        for prof, schedule in solution_package["professor_timetables"].items():
            print(f"\nSchedule for {prof}:")
            # Use the helper function to print the schedule neatly
            print_formatted_schedule(schedule)
        print("\n\n--- Program Timetables  ---")
        # Iterate directly through the original programs data
        for index, row in programs_df.iterrows():
            program_id = row['program_id']
            required_subjects = row['course_codes'].split(',')
            
            print(f"\nSchedule for {program_id}:")
            
            schedule_for_this_program = []
            for subject in required_subjects:
                if subject in session_map:
                    schedule_for_this_program.append(session_map[subject])
            
            print_formatted_schedule(schedule_for_this_program)
    else:
        print("\n--- No final solution was generated. ---")
                    
    
    

if __name__ == '__main__':
    main()
