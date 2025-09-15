# main.py (fixed, drop-in)
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
    # tolerant sort: session is (course_code, prof_id, room_id, timeslot)
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
    parser = argparse.ArgumentParser(description="AI-Based Timetable Generation System")
    parser.add_argument('--solver', required=True, choices=['ga', 'sat', 'hybrid'], 
                        help="Specify the solver to use: 'ga', 'sat', or 'hybrid'.")
    args = parser.parse_args()

    print("--- Loading University Data from CSV files ---")
    result = load_university_data()
    # Expecting tuple (university_data, kmeans_needed, programs_df)
    if not result:
        print("Failed to load university data; exiting.")
        return
    try:
        university_data, kmeans_needed, programs_df = result
    except Exception as e:
        print("Unexpected return shape from load_university_data():", e)
        return

    if not university_data:
        print("No university data available; exiting.")
        return

    # --- 3. Conditionally Analyze Data ---
    if kmeans_needed:
        print("\n--- Running K-Means Analyzer ---")
        dynamic_groups = run_student_clustering(university_data.get("student_registrations", {}))
    else:
        print("\n--- Using Pre-defined Student Groups (K-Means Bypassed) ---")
        dynamic_groups = university_data.get("student_registrations", {})

    print(f"\n--- Running Timetable Generation with {args.solver.upper()} Solver ---")
    solution_package = None

    start_time = time.time()
    if args.solver == 'ga':
        solution_package = solve_with_ga(university_data, dynamic_groups) 
    elif args.solver == 'sat':
        solution_package = solve_with_or_tools(university_data, dynamic_groups, time_limit=30)
    elif args.solver == "hybrid":
        seed_pkg = solve_with_or_tools(university_data, dynamic_groups, time_limit=20)
        seed_solution = None
        if seed_pkg and isinstance(seed_pkg, dict):
            seed_solution = seed_pkg.get("master_timetable")
        solution_package = solve_with_ga(university_data, dynamic_groups, seed_solution=seed_solution)

    end_time = time.time()
    print(f"\n--- Solver finished in {end_time - start_time:.2f} seconds ---")

    if not solution_package:
        print("\n--- No final solution was generated. ---")
        return

    # master_timetable might be a list of sessions
    master = solution_package.get("master_timetable", [])
    if not isinstance(master, list):
        # if GA returns a DEAP individual or alike, convert safely:
        try:
            master = list(master)
        except Exception:
            master = []

    session_map = {s[0]: s for s in master if isinstance(s, (list, tuple)) and len(s) >= 4}

    print("\n\n--- Professor Timetables ---")
    for prof, schedule in solution_package.get("professor_timetables", {}).items():
        print(f"\nSchedule for {prof}:")
        print_formatted_schedule(schedule)

    print("\n\n--- Program Timetables  ---")
    # programs_df may be a dataframe; fallback to dynamic_groups if None
    if programs_df is not None:
        for index, row in programs_df.iterrows():
            program_id = row.get('program_id')
            course_codes = str(row.get('course_codes', '')).split(',')
            print(f"\nSchedule for {program_id}:")
            schedule_for_program = [session_map[c] for c in course_codes if c in session_map]
            print_formatted_schedule(schedule_for_program)
    else:
        # fallback: iterate dynamic_groups if programs_df absent
        for prog_id, required_subjects in dynamic_groups.items():
            print(f"\nSchedule for {prog_id}:")
            schedule_for_program = [session_map[c] for c in required_subjects if c in session_map]
            print_formatted_schedule(schedule_for_program)

if __name__ == '__main__':
    main()
