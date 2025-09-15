from ortools.sat.python import cp_model

def solve_with_or_tools(university_data, student_groups, time_limit=10):
    
    # --- 1. Unpack Data ---
    all_courses = university_data['all_courses']
    course_enrollments = university_data['course_enrollments']
    faculty = university_data['faculty']
    rooms = university_data['rooms']

    COURSE_CODES = list(all_courses.keys())
    PROF_IDS = list(faculty.keys())
    ROOM_IDS = list(rooms.keys())
    TIMESLOTS = [
    'Mon_10AM', 'Mon_11AM', 'Mon_12PM', 'Mon_1PM', 'Mon_2PM', 'Mon_3PM', 'Mon_4PM', 'Mon_5PM',
    'Tue_10AM', 'Tue_11AM', 'Tue_12PM', 'Tue_1PM', 'Tue_2PM', 'Tue_3PM', 'Tue_4PM', 'Tue_5PM',
    'Wed_10AM', 'Wed_11AM', 'Wed_12PM', 'Wed_1PM', 'Wed_2PM', 'Wed_3PM', 'Wed_4PM', 'Wed_5PM',
    'Thu_10AM', 'Thu_11AM', 'Thu_12PM', 'Thu_1PM', 'Thu_2PM', 'Thu_3PM', 'Thu_4PM', 'Thu_5PM',
    'Fri_10AM', 'Fri_11AM', 'Fri_12PM', 'Fri_1PM', 'Fri_2PM', 'Fri_3PM', 'Fri_4PM', 'Fri_5PM',
    'Sat_10AM', 'Sat_11AM', 'Sat_12PM', 'Sat_1PM', 'Sat_2PM', 'Sat_3PM', 'Sat_4PM', 'Sat_5PM'
]
    preferences = university_data.get("preferences", {})
    prof_prefs = preferences.get("professors", {}) 
    course_prefs = preferences.get("courses", {})
    # --- 2. Create Model & Variables ---
    model = cp_model.CpModel()
    sessions = {}
    for c in COURSE_CODES:
        for p in PROF_IDS:
            if c in faculty[p]['expertise']:
                for r in ROOM_IDS:
                    for t in TIMESLOTS:
                        # (Your availability check logic would go here)
                        sessions[(c, p, r, t)] = model.NewBoolVar(f'session_{c}_{p}_{r}_{t}')

    # --- 3. Add Constraints ---
    
    # Constraints for courses, professors, rooms, and groups using safe .get()
    for c in COURSE_CODES:
        model.Add(sum(sessions.get((c, p, r, t), 0) for p in PROF_IDS for r in ROOM_IDS for t in TIMESLOTS) == 1)
    for p in PROF_IDS:
        for t in TIMESLOTS:
            model.Add(sum(sessions.get((c, p, r, t), 0) for c in COURSE_CODES for r in ROOM_IDS) <= 1)
    for r in ROOM_IDS:
        for t in TIMESLOTS:
            model.Add(sum(sessions.get((c, p, r, t), 0) for c in COURSE_CODES for p in PROF_IDS) <= 1)
    for group, courses_in_group in student_groups.items():
        for t in TIMESLOTS:
            model.Add(sum(sessions.get((c, p, r, t), 0) for c in courses_in_group for p in PROF_IDS for r in ROOM_IDS) <= 1)

    # Correct Room Capacity Constraint
    for (c, p, r, t), session_var in sessions.items():
        enrollment = course_enrollments.get(c, 0)
        capacity = rooms[r].get('capacity', 0)
        model.Add(enrollment <= capacity).OnlyEnforceIf(session_var)

    # --- REMOVED the redundant, incorrect capacity constraint block ---

    # --- 4. Add Objective Function ---
    objective_terms = []
    
    # Here's how 'prof_prefs' and 'course_prefs' are used:
    for (c, p, r, t), session_var in sessions.items():
        # Handle professor dislikes (negative score)
        if p in prof_prefs:
            disliked_slots = prof_prefs[p].get('dislikes_timeslot', [])
            if t in disliked_slots:
                objective_terms.append(-10 * session_var)

        # Handle course room preferences (positive score)
        if c in course_prefs:
            preferred_rooms = course_prefs[c].get('prefers_room', [])
            if r in preferred_rooms:
                objective_terms.append(5 * session_var)

    model.Maximize(sum(objective_terms))
    
    # --- 5. Solve and Return Solution ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    
    solution = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('--- Solution Found ---')
        # FIX: Iterate over sessions.items() for safety
        for (c, p, r, t), session_var in sessions.items():
            if solver.Value(session_var) == 1:
                print(f'  {t}: {c} with {p} in {r}')
                solution.append((c, p, r, t))
        prof_timetables = {prof_id: [] for prof_id in university_data['faculty'].keys()}
        prog_timetables = {group_id: [] for group_id in student_groups.keys()}
        
        session_map = {session[0]: session for session in solution}

        for session in solution:
            prof_id = session[1]
            if prof_id in prof_timetables:
                prof_timetables[prof_id].append(session)

        for prog_id, required_subjects in student_groups.items():
            for subject in required_subjects:
                if subject in session_map:
                    prog_timetables[prog_id].append(session_map[subject])

        # Return a single dictionary containing all results
        return {
            "master_timetable": solution,
            "professor_timetables": prof_timetables,
            "program_timetables": prog_timetables
        }
    
    else:
        print('--- No solution found. The model is likely INFEASIBLE. ---')
        
        # --- NEW: Infeasibility Debugging ---
        
        print('Calculating sufficient assumptions for infeasibility...')
        assumptions = solver.SufficientAssumptionsForInfeasibility()
        
        if not assumptions:
            print("Could not determine the exact conflict. Check for large-scale issues.")
        else:
            print("Conflict found! The following constraints cannot be satisfied simultaneously:")
            for assumption_index in assumptions:
                # The solver gives you an index of the conflicting constraint.
                # You can use this to get the constraint's name from the model's prototype.
                print(f"  - {model.Proto().constraints[assumption_index]}")
        
        return None
        