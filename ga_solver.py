import random
from deap import base, creator, tools, algorithms
from collections import deque
USE_TABU_SEARCH_POLISH=True


def solve_with_ga(university_data, student_groups, seed_solution=None):
    
    # --- 1. Unpack Data from university_data ---
    scheduled_courses = university_data['scheduled_courses']
    course_enrollments = university_data['course_enrollments']
    faculty = university_data['faculty']
    rooms = university_data['rooms']
    all_courses_info = university_data['all_courses']
    preferences = university_data.get("preferences", {})
    prof_prefs = preferences.get("professors", {}) 
    course_prefs = preferences.get("courses", {})

    # Create dynamic lists from the loaded data
    COURSE_LIST = [c['course_code'] for c in scheduled_courses]
    PROF_LIST = list(faculty.keys())
    ROOM_LIST = list(rooms.keys())
    TIMESLOTS = [
    'Mon_10AM', 'Mon_11AM', 'Mon_12PM', 'Mon_1PM', 'Mon_2PM', 'Mon_3PM', 'Mon_4PM', 'Mon_5PM',
    'Tue_10AM', 'Tue_11AM', 'Tue_12PM', 'Tue_1PM', 'Tue_2PM', 'Tue_3PM', 'Tue_4PM', 'Tue_5PM',
    'Wed_10AM', 'Wed_11AM', 'Wed_12PM', 'Wed_1PM', 'Wed_2PM', 'Wed_3PM', 'Wed_4PM', 'Wed_5PM',
    'Thu_10AM', 'Thu_11AM', 'Thu_12PM', 'Thu_1PM', 'Thu_2PM', 'Thu_3PM', 'Thu_4PM', 'Thu_5PM',
    'Fri_10AM', 'Fri_11AM', 'Fri_12PM', 'Fri_1PM', 'Fri_2PM', 'Fri_3PM', 'Fri_4PM', 'Fri_5PM',
    'Sat_10AM', 'Sat_11AM', 'Sat_12PM', 'Sat_1PM', 'Sat_2PM', 'Sat_3PM', 'Sat_4PM', 'Sat_5PM'
] 
    
    TIMESLOT_MAP = {
    'Mon_10AM': 1, 'Mon_11AM': 2, 'Mon_12PM': 3, 'Mon_1PM': 4, 'Mon_2PM': 5, 'Mon_3PM': 6, 'Mon_4PM': 7, 'Mon_5PM': 8,
    'Tue_10AM': 9, 'Tue_11AM': 10, 'Tue_12PM': 11, 'Tue_1PM': 12, 'Tue_2PM': 13, 'Tue_3PM': 14, 'Tue_4PM': 15, 'Tue_5PM': 16,
    'Wed_10AM': 17, 'Wed_11AM': 18, 'Wed_12PM': 19, 'Wed_1PM': 20, 'Wed_2PM': 21, 'Wed_3PM': 22, 'Wed_4PM': 23, 'Wed_5PM': 24,
    'Thu_10AM': 25, 'Thu_11AM': 26, 'Thu_12PM': 27, 'Thu_1PM': 28, 'Thu_2PM': 29, 'Thu_3PM': 30, 'Thu_4PM': 31, 'Thu_5PM': 32,
    'Fri_10AM': 33, 'Fri_11AM': 34, 'Fri_12PM': 35, 'Fri_1PM': 36, 'Fri_2PM': 37, 'Fri_3PM': 38, 'Fri_4PM': 39, 'Fri_5PM': 40,
    'Sat_10AM': 41, 'Sat_11AM': 42, 'Sat_12PM': 43, 'Sat_1PM': 44, 'Sat_2PM': 45, 'Sat_3PM': 46, 'Sat_4PM': 47, 'Sat_5PM': 48
}
    
    # --- 2. DEAP Toolbox Setup ---
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    def create_random_session():
        course = random.choice(COURSE_LIST)
        professor = random.choice(PROF_LIST)
        room = random.choice(ROOM_LIST)
        timeslot = random.choice(TIMESLOTS)
        return (course, professor, room, timeslot)

    toolbox.register("attr_session", create_random_session)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_session, n=len(COURSE_LIST))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # --- 3. Advanced Fitness Function ---
    # In ga_solver.py, this function goes inside solve_with_ga

    def evaluate_timetable(individual):
        score = 1000.0
        conflicts = []
        professor_schedule, room_schedule, group_schedule = set(), set(), set()
        
        # Initialize daily schedules for gap calculation
        group_daily_schedules = {group: {'Mon': [], 'Tue': [], 'Wed': []} for group in student_groups.keys()}

        for i, session in enumerate(individual):
            # Unpack the session details
            course_code, prof_id, room_id, timeslot = session
            
            # --- Advanced Hard Constraints ---
            if course_code not in faculty[prof_id]['expertise']:
                score -= 1000; conflicts.append(i)
            if course_enrollments.get(course_code, 0) > rooms[room_id]['capacity']:
                score -= 1000; conflicts.append(i)
            day = timeslot.split('_')[0]
            if faculty[prof_id].get('availability', {}).get(day) == 'unavailable':
                score -= 1000; conflicts.append(i)

            # --- Standard Clash Constraints ---
            prof_entry = (prof_id, timeslot)
            if prof_entry in professor_schedule:
                score -= 100; conflicts.append(i)
            else:
                professor_schedule.add(prof_entry)
            
            room_entry = (room_id, timeslot)
            if room_entry in room_schedule:
                score -= 100; conflicts.append(i)
            else:
                room_schedule.add(room_entry)
            
            # --- Student Group Logic (Clashes and Gap Data Collection in ONE place) ---
            time_val = TIMESLOT_MAP.get(timeslot)
            for group, courses_taken in student_groups.items():
                if course_code in courses_taken:
                    # Check for clashes
                    group_entry = (group, timeslot)
                    if group_entry in group_schedule:
                        score -= 100; conflicts.append(i)
                    else:
                        group_schedule.add(group_entry)
                    
                    # Collect data for gap analysis
                    if time_val and day in group_daily_schedules[group]:
                        group_daily_schedules[group][day].append(time_val)

            # --- Soft Constraints ---
            if prof_id in prof_prefs:
                disliked_slots = preferences["professors"][prof_id].get('dislikes_timeslot', [])
                if timeslot in disliked_slots:
                    score -= 10 
                liked_slots = preferences["professors"][prof_id].get('likes_timeslot', [])
                if timeslot in liked_slots:
                    score += 5

        # --- Post-Loop Calculation for Student Gaps ---
        for daily_schedule in group_daily_schedules.values():
            for times in daily_schedule.values():
                if len(times) > 1:
                    sorted_times = sorted(times)
                    for i in range(len(sorted_times) - 1):
                        gap = sorted_times[i+1] - sorted_times[i]
                        if gap > 1:
                            score -= 5 * (gap - 1)
                            
        individual.conflicts = list(set(conflicts))
        return (score,)

    # --- 4. Genetic Operators ---
    def mutate_timetable(individual):
        if hasattr(individual, 'conflicts') and individual.conflicts:
            index_to_fix = random.choice(individual.conflicts)
            session_to_fix = individual[index_to_fix]
            new_timeslot = random.choice(TIMESLOTS)
            course, professor, room, _ = session_to_fix
            individual[index_to_fix] = (course, professor, room, new_timeslot)
        return individual,

    toolbox.register("evaluate", evaluate_timetable)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_timetable)
    toolbox.register("select", tools.selTournament, tournsize=3) 

    
    # --- Tabu Search Integration ---
    # --- 4. Helper Functions for Tabu Search ---
    def generate_neighborhood(timetable, size=20):
        neighborhood = []
        num_classes = len(timetable)
        if num_classes < 2: return []
        for _ in range(size):
            neighbor = toolbox.clone(timetable)
            i, j = random.sample(range(num_classes), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            del neighbor.fitness.values
            neighborhood.append(neighbor)
        return neighborhood

    def tabu_search(initial_timetable, iterations=100, tabu_size=7):
        best_solution = toolbox.clone(initial_timetable)
        current_solution = toolbox.clone(initial_timetable)
        tabu_list = deque(maxlen=tabu_size)

        for _ in range(iterations):
            neighborhood = generate_neighborhood(current_solution)
            best_neighbor_in_loop = None
            for neighbor in neighborhood:
                # BUG FIX: Convert list of tuples to tuple of tuples to make it hashable
                neighbor_tuple = tuple(map(tuple, neighbor))
                if neighbor_tuple not in tabu_list:
                    neighbor.fitness.values = evaluate_timetable(neighbor)
                    if best_neighbor_in_loop is None or neighbor.fitness.values > best_neighbor_in_loop.fitness.values:
                        best_neighbor_in_loop = neighbor
            if best_neighbor_in_loop is None: break
            current_solution = best_neighbor_in_loop
            tabu_list.append(tuple(map(tuple, current_solution)))
            if current_solution.fitness.values > best_solution.fitness.values:
                best_solution = toolbox.clone(current_solution)
        return best_solution
    
    # --- 5. Running the GA ---
    print("--- Starting Genetic Algorithm Evolution ---")
    if seed_solution:
        print("Seeding GA population with SAT solver solution...")
       
        seed_individual = creator.Individual(seed_solution)
        
        # Now, create the population by cloning this proper DEAP Individual
        pop = [toolbox.clone(seed_individual) for _ in range(100)]
        # Apply heavy mutation to the first generation to create diversity
        for i in range(1, len(pop)): # Don't mutate the original seed
            toolbox.mutate(pop[i])
            del pop[i].fitness.values
    else:
        print("Starting with a random GA population...")
        pop = toolbox.population(n=100)
    
    # --- PERFORMANCE FIX: Use the fast, built-in algorithm ---
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=False)
    
    best_individual = tools.selBest(pop, 1)[0]
    print("--- Genetic Algorithm Finished ---")

    # --- 6. Optional Polishing Step ---
    if USE_TABU_SEARCH_POLISH:
        print("\n--- Polishing Best Solution with Tabu Search ---")
        final_solution = tabu_search(best_individual, iterations=200, tabu_size=10)
        print("--- Polishing Finished ---")
    else:
        final_solution = best_individual

    # --- 7. Final Analysis ---
    print("\n--- Final Best Timetable Found ---")
    
    print("\n--- Best Timetable Found (GA) ---")
    best_individual.sort(key=lambda session: TIMESLOT_MAP.get(session[3], 99))
    for session in best_individual:
        print(f"  {session[3]}: {session[0]} with {session[1]} in {session[2]}")
    
    final_score = best_individual.fitness.values[0]
    print(f"\nFinal Fitness Score: {final_score}")
    if final_score >= 1000:
        print("✅ This timetable has no hard conflicts.")
    else:
        print("⚠️ This timetable still has hard conflicts.")  
    final_solution = best_individual # or final_solution from Tabu Search
    
    # --- The Output Generation Logic is now here ---
   
    # Generate the final, structured output
    prof_timetables = {prof_id: [] for prof_id in university_data['faculty'].keys()}
    prog_timetables = {group_id: [] for group_id in student_groups.keys()}
    
    # Create a map for fast lookup of scheduled subjects
    session_map = {session[0]: session for session in final_solution}

    # Populate Professor Timetables by iterating through the final solution
    for session in final_solution:
        prof_id = session[1]
        if prof_id in prof_timetables:
            prof_timetables[prof_id].append(session)

    # Populate Program Timetables by using the correct data relationships
    for prog_id, required_subjects in student_groups.items():
        for subject in required_subjects:
            if subject in session_map:
                prog_timetables[prog_id].append(session_map[subject])
    
    # Return a single dictionary containing all results
    return {
        "master_timetable": final_solution,
        "professor_timetables": prof_timetables,
        "program_timetables": prog_timetables
    }
    