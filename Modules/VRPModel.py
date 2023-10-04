import math
from ortools.linear_solver import pywraplp
import time
from Assets.Data.Pricing.pricing_structure import remove_accents
import pandas as pd

class Heuristic:
    def __init__(self, adjacency_matrix,prices ,customer_datas, vehicle_data,output_df):
        self.adjacency_matrix = adjacency_matrix
        self.price_df = prices
        self.price_df['Location'] = self.price_df['Location'].apply(lambda x: remove_accents(x).lower())
        self.customer_datas = customer_datas
        print(self.customer_datas)
        self.vehicle_data = vehicle_data
        self.records = []

        self.routes_all_dict = {}
        vehicle_types = ['Iveco', 'Truck', 'Bulk', 'Lorry']

        for vechile in vehicle_types:
            self.all_routes = []
            self.generate_routes('kutahya', [],vechile)
            self.routes_all_dict[vechile] = self.all_routes
        print(self.routes_all_dict)
        dfs_list = []

        self.pre_df = pd.DataFrame(
            columns=['musteri kodu', 'siparis no', 'sevk tarihi', 'musteri', 'koli', 'palet', 'sevk adresi', 'arac id',
                     'arac tipi'])
        self.vehicle_counter = 0  # to generate unique vehicle IDs

        self.preprocess_customer_data()
        self.customer_dicts = self.organize_customers()
        print(self.customer_dicts)

        for group_key, group_data in self.customer_dicts.items():
            self.group_key = group_key
            self.customer_data = group_data
            print(group_key)
            print(group_data)
            self.cities_with_demand = set()
            self.output_df = output_df
            # Iterate through the customer data
            for customer_key, customer_value in self.customer_data.items():
                cities_dict = customer_value.get('cities', {})
                for city_name, city_data in cities_dict.items():
                    total_demand = city_data.get('total_demand', 0)
                    if total_demand > 0:
                        self.cities_with_demand.add(city_name)
            self.cities_with_demand_list = list(self.cities_with_demand)



            self.segmented_routes_data = self.segment_routes()


            self.D_i = {}
            self.D_i = self.create_demand_quantity_dict()
            self.aikt = {}
            self.create_aikt()


            self.Ctk = self.create_Ckt()
            self.b_id = self.create_bid_dict()
            self.d_index,self.i_index = self.create_indices()
            self.t_index = ['Truck','Lorry','Iveco','Bulk']

            self.p_t = {'Truck': 625, 'Lorry': 400, 'Iveco': 300,'Bulk':625}
            self.n_jt = {}
            self.Vt = vehicle_data
            total_demand = 0
            # Iterating through each layer of nested dictionaries to access the demand values.
            for customer_id, customer_data in self.customer_data.items():
                for city, city_data in customer_data['cities'].items():
                    for order_no, order_data in city_data['siparis_no'].items():
                        for demand_key, demand_value in order_data.items():
                            total_demand += demand_value

            print(f'Total demand: {total_demand}')
            print(self.Vt['Truck'])
            self.j_index = [x for x in range(0, math.ceil(total_demand *1000/ self.Vt['Truck']))]
            print(self.j_index)
            self.create_fdk_dict()

            self.create_DQk()

            self.output = None
            df = self.solve()
            dfs_list.append(df)

        self.final_df = pd.concat(dfs_list)
        self.final_df = pd.concat([self.final_df,self.pre_df])

    def preprocess_customer_data(self):
        data = []
        for customer_id, customer_info in list(self.customer_datas.items()):
            for city_name, city_data in list(customer_info['cities'].items()):
                total_demand = city_data.get('total_demand', 0)
                if total_demand > 33:
                    siparis_items = list(city_data['siparis_no'].items())
                    for idx, (siparis_no1, order_data1) in enumerate(siparis_items):
                        demand_items1 = list(order_data1.items())
                        for i, (koli1, demand_quantity1) in enumerate(demand_items1):
                            if demand_quantity1 >= 32 and demand_quantity1 <= 33:
                                self.process_demand(customer_id, customer_info, city_name, siparis_no1, koli1,
                                                    demand_quantity1, data)
                            for idy, (siparis_no2, order_data2) in enumerate(siparis_items[idx:],
                                                                             start=idx):  # start from current index
                                demand_items2 = list(order_data2.items())
                                for j, (koli2, demand_quantity2) in enumerate(demand_items2):
                                    total_demand = demand_quantity1 + demand_quantity2
                                    if 32 <= total_demand <= 33:
                                        combined_koli = f'{koli1} + {koli2}' if siparis_no1 != siparis_no2 else koli2  # different siparis_no
                                        combined_siparis_no = f'{siparis_no1} + {siparis_no2}' if siparis_no1 != siparis_no2 else siparis_no2  # different siparis_no
                                        self.process_demand(customer_id, customer_info, city_name, combined_siparis_no,
                                                            combined_koli, total_demand, data)
        df = pd.DataFrame(data)
        self.pre_df = df
        print(self.pre_df)

    def process_demand(self, customer_id, customer_info, city_name, siparis_no, koli_key, demand, data):
        self.vehicle_counter += 1  # increment the counter to generate a unique vehicle ID
        vehicle_id = f'pre{self.vehicle_counter}_Truck'

        # Handle removal for combined keys
        if '+' in siparis_no:  # if combined keys
            siparis_no_parts = siparis_no.split('+')
            koli_key_parts = koli_key.split('+')
            for part_siparis_no, part_koli_key in zip(siparis_no_parts, koli_key_parts):
                cleaned_part_siparis_no = part_siparis_no.strip()  # remove leading and trailing spaces
                cleaned_part_koli_key = part_koli_key.strip()  # remove leading and trailing spaces
                # Check if the keys exist before attempting to delete
                if cleaned_part_siparis_no in self.customer_datas[customer_id]['cities'][city_name]['siparis_no'] and \
                        cleaned_part_koli_key in self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][
                    cleaned_part_siparis_no]:

                    # Obtain the individual demand
                    individual_demand = \
                    self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][cleaned_part_siparis_no][
                        cleaned_part_koli_key]

                    # Add to DataFrame
                    row_dict = {
                        'musteri kodu': customer_id,
                        'siparis no': cleaned_part_siparis_no,
                        'sevk tarihi': 'Unknown',  # Placeholder, replace with actual data if available
                        'musteri': customer_info.get('musteri_ismi', 'Unknown'),
                        'koli': cleaned_part_koli_key,
                        'palet': round(float(individual_demand), 2),
                        'sevk adresi': 'Unknown',  # Placeholder, replace with actual data if available
                        'arac id': vehicle_id,
                        'arac tipi': 'Tır'
                    }
                    data.append(row_dict)

                    # Delete from customer_datas
                    del self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][cleaned_part_siparis_no][
                        cleaned_part_koli_key]
                    # Check if further cleanup needed
                    if not self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][cleaned_part_siparis_no]:
                        del self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][cleaned_part_siparis_no]
        else:  # Single siparis_no
            # Add to DataFrame
            row_dict = {
                'musteri kodu': customer_id,
                'siparis no': siparis_no,
                'sevk tarihi': 'Unknown',  # Placeholder, replace with actual data if available
                'musteri': customer_info.get('musteri_ismi', 'Unknown'),
                'koli': koli_key,
                'palet': round(float(demand ), 2),
                'sevk adresi': 'Unknown',  # Placeholder, replace with actual data if available
                'arac id': vehicle_id,
                'arac tipi': 'Tır'
            }
            data.append(row_dict)

            # Delete from customer_datas
            if koli_key in self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][siparis_no]:
                del self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][siparis_no][koli_key]
                # Check if further cleanup needed
                if not self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][siparis_no]:
                    del self.customer_datas[customer_id]['cities'][city_name]['siparis_no'][siparis_no]

    def group_routes_by_common_city(self):
        all_routes = self.routes_all_dict['Truck']
        groups = {}  # Dictionary to hold groups of cities
        group_counter = 1  # Counter to number the groups

        def find_group(city):
            """Find the group a city belongs to."""

            for group_key, cities in groups.items():
                if city in cities:
                    return group_key
            return None

        for route in all_routes:
            cities_to_group = set(route[1:])  # Skip the first city
            grouped = False
            for city in cities_to_group:
                group_key = find_group(city)
                if group_key:
                    # If the city is already in a group, add the rest of the cities to that group
                    groups[group_key].update(cities_to_group)
                    grouped = True
                    break  # No need to check other cities in this route
            if not grouped:
                # If none of the cities are grouped yet, start a new group
                group_key = f'Group_{group_counter}'
                groups[group_key] = cities_to_group
                group_counter += 1

        # Convert sets to lists for the final output
        for group_key, cities in groups.items():
            groups[group_key] = list(cities)

        return groups

    def organize_customers(self):
        # Get grouped cities
        grouped_cities = self.group_routes_by_common_city()

        # Initialize an empty dictionary to hold grouped customer data
        grouped_customers = {}

        # A counter to label groups uniquely
        group_counter = 1

        # Map each group of cities to a unique group key
        group_keys = {f'Group_{i}': cities for i, cities in enumerate(grouped_cities.values(), start=1)}

        # Iterate through the customer data
        for musteri_numarasi, customer_value in self.customer_datas.items():
            # For each city in the customer data
            for city_name, city_data in customer_value['cities'].items():
                # Determine which group this city belongs to
                for group_key, cities in group_keys.items():
                    if city_name in cities:
                        # Check if this group key is already in grouped_customers
                        if group_key not in grouped_customers:
                            grouped_customers[group_key] = {}

                        # Check if this customer is already in this group.
                        # If not, initialize a new entry for them.
                        if musteri_numarasi not in grouped_customers[group_key]:
                            grouped_customers[group_key][musteri_numarasi] = {
                                'musteri_ismi': customer_value['musteri_ismi'],
                                'cities': {}
                            }
                        # Copy this city's data to the grouped customer data.
                        grouped_customers[group_key][musteri_numarasi]['cities'][city_name] = city_data

        return grouped_customers

    def create_output_df(self, xjtk, yijt):
        data = []  # This will hold dictionaries, each representing a row in your final DataFrame

        for i in self.i_index:
            for j in self.j_index:
                for t in self.t_index:
                    y_value = yijt[i, j, t].solution_value()
                    if y_value > 0:
                        # Use the 'ŞİPARİŞ NO' value to look up the required information in self.df
                        siparis_no = i  # You might want to replace this with the actual order number
                        matching_rows = self.output_df.loc[self.output_df['siparis no'] == siparis_no]

                        # Assuming there's exactly one matching row, so use iloc[0] to get that row
                        matching_row = matching_rows.iloc[0]

                        # Create a dictionary for each non-zero y_value
                        vehicle_type_mapping = {
                            'Truck': 'Tır',
                            'Lorry': 'Kamyon',
                            'Iveco': 'Iveco',
                            'Bulk': 'Dökme'
                        }

                        row_dict = {
                            'musteri kodu': matching_row['musteri kodu'],
                            'siparis no': siparis_no,
                            'sevk tarihi': matching_row['sevk tarihi'],
                            'musteri': matching_row['musteri'],  # Assuming 'musteri' is the correct column name
                            'koli': matching_row['koli'],
                            'palet': round(float(y_value/1000), 2),
                            'sevk adresi': matching_row['sevk adresi'],
                            'arac id': f'{j}_{self.group_key}',
                            'arac tipi' : vehicle_type_mapping.get(f'{t}', f'{t}')
                        }
                        data.append(row_dict)

        # Create a DataFrame
        df = pd.DataFrame(data)

        # If the order of columns in df does not match the order you want,
        # you can rearrange the columns like so:
        df = df[['musteri kodu', 'siparis no', 'sevk tarihi', 'musteri', 'koli', 'palet', 'sevk adresi', 'arac id','arac tipi']]

        return df

    def create_DQk(self):
        # Initialize a dictionary to store the DQk values
        self.DQk = {}

        # Iterate through the segmented_routes_data
        for vehicle_type, routes in self.segmented_routes_data.items():
            for route_key, cities_in_route in routes.items():
                # Initialize the demand quantity for this route to 0
                self.DQk[route_key] = 0

                # Iterate through the customer_data to find the demands for this route
                for customer_id, customer_data in self.customer_data.items():
                    for city, city_data in customer_data['cities'].items():
                        # Only proceed if the city is in the current route
                        if city in cities_in_route:
                            for siparis_no, siparis_data in city_data['siparis_no'].items():
                                for item_id, quantity in siparis_data.items():
                                    # Accumulate the demand quantity for this route
                                    self.DQk[route_key] += quantity*1000

    def solve(self):
        start_time = time.time()
        # Create the model
        solver = pywraplp.Solver.CreateSolver('CBC')
        solver.SetNumThreads(-1)

        # Decision variables
        xjtk = {}
        for j in self.j_index:
            for t in self.t_index:
                for k in self.k_index:
                    xjtk[j, t, k] = solver.BoolVar(f'x_{j}_{t}_{k}')

        yijt = {}
        for i in self.i_index:
            for j in self.j_index:
                for t in self.t_index:
                    yijt[i, j, t] = solver.NumVar(0, solver.infinity(), f'y_{i}_{j}_{t}')

        sdjt = {}
        for d in self.d_index:
            for j in self.j_index:
                for t in self.t_index:
                    sdjt[d, j, t] = solver.BoolVar(f'sd_{d}_{j}_{t}')

        njt = {}
        for j in self.j_index:
            for t in self.t_index:
                njt[j, t] = solver.IntVar(0, solver.infinity(), f'n_{j}_{t}')

        # Objective function
        objective = solver.Objective()
        for j in self.j_index:
            for t in self.t_index:
                for k in self.k_index:
                    coeff = float(self.Ctk[(t, k)])
                    objective.SetCoefficient(xjtk[j, t, k], coeff)

        for j in self.j_index:
            for t in self.t_index:
                objective.SetCoefficient(njt[j, t], self.p_t[t])
        objective.SetMinimization()

        for j in self.j_index:
            for t in self.t_index:
                solver.Add(sum(xjtk[j, t, k] for k in self.k_index) <= 1)

        for k in self.k_index:
            solver.Add(
                sum(self.Vt[t] * xjtk[j, t, k] for j in self.j_index for t in self.t_index) >= self.DQk[k])


        for i in self.i_index:
              solver.Add(sum(yijt[i, j, t] for j in self.j_index for t in self.t_index) == self.D_i[i])

        for j in self.j_index:
            for t in self.t_index:
                solver.Add(njt[j, t] == sum(sdjt[d, j, t] for d in self.d_index))

        M = 500000

        for i in self.i_index:
            for j in self.j_index:
                for t in self.t_index:
                    solver.Add(
                        yijt[i, j, t] <= M * sum(
                            self.aikt[(i, k, t)] * xjtk[j, t, k]
                            for k in self.k_index
                        )
                    )



        for k in self.k_index:
            solver.Add(sum(xjtk[j, t, k] for j in self.j_index for t in self.t_index) >= 1)



        for i in self.i_index:
            for j in self.j_index:
                for t in self.t_index:
                    solver.Add(sum(M * self.b_id[(i, d)] * sdjt[d, j, t] for d in self.d_index
                    ) >= yijt[i, j, t])



        for j in self.j_index:
            for t in self.t_index:
                solver.Add(sum(yijt[i, j, t] for i in self.i_index) <= self.Vt[t])

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print(f'Total cost = {objective.Value()}')
            for j in self.j_index:
                for t in self.t_index:
                    for k in self.k_index:
                        x_value = xjtk[j, t, k].solution_value()
            for d in self.d_index:
                for j in self.j_index:
                    for t in self.t_index:
                        s_value = sdjt[d, j, t].solution_value()
            total_y = 0
            for i in self.i_index:
                for j in self.j_index:
                    for t in self.t_index:
                        y_value = yijt[i, j, t].solution_value()
                        if y_value > 0:
                            total_y+= y_value
            return self.create_output_df(xjtk, yijt)
        else:
            print('The problem does not have an optimal solution.')

        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {elapsed_time} seconds')

    def create_indices(self):
        i = []
        d = []
        for customer_no, customer_data in self.customer_data.items():
            for city, city_data in customer_data['cities'].items():
                customer_city_index = f'{customer_no}_{city}'  # Create a combined index for customer and city
                if customer_city_index not in d:
                    d.append(customer_city_index)  # Append to the d list if it's a new customer-city pair
                for siparis_no in city_data['siparis_no']:
                    if siparis_no not in i:
                        i.append(siparis_no)  # Append to the i list if it's a new demand
        return d,i

    def create_Ckt(self):
        Ckt = {}
        segmented_routes = self.segment_routes()
        for vehicle_type, routes in segmented_routes.items():
            for route_key, route in routes.items():
                last_city = route[-1]  # The last city in the route
                # Assuming a cost of 0 if the city or vehicle type is not found in the price data
                price = self.price_df.set_index('Location').get(vehicle_type, {}).get(last_city, 0)
                Ckt[(vehicle_type, route_key)] = price
        return Ckt


    def generate_routes(self,current_city, current_route,vechile):
        current_route.append(current_city)

        # Get neighbors of the current city
        neighbors = self.adjacency_matrix[vechile].loc[current_city]
        has_neighbors = False
        for neighbor_city, value in neighbors.items():
            if value > 0 and neighbor_city not in current_route:  # Check if the neighbor_city is not already in the current_route to prevent cycles
                has_neighbors = True
                self.generate_routes(neighbor_city, current_route,vechile)

        if not has_neighbors:
            # If a city has no unvisited neighbors, the route is complete.
            # Append a copy of the current route to the routes list.
            self.all_routes.append(list(current_route))

        # Important: remove the last appended city before exiting this recursion level.
        current_route.pop()

    def get_all_siparis_no(self):
        # Assuming all possible siparis_no values are unique and can be gathered from self.customer_data
        all_siparis_no = set()
        for customer_info in self.customer_data.values():
            for city_data in customer_info['cities'].values():
                all_siparis_no.update(city_data.get('siparis_no', []))
        return list(all_siparis_no)

    def create_bid_dict(self):
        bid = {}  # Initialize the dictionary

        # Assuming all_siparis_no is a list of all possible siparis_no values
        all_siparis_no = self.get_all_siparis_no()

        for customer_id, customer_info in self.customer_data.items():
            for city, city_data in customer_info['cities'].items():
                # Create a unique key for customer and city combination
                customer_city_key = f'{customer_id}_{city}'

                # Check each siparis_no to see if it exists for the current customer_city combination
                for siparis_no in all_siparis_no:
                    if siparis_no in city_data.get('siparis_no', []):
                        # Assign a value of 1 to indicate demand i belongs to customer d at the specified city
                        bid[(siparis_no, customer_city_key)] = 1
                    else:
                        # Assign a value of 0 to indicate no demand i for customer d at the specified city
                        bid[(siparis_no, customer_city_key)] = 0

        return bid

    def convert_routes(self):
        routes_dict = {}
        for i, route in enumerate(self.all_routes):
            key = f'route{i + 1}'
            segments = [[route[i], route[i + 1]] for i in range(len(route) - 1)]
            routes_dict[key] = segments
        return routes_dict

    def create_demand_quantity_dict(self):
        Demandi_Di = {}  # Initialize the dictionary

        for customer_info in self.customer_data.values():
            for city_data in customer_info['cities'].values():
                for siparis_no, items in city_data['siparis_no'].items():
                    demand_quantity = sum(items.values())
                    Demandi_Di[
                        siparis_no] = int(demand_quantity * 1000) # Assign the demand quantity to the corresponding demand number

        return Demandi_Di

    def get_all_route_keys_and_vehicle_types(self):
        all_route_keys = set()
        all_vehicle_types = set()
        for vehicle_type, routes in self.segmented_routes_data.items():
            all_vehicle_types.add(vehicle_type)
            all_route_keys.update(routes.keys())
        return list(all_route_keys), list(all_vehicle_types)

    def create_fdk_dict(self):
        f_dk = {}  # Initialize the dictionary

        # Collect all unique route keys
        route_keys_set = set()
        for vehicle_type, routes in self.segmented_routes_data.items():
            for route_key in routes.keys():
                route_keys_set.add(route_key)

        # Iterate through customer data and check each route
        for customer_id, customer_info in self.customer_data.items():
            for city, city_data in customer_info['cities'].items():
                # Create a unique key for customer and city combination
                customer_city_key = f'{customer_id}_{city}'

                for route_key in route_keys_set:
                    # Retrieve route data using route_key
                    route = self.get_route_by_key(route_key)

                    if city in route:  # Check if the city is in the route
                        # Assign a value of 1 to indicate customer d is in route k
                        f_dk[(customer_city_key, route_key)] = 1
                    else:
                        # Assign a value of 0 to indicate customer d is not in route k
                        f_dk[(customer_city_key, route_key)] = 0

        self.f_dk= f_dk

    def get_route_by_key(self, route_key):
        # Implement logic to return the route data for the given route_key
        for vehicle_type, routes in self.segmented_routes_data.items():
            if route_key in routes:
                return routes[route_key]
        return None

    def create_aikt(self):
        # Initialize an empty set to collect unique route keys
        route_keys_set = set()

        # Initialize an empty dictionary to store the aikt values
        self.aikt = {}

        # Get all possible siparis_no values
        all_siparis_no = self.get_all_siparis_no()

        # Initialize aikt to 0 for all possible combinations
        for vehicle_type, routes in self.segmented_routes_data.items():
            for route_key, route in routes.items():
                # Add the route_key to the set
                route_keys_set.add(route_key)

                for siparis_no in all_siparis_no:
                    # Initialize to 0 by default
                    self.aikt[(siparis_no, route_key, vehicle_type)] = 0

                    for customer_id, customer_data in self.customer_data.items():
                        for city, city_data in customer_data['cities'].items():
                            if city in route:  # Only proceed if the city is in the route
                                if siparis_no in city_data['siparis_no']:
                                    # Set a value of 1 in aikt for the corresponding indices
                                    self.aikt[(siparis_no, route_key, vehicle_type)] = 1
                                    # You can also include code here to handle the demand values, if needed

        # Convert the set to a list to finalize the k_index
        self.k_index = list(route_keys_set)

    def segment_routes(self):
        segmented_routes = {}
        for vehicle, routes in self.routes_all_dict.items():
            segmented_routes[vehicle] = {}
            route_id = 1  # Reset route_id for each vehicle
            for route in routes:
                for i in range(len(route)):
                    subroute = route[0:i + 2]  # Subroute includes cities from the start up to the current city
                    # Check if the last city in the subroute is in cities_with_demand_list
                    if subroute[-1] in self.cities_with_demand_list:
                        # If the city is Hatay, and Mersin is in the subroute, remove Mersin
                        if subroute[-1] == 'hatay' and 'mersin' in subroute:
                            subroute.remove('mersin')
                        # Avoid adding duplicate routes
                        if subroute not in segmented_routes[vehicle].values():
                            subroute_key = f'route{route_id}'
                            segmented_routes[vehicle][subroute_key] = subroute
                            route_id += 1  # Increment route_id only when a route is added
        return segmented_routes


