import pandas as pd
import re
import random
import math
import os

################################################
num_hash_functions = 275 # r = 25 & b = 11
upper_bound_on_number_of_distinct_elements  = 10000000
#upper_bound_on_number_of_distinct_elements =   138492
#upper_bound_on_number_of_distinct_elements =  3746518

################################################


### primality checker
def is_prime(number):
	if number == 2:
		return True
	if (number % 2) == 0:
		return False
	for j in range(3, int(math.sqrt(number)+1), 2):
		if (number % j) == 0:
			return False
	return True


set_of_all_hash_functions = set()
while len(set_of_all_hash_functions) < num_hash_functions:
	a = random.randint(1, upper_bound_on_number_of_distinct_elements-1)
	b = random.randint(0, upper_bound_on_number_of_distinct_elements-1)
	p = random.randint(upper_bound_on_number_of_distinct_elements, 10*upper_bound_on_number_of_distinct_elements)
	while is_prime(p) == False:
		p = random.randint(upper_bound_on_number_of_distinct_elements, 10*upper_bound_on_number_of_distinct_elements)
	#
	current_hash_function_id = tuple([a, b, p])
	set_of_all_hash_functions.add(current_hash_function_id)

path_ = os.getcwd()
f = open(path_ + '/hash_functions/275.tsv', 'w')
f.write("a\tb\tp\tn" + "\n")
for a, b, p in set_of_all_hash_functions:
	f.write(str(a) + "\t" + str(b) + "\t" + str(p) + "\t" + str(upper_bound_on_number_of_distinct_elements) + "\n")
f.close()
