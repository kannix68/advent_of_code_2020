#!/usr/bin/env python
# coding: utf-8

# # Advent of Code 2020
# 
# This solution (Jupyter notebook; python3.7) by kannix68, @ 2020-12.  \
# Using anaconda distro, conda v4.9.2. installation on MacOS v10.14.6 "Mojave".

# ## Generic AoC code

# In[ ]:


import sys
print("Python version:", sys.version)
print("Version info:", sys.version_info)


# In[ ]:


DEBUG_FLAG = 0
VERBOSE_LEVEL = 1


# In[ ]:


# Code functions

def assert_msg(msg, assertion):
  """Assert boolean condition with message, ok=message printed, otherwise assertion-fail."""
  assert assertion, "ERROR on assert: {}".format(msg)
  print("assert-OK: {}".format(msg))

def expect_msg(msg, expected, inputs):
  assert_msg(msg.format(inputs, expected), expected == solve(inputs))

def log_debug(*args,**kwargs):
  """Print message only if DEBUG_FLAG > 0."""
  if DEBUG_FLAG > 0:
    print('D: ', end='')
    print(*args,**kwargs)

def log_info(*args,**kwargs):
  """Print message only if VERBOSE_LEVEL > 0."""
  if VERBOSE_LEVEL > 0:
    print('I: ', end='')
    print(*args,**kwargs)

def log_error(*args,**kwargs):
  """Print error message."""
  print('E: ', end='')
  print(*args,**kwargs)

def read_file_to_str(filename):
  """Read a file's content into one string."""
  with open(filename, 'r') as inputfile:
    data = inputfile.read()
  return data

def read_file_to_list(filename):
  """Read a file's content into a list of strings (per line), each ending whitespace stripped."""
  with open(filename, 'r') as inputfile:
    lines_list = inputfile.readlines()
  #lines_list = [line.rstrip('\n') for line in lines_list] # via list comprehension
  lines_list = list(map(lambda it: it.rstrip(), lines_list)) # via map
  return lines_list

def lrange(*args,**kwargs):
  return list(range(*args,**kwargs))

def lmap(*args,**kwargs):
  return list(map(*args,**kwargs))


# ## Problem domain code

# ### Day 1: Report Repair

# In[ ]:


print("Day 1 a")


# In[ ]:


THIS_YEAR = 2020 # "Last christmas, I gave you my heart... this year..." - Wham!


# In[ ]:


test_str = """
1721
979
366
299
675
1456""".strip()
tests = list(map(int, test_str.split("\n")))
log_debug(tests)


# In[ ]:


import itertools
#from operator import mul
import numpy as np


# In[ ]:


def solve01a(l):
  for v in itertools.combinations(l, 2):
    v = np.array(v) # using numpy for elegance, array "object" methods .sum() and .prod()
    #print(v)
    if v.sum() == THIS_YEAR:
      log_info(f"found {v}")
      p = v.prod()
      log_debug(f"product={p}")
      break
  return p


# In[ ]:


result = solve01a(tests)
print("tests solution", result)


# In[ ]:


ins = list(map(int, read_file_to_list('./in/day01.in')))
#ins


# In[ ]:


result = solve01a(ins)
print("Day 1 a solution:", result)


# In[ ]:


def solve01b(l):
  for v in itertools.combinations(l, 3):
    v = np.array(v)
    #print(v)
    if v.sum() == THIS_YEAR:
      log_info(f"found {v}")
      p = v.prod() #np.prod(np.array(v))
      log_debug(f"product={p}")
      break
  return p


# In[ ]:


print("Day 1 b")
print("test results:", solve01b(tests))


# In[ ]:


print("Day 1 b solution:", solve01b(ins))


# ### Day 2: Password Philosophy

# In[ ]:


print("Day 2 a")


# In[ ]:


test_str = """
1-3 a: abcde
1-3 b: cdefg
2-9 c: ccccccccc
""".strip()
tests = test_str.split("\n")
#tests


# In[ ]:


def solve02a(l):
  ct = 0
  for line in l:
    rules, pwd = line.split(': ')
    nums, char = rules.split(' ')
    min_num, max_num = map(int, nums.split('-'))
    #print(min_num, max_num, char, pwd)
    num_ocur = pwd.count(char)
    if num_ocur >= min_num and num_ocur <= max_num:
      #print("  pwd is valid")
      ct += 1
    #else:
    #  print("  pwd is INvalid")
  log_debug(f"num of valid passwords={ct}")
  return ct


# In[ ]:


result = solve02a(tests)
print("tests result:", result)


# In[ ]:


ins = read_file_to_list('./in/day02.in')
print("Day 2 a solution:", solve02a(ins))


# In[ ]:


def solve02b(l):
  ct = 0
  for line in l:
    rules, pwd = line.split(': ')
    nums, char = rules.split(' ')
    min_num, max_num = map(int, nums.split('-'))
    #print(min_num, max_num, char, pwd)
    num_ocur = pwd[min_num-1].count(char) + pwd[max_num-1].count(char)
    if num_ocur == 1:
      #print("  pwd is valid")
      ct += 1
    #else:
    #  print("  pwd is INvalid")
  log_debug(f"num of valid passwords={ct}")
  return ct


# In[ ]:


print("Day 2 b")
print("assert day 2 b test conditions")
assert( 1 == solve02b([tests[0]]) )
assert( 0 == solve02b([tests[1]]) )
assert( 0 == solve02b([tests[2]]) )
print("assertions were ok.")


# In[ ]:


print("tests result:", solve02b(tests))


# In[ ]:


print("Day 2 b solution:", solve02b(ins))


# ### Day 3: Toboggan Trajectory

# In[ ]:


print("Day 3 a")


# In[ ]:


test_str = """
..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#
""".strip()
tests = test_str.split("\n")
log_debug(tests)


# In[ ]:


def prepare_input(l):
  outlist = []
  for line in l:
    outlist.append(list(map(lambda it: 1 if it == '#' else 0, list(line))))
  return outlist
tests = prepare_input(tests)
log_debug(tests)


# In[ ]:


def solve03a(l2d):
  num_rows = len(l2d)
  num_cols = len(l2d[0])
  log_info(f"num rows={num_rows}, cols={num_cols}")
  posx, posy = [0, 0]
  dx, dy = [3, 1]
  ct = 0
  tpath = ''
  for iter in range(1, num_rows+2):
    #print(f"iter {iter}")
    if l2d[posy][posx%num_cols] == 1:
      ct += 1
      tpath += 'X'
    else:
      tpath += '0'
    posx += dx
    posy += dy
    #print(f"new pos={[posx, posy]}")
    if posy > num_rows-1:
      log_debug(f"break at iter#={iter}")
      break
    else:
      iter += 1
  outstr = f"encountered {ct} trees."
  if DEBUG_FLAG > 0:
    outstr += f"Path={tpath}"
  log_info(outstr)
  return ct


# In[ ]:


print("Day 3 a tests:")
print(solve03a(tests))


# In[ ]:


ins = prepare_input(read_file_to_list('./in/day03.in'))


# In[ ]:


result = solve03a(ins)
print("Day 3 a solution:", result)


# In[ ]:


def solve03b(l2d, vec):
  num_rows = len(l2d)
  num_cols = len(l2d[0])
  log_debug(f"num rows={num_rows}, cols={num_cols}, vector={vec}")
  posx, posy = [0, 0]
  dx, dy = vec #reversed(vec)
  ct = 0
  for iter in range(0, num_rows+1):
    #print(f"i={iter} @{[posx, posy]} : {l2d[posy][posx%num_cols]}")
    if l2d[posy][posx%num_cols] == 1:
      ct += 1
    posx += dx
    posy += dy
    if posy > num_rows-1:
      log_debug(f"break at iter#={iter}")
      break
    else:
      iter += 1
  log_debug(f"encountered {ct} trees.")
  return ct


# In[ ]:


print("Day 3 b")
#print("number of trees encountered:", solve3b(tests, [3, 1]))


# In[ ]:


print("assert day 3 b test conditions:")
assert( 2 == solve03b(tests, [1, 1]))
assert( 7 == solve03b(tests, [3, 1]))
assert( 3 == solve03b(tests, [5, 1]))
assert( 4 == solve03b(tests, [7, 1]))
assert( 2 == solve03b(tests, [1, 2]))
print("assertions were ok.")


# In[ ]:


p = solve03b(tests, [1, 1]) * solve03b(tests, [3, 1]) * solve03b(tests, [5, 1])     * solve03b(tests, [7, 1]) * solve03b(tests, [1, 2])
print("day 3 b test result (product):", p)


# In[ ]:


p = solve03b(ins, [1, 1]) * solve03b(ins, [3, 1]) * solve03b(ins, [5, 1])       * solve03b(ins, [7, 1]) * solve03b(ins, [1, 2])
print("day 3 b solution (product):", p)


# ### Day 4: Passport Processing

# In[ ]:


DEBUG_FLAG = 0


# In[ ]:


fields_mandat = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
fields_opt = {'cid'}


# In[ ]:


test_str = """
ecl:gry pid:860033327 eyr:2020 hcl:#fffffd
byr:1937 iyr:2017 cid:147 hgt:183cm

iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884
hcl:#cfa07d byr:1929

hcl:#ae17e1 iyr:2013
eyr:2024
ecl:brn pid:760753108 byr:1931
hgt:179cm

hcl:#cfa07d eyr:2025 pid:166559648
iyr:2011 ecl:brn hgt:59in
""".strip()
tests = test_str.split("\n\n")
log_debug(tests)


# In[ ]:


import re


# In[ ]:


def passport_valid(passport):
  entries = re.split(r'\s+', passport)
  log_debug(entries)
  fields = []
  for entry in entries:
    field = entry.split(':')[0]
    fields.append(field)
  #log_debug(sorted(fields))
  b = fields_mandat.issubset(fields)
  log_debug("valid?:", b)
  return b


# In[ ]:


def solve04a(passports):
  ct = 0
  for passport in passports:
    if passport_valid(passport):
      ct +=1
  log_debug("valid-count:", ct)
  return ct


# In[ ]:


print("tests valid-count:", solve04a(tests))


# In[ ]:


ins = read_file_to_str('./in/day04.in').split("\n\n")
print("Day 4 a solution: valid-count:", solve04a(ins))


# In[ ]:


def passport_valid2(passport):
  entries = re.split(r'\s+', passport)
  log_debug(entries)
  fields = []
  values = []
  for entry in entries:
    field, val = entry.split(':')
    fields.append(field)
    values.append(val)
  #log_debug(sorted(fields))
  if not fields_mandat.issubset(fields):
    log_debug("invalid: mandatory fields missing")
    return False 
  for idx, field in enumerate(fields):
    val = values[idx]
    if field == 'byr':
      # byr (Birth Year) - four digits; at least 1920 and at most 2002.
      ival = int(val)
      if not (ival >= 1920 and ival <= 2002):
        log_debug(f"invalid: byr value {val}")
        return False
    elif field == 'iyr':
      # iyr (Issue Year) - four digits; at least 2010 and at most 2020.
      ival = int(val)
      if not (ival >= 2010 and ival <= THIS_YEAR):
        log_debug(f"invalid: iyr value {val}")
        return False
    elif field == 'eyr':
      # eyr (Expiration Year) - four digits; at least 2020 and at most 2030
      ival = int(val)
      if not (ival >= THIS_YEAR and ival <= 2030):
        log_debug(f"invalid: eyr value {val}")
        return False
    elif field == 'hgt':
      # hgt (Height) - a number followed by either cm or in:
      # - If cm, the number must be at least 150 and at most 193.
      # - If in, the number must be at least 59 and at most 76.
      # py-regex: ^(\d+)(?=cm|in)(cm|in)$
      if not re.match(r'^\d+(cm|in)$', val):
        log_debug(f"invalid: hgt val={val}, form.")
        return False
      numstr, unit = re.split(r'(?=cm|in)', val)
      num = int(numstr)
      if unit == 'cm':
        if not (num >= 150 and num <= 193):
          log_debug(f"invalid: hgt val={val} num={num}")
          return False
      elif unit == 'in':
        if not (num >= 59 and num <= 76):
          log_debug(f"invalid: hgt val={val} num={num}")
          return False
      else:
        log_debug(f"invalid: hgt val={val} unit={unit}")
        return False
    elif field == 'hcl':
      # hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
      if not re.match(r'^#[0-9a-f]{6}$', val):
        log_debug(f"invalid: hcl value {val}")
        return False
    elif field == 'ecl':
      # ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
      if not val in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']:
        log_debug(f"invalid: ecl value {val}")
        return False
    elif field == 'pid':
      # pid (Passport ID) - a nine-digit number, including leading zeroes.
      if not re.match(r'^[0-9]{9}$', val):
        log_debug(f"invalid: pid value {val}")
        return False
  log_debug("valid!")
  return True


# In[ ]:


tests_invalid = """
eyr:1972 cid:100
hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926

iyr:2019
hcl:#602927 eyr:1967 hgt:170cm
ecl:grn pid:012533040 byr:1946

hcl:dab227 iyr:2012
ecl:brn hgt:182cm pid:021572410 eyr:2020 byr:1992 cid:277

hgt:59cm ecl:zzz
eyr:2038 hcl:74454a iyr:2023
pid:3556412378 byr:2
""".strip().split("\n\n")


# In[ ]:


print("tests, all invalid:")
for passport in tests_invalid:
  print(passport.replace("\n", " "))
  print("valid?:", passport_valid2(passport))
  print()


# In[ ]:


tests_valid = """
pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980
hcl:#623a2f

eyr:2029 ecl:blu cid:129 byr:1989
iyr:2014 pid:896056539 hcl:#a97842 hgt:165cm

hcl:#888785
hgt:164cm byr:2001 iyr:2015 cid:88
pid:545766238 ecl:hzl
eyr:2022

iyr:2010 hgt:158cm hcl:#b6652a ecl:blu byr:1944 eyr:2021 pid:093154719
""".strip().split("\n\n")


# In[ ]:


print("tests, all valid:")
for passport in tests_valid:
  print(passport.replace("\n", " "))
  print("valid?:", passport_valid2(passport))
  print()


# In[ ]:


def solve04b(passports):
  ct = 0
  for passport in passports:
    log_debug(passport)
    if passport_valid2(passport):
      ct +=1
  log_debug("valid-count:", ct)
  return ct


# In[ ]:


assert( 0 == solve04b(tests_invalid) )


# In[ ]:


assert( 4 == solve04b(tests_valid) )


# In[ ]:


result = solve04b(ins)
print("Day 4 b result:", result)


# ### Day 5: Binary Boarding

# In[ ]:


import functools
import operator

# see: [python - How to make a flat list out of list of lists? - Stack Overflow](https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
def flatten_list(l):
  """Flatten a list."""
  return functools.reduce(operator.iconcat, l, [])

def get_seat_id(s):
  rows = lrange(0, 128)
  cols = lrange(0, 8)
  #log_debug(cols)
  for c in s:
    if c == 'F':
      rows = rows[:len(rows)//2]
    elif c == 'B':
      rows = rows[len(rows)//2:]
    elif c == 'L':
      cols = cols[:len(cols)//2]
    elif c == 'R':
      cols = cols[len(cols)//2:]
  result_list = flatten_list([rows, cols])
  log_debug(result_list)
  return result_list[0]*8 + result_list[1]


# In[ ]:


boardingpass = 'FBFBBFFRLR'
get_seat_id(boardingpass)


# In[ ]:


# Given tests:
assert(357 == get_seat_id('FBFBBFFRLR'))


# In[ ]:


assert(567 == get_seat_id('BFFFBBFRRR'))
assert(119 == get_seat_id('FFFBBBFRRR'))
assert(820 == get_seat_id('BBFFBBFRLL'))


# In[ ]:


ins = read_file_to_list('./in/day05.in')
print( "Day 5 a solution:", max(map(get_seat_id, ins)) )


# In[ ]:


print("number of boarding passes given:", (len(ins)))
#print("number of used rows in plane:", (len(ins)+1)/8.0)

min_seat_id = 0*8 + 0  # from min row and min column/seat
max_seat_id = 127*8 + 7  # from max row and max column/seat
print("seat_id min/max", [min_seat_id, max_seat_id])


# In[ ]:


seat_ids = lrange(min_seat_id, max_seat_id+1)
for boardingpass in ins: # remove used/given seat_id
  seat_ids.remove(get_seat_id(boardingpass))
log_debug("ids remain unseen:")
log_debug(seat_ids)
for seat_id in seat_ids:
  if not( (seat_id-1) in seat_ids and (seat_id>min_seat_id) )     and not( (seat_id+1) in seat_ids and (seat_id<max_seat_id) ):
    print("(Day 5 b solution) found id:", seat_id)


# ### Day 6: Custom Customs

# In[ ]:


DEBUG_FLAG = 0


# In[ ]:


test_str = """
abcx
abcy
abcz
""".strip()
test = test_str.split("\n")
log_debug(test)


# In[ ]:


from collections import defaultdict


# In[ ]:


def get_group_answers(answers_in):
  answers = defaultdict(int)
  for tanswers in answers_in:
    for tanswer in tanswers:
      answers[tanswer] += 1
  log_debug(answers)
  log_debug(len(answers.keys()), answers.keys())
  return answers


# In[ ]:


print("testing...", get_group_answers(test))


# In[ ]:


assert( 6 == len(get_group_answers(test).keys()) )


# In[ ]:


test_str = """
abc

a
b
c

ab
ac

a
a
a
a

b
""".strip()
tests = test_str.split("\n\n")
log_debug(tests)


# In[ ]:


def solve06a(groupanswers):
  i = 0
  for groupanswer in groupanswers:
    result = get_group_answers(groupanswer.split("\n")).keys()
    log_debug(f"distinctanswers={result} for {groupanswer}")
    i += len(result)
  log_debug(f"answerssum={i}")
  return i


# In[ ]:


assert( 11 == solve06a(tests) )
print("test assertion ok.")


# In[ ]:


ins = read_file_to_str('./in/day06.in').split("\n\n")
print("Day 6 a solution: groupanwers-sum:", solve06a(ins))


# In[ ]:


print("Day 6 b")


# In[ ]:


def get_group_answers2(answers_in):
  answers = defaultdict(int)
  num_persons = len(answers_in)
  for tanswers in answers_in:
    for tanswer in tanswers:
      answers[tanswer] += 1
  log_debug(answers)
  log_debug(len(answers.keys()), answers.keys())
  ct = 0
  #for idx, (key, val) in enumerate(d.items()):
  for key, val in answers.items():
    if val == num_persons:
      ct += 1
  return ct

def solve06b(groupanswers):
  i = 0
  for groupanswer in groupanswers:
    result = get_group_answers2(groupanswer.split("\n"))
    log_debug(f"all-answers={result} for {groupanswer}")
    i += result
  log_info(f"all-answers-sum={i}")
  return i


# In[ ]:


assert( 6 == solve06b(tests) )
print("test assertion ok.")


# In[ ]:


print("Day 6 b solution: groupanwers-sum:", solve06b(ins))


# ### Day 7: Handy Haversacks

# In[ ]:


DEBUG_FLAG = 0


# In[ ]:


import networkx as nx


# In[ ]:


test_str = """
light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags.
""".strip()
tests = test_str.split("\n")
log_debug(test)


# In[ ]:


def get_bag_graph(l):
  graph = nx.DiGraph()
  for line in l:
    try:
      src, trg = line.split(" bags contain ")
    except ValueError:
      log_error(f"parse error, input=>{line}<")
    bags_contained = trg.replace(".", "").split(", ")
    if not (len(bags_contained) == 1 and bags_contained[0].startswith("no other")):
      graph.add_node(src)
      for idx, bag_in in enumerate(bags_contained):
        rxm = re.match(r"^(\d+)\s+(.*?)\s+bag", bag_in)
        res = [int(rxm.group(1)), rxm.group(2)]
        #log_debug("src:", src, "; trg:", res)
        bags_contained[idx] = res
        graph.add_node(res[1])
        #log_debug(f"add_edge {src} => {res[0]} {res[1]}")
        graph.add_edge(src, res[1], weight=res[0])
    else:
      graph.add_edge(src, "END", weight=0)
    #print(src, bags_contained)
  log_info( "graph # of nodes:", len(graph.nodes()) )
  log_info( "graph # of edges:", len(graph.edges()) )
  return graph


# In[ ]:


graph = get_bag_graph(tests)
for e in graph.edges():
  log_debug(e, nx.get_edge_attributes(graph, 'weight')[e])


# In[ ]:


def get_paths_to(graph, trg):
  paths = []
  for src in graph.nodes():
    #log_debug("src:", src)
    for p in nx.all_simple_paths(graph, src, trg):
      paths.append(p)
  return paths


# In[ ]:


def solve07a(l, trg):
  graph = get_bag_graph(l)
  sources = lmap(lambda it: it[0], get_paths_to(graph, trg))
  num_sources = len(set(sources))
  return num_sources


# In[ ]:


trg = 'shiny gold'
assert( 4 == solve07a(tests, trg) )


# In[ ]:


ins = read_file_to_str('./in/day07.in').strip().split("\n")
print("Day 7 a solution: num-distinct-src-colors", solve07a(ins, 'shiny gold'))


# In[ ]:


print("Day 7 b")


# In[ ]:


edge_weights = nx.get_edge_attributes(graph, 'weight')

#for p in nx.all_simple_edge_paths(graph, 'shiny gold', "END"): # not available
seen_subpaths = []
for p in nx.all_simple_paths(graph, 'shiny gold', "END"):
  log_debug(p)
  for snode_idx in range(len(p)-1):
    tup = tuple([p[snode_idx], p[snode_idx+1]])
    subpath = tuple(p[0:snode_idx+2])
    log_debug("subpath:", subpath)
    if not subpath in seen_subpaths:
      seen_subpaths.append(subpath)
      log_debug("    new subpath")
    else:
      log_debug("    already SEEN subpath")
    log_debug(f"  path-edge#{snode_idx}: {tup} {edge_weights[tup]}")
  log_debug(seen_subpaths)
  


# In[ ]:


# see: [python - Getting subgraph of nodes between two nodes? - Stack Overflow](https://stackoverflow.com/questions/32531117/getting-subgraph-of-nodes-between-two-nodes)
def subgraph_between(graph, start_node, end_node):
  paths_between_generator = nx.all_simple_paths(graph, source=start_node,target=end_node)
  nodes_between_set = {node for path in paths_between_generator for node in path}
  return( graph.subgraph(nodes_between_set) )


# In[ ]:


subgraph = subgraph_between(graph, 'shiny gold', 'END')
for p in subgraph.edges:
  log_debug(p)
log_info("sub-paths for shiny gold:")
for p in nx.all_simple_paths(subgraph, 'shiny gold', "END"):
  log_info(p)


# In[ ]:


edge_weights = nx.get_edge_attributes(graph, 'weight')
seen_subpaths = []
for p in nx.all_simple_paths(graph, 'shiny gold', "END"):
  log_debug(p)
  for start_idx in reversed(range(len(p)-2)):
    seen = False
    subpath = tuple(p[0:start_idx+2])
    if not subpath in seen_subpaths:
      seen_subpaths.append(subpath)
    else:
      seen = True
    tup = tuple([p[start_idx], p[start_idx+1]])
    w = edge_weights[tup]
    log_debug(f"  subedge={tup}, weight={w}; subpath={subpath}, seen={seen}")


# In[ ]:


# Personal solution to day 7 a UNFINISHED.
clr = 'shiny gold'
clr_edges = filter(lambda it: it[0]==clr, list(graph.edges))
for edge in clr_edges:
  log_debug(edge, edge_weights[edge])


# In[ ]:


# "Inspiration" soltion, copied/stolen from user el-guish's solution in:
# [- 2020 Day 07 Solutions - : adventofcode](https://www.reddit.com/r/adventofcode/comments/k8a31f/2020_day_07_solutions/)
# Using recursion.

rules = open('in/day07.in').readlines()

def parse_rule(r):
  parent, contents = r[:-2].split(' bags contain ')
  childs =  [parse_child_bag(c) for c in contents.split(',') if c != 'no other bags' and c != 'no other bag']
  return (parent, childs)

def parse_child_bag(child_st):
  cparts = child_st.split()
  qty = int(cparts[0])
  color = ' '.join(cparts[1:-1])
  return (color, qty)

def required_contents(bag_color):
  return sum(q + q * required_contents(color) for color, q in contains[bag_color] )

contains = dict(parse_rule(r) for r in test_str.split("\n"))
log_debug("test rules (parsed):", contains)
print("tests result", required_contents('shiny gold'))

contains = dict(parse_rule(r) for r in rules)
print("Day 7 b solution", required_contents('shiny gold'))


# In[ ]:




