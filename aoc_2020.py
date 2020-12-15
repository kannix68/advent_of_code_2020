#!/usr/bin/env python
# coding: utf-8

# # Advent of Code 2020
# 
# This solution (Jupyter notebook; python3.7) by kannix68, @ 2020-12.  \
# Using anaconda distro, conda v4.9.2. installation on MacOS v10.14.6 "Mojave".

# ## Generic AoC code

# In[ ]:


import sys
import logging

import lib.aochelper as aoc
from lib.aochelper import map_list as mapl
from lib.aochelper import filter_list as filterl

print("Python version:", sys.version)
print("Version info:", sys.version_info)

log = aoc.getLogger(__name__)
print(f"initial log-level={log.getEffectiveLevel()}")

EXEC_RESOURCE_HOGS = False
EXEC_EXTRAS = False


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
log.warning(tests)


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
      log.info(f"found {v}")
      p = v.prod()
      log.debug(f"product={p}")
      break
  return p


# In[ ]:


result = solve01a(tests)
print("tests solution", result)


# In[ ]:


ins = list(map(int, aoc.read_file_to_list('./in/day01.in')))
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
      log.info(f"found {v}")
      p = v.prod() #np.prod(np.array(v))
      log.debug(f"product={p}")
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
  log.debug(f"num of valid passwords={ct}")
  return ct


# In[ ]:


result = solve02a(tests)
print("tests result:", result)


# In[ ]:


ins = aoc.read_file_to_list('./in/day02.in')
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
  log.debug(f"num of valid passwords={ct}")
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
log.debug(tests)


# In[ ]:


def prepare_input(l):
  outlist = []
  for line in l:
    outlist.append(list(map(lambda it: 1 if it == '#' else 0, list(line))))
  return outlist
tests = prepare_input(tests)
log.debug(tests)


# In[ ]:


def solve03a(l2d):
  num_rows = len(l2d)
  num_cols = len(l2d[0])
  log.info(f"num rows={num_rows}, cols={num_cols}")
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
      log.debug(f"break at iter#={iter}")
      break
    else:
      iter += 1
  outstr = f"encountered {ct} trees."
  if log.getEffectiveLevel() <= logging.DEBUG:
    outstr += f"Path={tpath}"
  log.info(outstr)
  return ct


# In[ ]:


print("Day 3 a tests:")
print(solve03a(tests))


# In[ ]:


ins = prepare_input(aoc.read_file_to_list('./in/day03.in'))


# In[ ]:


result = solve03a(ins)
print("Day 3 a solution:", result)


# In[ ]:


def solve03b(l2d, vec):
  num_rows = len(l2d)
  num_cols = len(l2d[0])
  log.debug(f"num rows={num_rows}, cols={num_cols}, vector={vec}")
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
      log.debug(f"break at iter#={iter}")
      break
    else:
      iter += 1
  log.debug(f"encountered {ct} trees.")
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
log.debug(tests)


# In[ ]:


import re


# In[ ]:


def passport_valid(passport):
  entries = re.split(r'\s+', passport)
  log.debug(entries)
  fields = []
  for entry in entries:
    field = entry.split(':')[0]
    fields.append(field)
  #log.debug(sorted(fields))
  b = fields_mandat.issubset(fields)
  log.debug(f"valid?: {b}")
  return b


# In[ ]:


def solve04a(passports):
  ct = 0
  for passport in passports:
    if passport_valid(passport):
      ct +=1
  log.debug(f"valid-count: {ct}")
  return ct


# In[ ]:


print("tests valid-count:", solve04a(tests))


# In[ ]:


ins = aoc.read_file_to_str('./in/day04.in').split("\n\n")
print("Day 4 a solution: valid-count:", solve04a(ins))


# In[ ]:


def passport_valid2(passport):
  entries = re.split(r'\s+', passport)
  log.debug(entries)
  fields = []
  values = []
  for entry in entries:
    field, val = entry.split(':')
    fields.append(field)
    values.append(val)
  #log.debug(sorted(fields))
  if not fields_mandat.issubset(fields):
    log.debug("invalid: mandatory fields missing")
    return False 
  for idx, field in enumerate(fields):
    val = values[idx]
    if field == 'byr':
      # byr (Birth Year) - four digits; at least 1920 and at most 2002.
      ival = int(val)
      if not (ival >= 1920 and ival <= 2002):
        log.debug(f"invalid: byr value {val}")
        return False
    elif field == 'iyr':
      # iyr (Issue Year) - four digits; at least 2010 and at most 2020.
      ival = int(val)
      if not (ival >= 2010 and ival <= THIS_YEAR):
        log.debug(f"invalid: iyr value {val}")
        return False
    elif field == 'eyr':
      # eyr (Expiration Year) - four digits; at least 2020 and at most 2030
      ival = int(val)
      if not (ival >= THIS_YEAR and ival <= 2030):
        log.debug(f"invalid: eyr value {val}")
        return False
    elif field == 'hgt':
      # hgt (Height) - a number followed by either cm or in:
      # - If cm, the number must be at least 150 and at most 193.
      # - If in, the number must be at least 59 and at most 76.
      # py-regex: ^(\d+)(?=cm|in)(cm|in)$
      if not re.match(r'^\d+(cm|in)$', val):
        log.debug(f"invalid: hgt val={val}, form.")
        return False
      numstr, unit = re.split(r'(?=cm|in)', val)
      num = int(numstr)
      if unit == 'cm':
        if not (num >= 150 and num <= 193):
          log.debug(f"invalid: hgt val={val} num={num}")
          return False
      elif unit == 'in':
        if not (num >= 59 and num <= 76):
          log.debug(f"invalid: hgt val={val} num={num}")
          return False
      else:
        log.debug(f"invalid: hgt val={val} unit={unit}")
        return False
    elif field == 'hcl':
      # hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
      if not re.match(r'^#[0-9a-f]{6}$', val):
        log.debug(f"invalid: hcl value {val}")
        return False
    elif field == 'ecl':
      # ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
      if not val in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']:
        log.debug(f"invalid: ecl value {val}")
        return False
    elif field == 'pid':
      # pid (Passport ID) - a nine-digit number, including leading zeroes.
      if not re.match(r'^[0-9]{9}$', val):
        log.debug(f"invalid: pid value {val}")
        return False
  log.debug("valid!")
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
    log.debug(passport)
    if passport_valid2(passport):
      ct +=1
  log.debug(f"valid-count: {ct}")
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
  rows = aoc.range_list(0, 128)
  cols = aoc.range_list(0, 8)
  #log.debug(cols)
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
  log.debug(result_list)
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


ins = aoc.read_file_to_list('./in/day05.in')
print( "Day 5 a solution:", max(map(get_seat_id, ins)) )


# In[ ]:


print("number of boarding passes given:", (len(ins)))
#print("number of used rows in plane:", (len(ins)+1)/8.0)

min_seat_id = 0*8 + 0  # from min row and min column/seat
max_seat_id = 127*8 + 7  # from max row and max column/seat
print("seat_id min/max", [min_seat_id, max_seat_id])


# In[ ]:


seat_ids = aoc.range_list(min_seat_id, max_seat_id+1)
for boardingpass in ins: # remove used/given seat_id
  seat_ids.remove(get_seat_id(boardingpass))
log.debug("ids remain unseen:")
log.debug(seat_ids)
for seat_id in seat_ids:
  if not( (seat_id-1) in seat_ids and (seat_id>min_seat_id) )     and not( (seat_id+1) in seat_ids and (seat_id<max_seat_id) ):
    print("(Day 5 b solution) found id:", seat_id)


# ### Day 6: Custom Customs

# In[ ]:


test_str = """
abcx
abcy
abcz
""".strip()
test = test_str.split("\n")
log.debug(test)


# In[ ]:


from collections import defaultdict


# In[ ]:


def get_group_answers(answers_in):
  answers = defaultdict(int)
  for tanswers in answers_in:
    for tanswer in tanswers:
      answers[tanswer] += 1
  #log.debug(answers)
  #log.debug(f"len={len(answers.keys())}, vals={answers.keys()}")
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
log.debug(tests)


# In[ ]:


def solve06a(groupanswers):
  i = 0
  for groupanswer in groupanswers:
    result = get_group_answers(groupanswer.split("\n")).keys()
    #log.debug(f"distinctanswers={result} for {groupanswer}")
    i += len(result)
  log.info(f"answerssum={i}")
  return i


# In[ ]:


assert( 11 == solve06a(tests) )
print("test assertion ok.")


# In[ ]:


ins = aoc.read_file_to_str('./in/day06.in').split("\n\n")
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
  #log.debug(answers)
  #log.debug(len(answers.keys()), answers.keys())
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
    #log.debug(f"all-answers={result} for {groupanswer}")
    i += result
  log.info(f"all-answers-sum={i}")
  return i


# In[ ]:


assert( 6 == solve06b(tests) )
print("test assertion ok.")


# In[ ]:


print("Day 6 b solution: groupanwers-sum:", solve06b(ins))


# ### Day 7: Handy Haversacks

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
log.debug(test)


# In[ ]:


def get_bag_graph(l):
  graph = nx.DiGraph()
  for line in l:
    try:
      src, trg = line.split(" bags contain ")
    except ValueError:
      log.error(f"parse error, input=>{line}<")
    bags_contained = trg.replace(".", "").split(", ")
    if not (len(bags_contained) == 1 and bags_contained[0].startswith("no other")):
      graph.add_node(src)
      for idx, bag_in in enumerate(bags_contained):
        rxm = re.match(r"^(\d+)\s+(.*?)\s+bag", bag_in)
        res = [int(rxm.group(1)), rxm.group(2)]
        #log.debug("src:", src, "; trg:", res)
        bags_contained[idx] = res
        graph.add_node(res[1])
        #log.debug(f"add_edge {src} => {res[0]} {res[1]}")
        graph.add_edge(src, res[1], weight=res[0])
    else:
      graph.add_edge(src, "END", weight=0)
    #print(src, bags_contained)
  log.info( f"graph # of nodes: {len(graph.nodes())}" )
  log.info( f"graph # of edges: {len(graph.edges())}" )
  return graph


# In[ ]:


graph = get_bag_graph(tests)
for e in graph.edges():
  log.debug(f"  edge: {e} attrs={nx.get_edge_attributes(graph, 'weight')[e]}")


# In[ ]:


def get_paths_to(graph, trg):
  paths = []
  for src in graph.nodes():
    #log.debug("src:", src)
    for p in nx.all_simple_paths(graph, src, trg):
      paths.append(p)
  return paths


# In[ ]:


def solve07a(l, trg):
  graph = get_bag_graph(l)
  sources = aoc.map_list(lambda it: it[0], get_paths_to(graph, trg))
  num_sources = len(set(sources))
  return num_sources


# In[ ]:


trg = 'shiny gold'
assert( 4 == solve07a(tests, trg) )


# In[ ]:


ins = aoc.read_file_to_str('./in/day07.in').strip().split("\n")
print("Day 7 a solution: num-distinct-src-colors", solve07a(ins, 'shiny gold'))


# In[ ]:


print("Day 7 b")


# In[ ]:


edge_weights = nx.get_edge_attributes(graph, 'weight')

#for p in nx.all_simple_edge_paths(graph, 'shiny gold', "END"): # not available
seen_subpaths = []
for p in nx.all_simple_paths(graph, 'shiny gold', "END"):
  log.debug(p)
  for snode_idx in range(len(p)-1):
    tup = tuple([p[snode_idx], p[snode_idx+1]])
    subpath = tuple(p[0:snode_idx+2])
    log.debug(f"subpath: {subpath}")
    if not subpath in seen_subpaths:
      seen_subpaths.append(subpath)
      log.debug("    new subpath")
    else:
      log.debug("    already SEEN subpath")
    log.debug(f"  path-edge#{snode_idx}: {tup} {edge_weights[tup]}")
  log.debug(seen_subpaths)
  


# In[ ]:


# see: [python - Getting subgraph of nodes between two nodes? - Stack Overflow](https://stackoverflow.com/questions/32531117/getting-subgraph-of-nodes-between-two-nodes)
def subgraph_between(graph, start_node, end_node):
  paths_between_generator = nx.all_simple_paths(graph, source=start_node,target=end_node)
  nodes_between_set = {node for path in paths_between_generator for node in path}
  return( graph.subgraph(nodes_between_set) )


# In[ ]:


subgraph = subgraph_between(graph, 'shiny gold', 'END')
for p in subgraph.edges:
  log.debug(p)
log.info("sub-paths for shiny gold:")
for p in nx.all_simple_paths(subgraph, 'shiny gold', "END"):
  log.info(p)


# In[ ]:


edge_weights = nx.get_edge_attributes(graph, 'weight')
seen_subpaths = []
for p in nx.all_simple_paths(graph, 'shiny gold', "END"):
  log.debug(p)
  for start_idx in reversed(range(len(p)-2)):
    seen = False
    subpath = tuple(p[0:start_idx+2])
    if not subpath in seen_subpaths:
      seen_subpaths.append(subpath)
    else:
      seen = True
    tup = tuple([p[start_idx], p[start_idx+1]])
    w = edge_weights[tup]
    log.debug(f"  subedge={tup}, weight={w}; subpath={subpath}, seen={seen}")


# In[ ]:


# Personal solution to day 7 a UNFINISHED.
clr = 'shiny gold'
clr_edges = filter(lambda it: it[0]==clr, list(graph.edges))
for edge in clr_edges:
  log.debug(f"edge={edge}, edge-weight={edge_weights[edge]}")


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
log.debug("test rules (parsed):", contains)
print("tests result", required_contents('shiny gold'))

contains = dict(parse_rule(r) for r in rules)
print("Day 7 b solution", required_contents('shiny gold'))


# ### Day 8: Handheld Halting

# In[ ]:


def read_prog(l):
  outlst = aoc.map_list(lambda it: it.split(' '), l)
  for instr in outlst:
    instr[1] = int(instr[1])
  return outlst


# In[ ]:


def run_cpu_prog(prog):
  cpuct = 0
  pptr = 0
  prog_len = len(prog)
  seen = []
  acc = 0
  while True:
    cpuct += 1
    if pptr in seen:
      log.info(f"found inf-loop @cpuct={cpuct} @instr#={pptr} : {instr}")
      break
    elif pptr == prog_len:
      log.info(f"found prog-term @cpuct={cpuct} @instr#={pptr} : {instr}")
      break
    else:
      seen.append(pptr)
    instr = prog[pptr]
    op, par = instr
    log.debug(f"instr#{cpuct} instr={instr}")
    if cpuct > 10_000:
      raise Exception("failsafe")
    if op == 'nop':
      pptr += 1
      #log.debug(f"  new pptr={pptr}")
    elif op == 'acc':
      acc += par
      pptr += 1
      #log.debug(f"  new acc={acc}")
    elif op == 'jmp':
      pptr += par
      #log.debug(f"  jmp for={par} to={pptr}")
    else:
      raise Exception(f"unknown opcode in {instr}")
  return acc


# In[ ]:


tests = """
nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6
""".strip().split("\n")
log.debug(tests)
test_prog = read_prog(tests)
log.debug(test_prog)


# In[ ]:


run_cpu_prog(test_prog)


# In[ ]:


ins = aoc.read_file_to_str('./in/day08.in').strip().split("\n")
prog = read_prog(ins)
print("Day 8 a solution: acc:", run_cpu_prog(prog))


# In[ ]:


print("Day 8 b")


# In[ ]:


def check_cpu_prog(prog):
  prog_len = len(prog)
  cpuct = 0
  pptr = 0
  seen = []
  acc = 0
  while True:
    if pptr == prog_len:
      log.debug(f"OK: prog terminates! @cpuct={cpuct} @instr#={pptr} : last-instr={instr}")
      return True
    cpuct += 1
    instr = prog[pptr]
    op, par = instr
    #log.debug(f"instr#{cpuct} {instr}")
    if pptr in seen:
      log.debug(f"Fail: found inf-loop @cpuct={cpuct} @instr#={pptr} : {instr}")
      return False
    else:
      seen.append(pptr)
    if cpuct > 10_000:
      raise Exception("failsafe")
    if op == 'nop':
      pptr += 1
      #log.debug(f"  new pptr={pptr}")
    elif op == 'acc':
      acc += par
      pptr += 1
      #log.debug(f"  new acc={acc}")
    elif op == 'jmp':
      pptr += par
      #log.debug(f"  jmp for={par} to={pptr}")
    else:
      raise Exception(f"unknown opcode in {instr}")
  return acc


# In[ ]:


print("test result: check-cpu-prog", check_cpu_prog(test_prog))


# In[ ]:


from copy import deepcopy

def check_prog_variations(prog):
  base_prog = deepcopy(prog)
  altinstrs = []
  for idx, instr in enumerate(base_prog):
    if instr[0] in ['nop', 'jmp']:
      altinstrs.append([idx, instr])
  log.debug(f"alternate instructions: {altinstrs}")
  
  if check_cpu_prog(base_prog):
    #log.debug("prog=", base_prog)
    acc = run_cpu_prog(base_prog)
    log.debug(f"prog ok, acc={acc}")
  for elem in altinstrs:
    #log.debug("elem:", elem)
    idx, v = elem
    instr, par = v
    prog = deepcopy(base_prog)
    if instr == 'nop':
      prog[idx][0] = 'jmp'
    elif instr == 'jmp':
      prog[idx][0] = 'nop'
    #log.debug(f"new-instr @{idx}={prog[idx][0]}")
    #log.debug("new-prog=", prog)
    if check_cpu_prog(prog):
      acc = run_cpu_prog(prog)
      log.info(f"prog ok, acc={acc}")
      break    
  return acc


# In[ ]:


result = check_prog_variations(test_prog)
print("test result: check-prog-variations", result)


# In[ ]:


result = check_prog_variations(read_prog(ins))
print("Day 8 b result: check-prog-variations", result)


# ### Day 9: Encoding Error

# In[ ]:


tests = """
35
20
15
25
47
40
62
55
65
95
102
117
150
182
127
219
299
277
309
576
""".strip().split("\n")


# In[ ]:


from typing import List

def pp_lst(lst):
  return "[" +  str.join(',', aoc.map_list(str, lst)) + "]"

def check_xmas_data(xmas_data: int, preamble: List[int]) -> bool:
  preamble_len = len(preamble)
  #log.debug("[check_xmas_data] xmas_data:", xmas_data, ", preamble_len;:", len(preamble))
  ok = False
  for combi in itertools.combinations(preamble, 2): # for entries no combination with itself!
    if sum(combi) == xmas_data:
      ok = True
      #log.info(f"[check_xmas_data] OK: xmas-data-elem {xmas_data} is sum of prev-elems:{combi}")
      break
  return ok

def check_xmas_data_seq(xmas_data_seq: List[int], preamble: List[int]) -> bool:
  #log.debug("[check_xmas_data_seq] xmas_data_seq:", xmas_data_seq, ", preamble_len;:", len(preamble))
  preamble_len = len(preamble)
  all_ok = True
  for xmas_data in xmas_data_seq:
    #log.info(f"[check_xmas_data_seq] elem={xmas_data} preamble={pp_lst(preamble)}")
    ok = check_xmas_data(xmas_data, preamble)
    preamble.pop(0)
    preamble.append(xmas_data)
    #log.info(f"  p appended={xmas_data}, removed={remvd}, preamble={str.join(',', lmap(str, preamble))}")
    all_ok &= ok
  return all_ok


# In[ ]:


preamble0 = list(range(1, 25+1)) # numbers 1..25
log.debug(preamble0)


# In[ ]:


assert( True == check_xmas_data(26, preamble0) )
assert( True == check_xmas_data(49, preamble0) )
assert( False == check_xmas_data(100, preamble0) )
assert( False == check_xmas_data(50, preamble0) )


# In[ ]:


preamble1 = flatten_list( [[20], list(range(1, 20)), list(range(21, 26))] )
log.debug(preamble1)


# In[ ]:


assert( True == check_xmas_data_seq([45, 26], preamble1) )
assert( False == check_xmas_data_seq([45, 65], preamble1) )
assert( True == check_xmas_data_seq([45, 64], preamble1) )
assert( True == check_xmas_data_seq([45, 66], preamble1) )


# In[ ]:


def verify_xmas_data_seq(xmas_data_rawseq: List[int], preamble_len=25) -> bool:
  """List `xmas_data_rawseq` contains the preamble as head."""
  preamble = xmas_data_rawseq[0:preamble_len]
  xmas_data_seq = xmas_data_rawseq[preamble_len:]
  log.debug(f"[verify_xmas_data_seq] xmas_data_seq:{pp_lst(xmas_data_seq)}, preamble:{pp_lst(preamble)}")
  preamble_len = len(preamble)
  oks = []
  for xmas_data in xmas_data_seq:
    #log.info(f"[check_xmas_data_seq] elem={xmas_data} preamble={str.join(',', lmap(str, preamble))}")
    ok = check_xmas_data(xmas_data, preamble)
    oks.append([xmas_data, ok])
    preamble.pop(0)
    preamble.append(xmas_data)
  return oks


# In[ ]:


raw_testdata = aoc.map_list(int, tests)
res = verify_xmas_data_seq(raw_testdata, preamble_len=5)
res = aoc.map_list(lambda it: it[0], aoc.filter_list(lambda it: it[1] == False, res))
log.info(f"test False results: {res}")
assert( [127] == res )


# In[ ]:


ins = aoc.map_list(int, aoc.read_file_to_list('./in/day09.in'))
log.debug(ins)


# In[ ]:


res = verify_xmas_data_seq(ins, preamble_len=25)
res = aoc.map_list(lambda it: it[0], aoc.filter_list(lambda it: it[1] == False, res))
log.info(f"found invalid number(s): {res}")
invalid_num = res[0]
print("Day 8 a solution:", invalid_num)


# In[ ]:


# see: [python - List all contiguous sub-arrays](https://stackoverflow.com/questions/41576911/list-all-contiguous-sub-arrays)
def get_all_windows(lst, min_win_len=1):
  """Generator yielding all sub-windows (contiguous sublists) of given list with min_win_len."""
  for win_len in range(min_win_len, len(lst)+1):
    for idx in range(len(lst)-win_len+1):
      yield lst[idx:idx+win_len]


# In[ ]:


test_invalidnum = 127
raw_testdata2 = raw_testdata.copy()
raw_testdata2.remove(test_invalidnum)
for subl in get_all_windows(raw_testdata2):
  if sum(subl) == test_invalidnum:
    log.info(f"found fulfilling-window: {subl}")
    break


# In[ ]:


ins2 = ins.copy()
ins2.remove(invalid_num)
for subl in get_all_windows(ins2):
  if sum(subl) == invalid_num:
    log.info(f"found fulfilling-window: {subl}")
    min_elem = min(subl)
    max_elem = max(subl)
    solut = min_elem+max_elem
    log.info(f"min, max, sum: {[min_elem, max_elem, solut]}")
    break


# ### Day 10: Adapter Array

# In[ ]:


def solve10a(loi):
  current = 0
  remainders = loi.copy()
  chain = [current]
  jolts = []
  for i in range(len(remainders)):
    targets = filterl(lambda it: it >= current and it <= current + 3, remainders)
    target = min(targets)
    remainders.remove(target)
    #log.debug(f"#{i} from={current} targets={targets}, target={target}, djolt={target-current}, remainders={remainders}")
    chain.append(target)
    jolts.append(target-current)
    current = target
    if len(remainders) == 0:
      jolts.append(3) # final device 3 jolts higher than lasta dapter in chain
      j1 = jolts.count(1)
      j3 = jolts.count(3)
      res = j1*j3
      log.info(f"chain {aoc.cl(chain)} terminated ok, jolts={aoc.cl(jolts)}, j1#={j1}, j3#={j3}, res={res}")
      return j1*j3
  raise Exception("solution not found")


# In[ ]:


tests = """
16
10
15
5
1
11
7
19
6
12
4
""".strip().split("\n")
tests1 = aoc.map_list(int, tests)

log.debug(f"test1={tests1}")
res = solve10a(tests1)
aoc.assert_msg("test 1", 7*5 == res)
log.info(f"tests1 solution: {res}")

tests = """
28
33
18
42
31
14
46
20
48
47
24
23
49
45
19
38
39
11
1
32
25
35
8
17
7
9
4
2
34
10
3
""".strip().split("\n")
log.setLevel( logging.INFO )
tests2 = mapl(int, tests)
res = solve10a(tests2)
aoc.assert_msg("test 2", 220 == res)
log.info(f"tests2 solution: {res}")


# In[ ]:


ins = mapl(int, aoc.read_file_to_list('./in/day10.in'))
res = solve10a(ins)
log.info(f"Day 10 a solution: {res}")


# In[ ]:


import time
def find_paths(loi): # loi is a list of ints (input)
  start_tm = int(time.time())
  end_elem = max(loi)
  partials = {0: [[0]]}
  found_num = 0
  current = 0
  iter = 0
  lastlevel_partials = 0 # just only for printing (debugging)
  last_partials = [[0, 1]]
  elems_avail = loi.copy()
  for lvl in range(1, len(loi)+1):
    last_partials_keys = mapl(lambda it: it[0], last_partials)
    min_last_elem = min(last_partials_keys)
    elems_avail = filterl(lambda it: it > min_last_elem, elems_avail)
    filtered_elems = {}
    last_partials_count = {}
    
    for src in sorted(set(last_partials_keys)):
      filtered_elems[src] = filterl(lambda it: it > src and it <= src + 3, elems_avail)
      last_partials_count[src] = sum(mapl(lambda it: it[1], filterl(lambda it: it[0]==src, last_partials)))

    partials_diff = len(last_partials_keys)-lastlevel_partials
    needed_tm = int(time.time()) - start_tm
    log.debug(f"level={lvl} @{needed_tm}s, found={found_num}, paths-diff={partials_diff:,} before-partials-#={len(last_partials):,}, min-last-elem={min_last_elem}, elems_avail#={len(elems_avail)}")
    log.debug(f"  last-partials-ct={last_partials_count}")
    lastlevel_partials = len(last_partials)
    partials = []
    for partial in sorted(set(last_partials_keys)): #last_partials:
      iter += 1
      if iter % 100_000_000 == 0:
        log.debug(f"at iter#={iter:,}, found#={found_num}, level={lvl}")
      #if iter > 10_000_000_000: # FAILSAFE
      #  return found
      targets = filtered_elems[partial]
      for target in targets:
        if target == end_elem:
          if found_num % 100_000 == 0:
            log.debug(f"at found# {found_num}")
          found_num += last_partials_count[partial]
        else:
          partials.append( [target, last_partials_count[partial]] )
      last_partials = partials
  log.info(f"level={lvl} @{needed_tm}s, found={found_num}, paths-diff={partials_diff:,} before-partials-#={len(last_partials):,}, min-last-elem={min_last_elem}, elems_avail#={len(elems_avail)}")
  return found_num


# In[ ]:


#log.setLevel( aoc.LOGLEVEL_TRACE )
log.debug(f"effective-log-level={log.getEffectiveLevel()}")
found = find_paths(tests1)
log.info(f"tests1 found {found} from {tests1}")
assert( 8 == found )
#found == 8


# In[ ]:


found = find_paths(tests2)
log.info(f"test2 found {found} paths") # 19208
assert( 19208 == found )


# In[ ]:


found = find_paths(ins)
log.info(f"Day 10 b solution: found {found} paths")


# ### Day 11: Seating System

# In[ ]:


#log.setLevel( aoc.LOGLEVEL_TRACE )
log.debug(f"effective-log-level={log.getEffectiveLevel()}")


# In[ ]:


import copy # for deepcopy
import hashlib

class CellularWorld:
  def __init__(self, world, store_hashes=False):
    """World object constructor, world has to be given as a list-of-lists of chars."""
    self.world = world
    self.dim = [len(world[0]), len(world)]
    self.iter_num = 0
    log.info(f'[CellularWorld] new dim={self.dim}')
    self.world = world
    self.store_hashes = store_hashes
    if self.store_hashes:
      self.world_hashes = [self.get_hash()]
  
  def repr(self):
    """Return representation str (can be used for printing)."""
    return str.join("\n", map(lambda it: str.join('', it), self.world))
  
  def set_world(self, world):
    self.world = world
    self.dim = [len(world[0]), len(world)]

  def get_hash(self):
    return hashlib.sha1(self.repr().encode()).hexdigest()

  def get_neighbors8(self, x, y):
    """Get cell's surrounding 8 neighbors, omitting boundaries."""
    log.trace(f"[CellularWorld]:get_neighbors8({x},{y})")
    dim_x = self.dim[0]
    dim_y = self.dim[1]
    neighbors = ''
    for nx in range(x-1, x+2):
      for ny in range(y-1, y+2):
        if (nx >= 0 and nx < dim_x) and (ny >= 0 and ny < dim_y) and not (nx == x and ny == y):
          #log.info(f"  neighb={[nx, ny]}")
          neighbors += self.world[ny][nx]
    return neighbors
  
  def iterate(self, steps=1):
    for i in range(steps):
      world2 = copy.deepcopy(self.world)
      for y in range(self.dim[1]):
        for x in range(self.dim[0]):
          val = self.world[y][x]
          neighbors = self.get_neighbors8(x, y)
          #log.trace(f"[{x},{y}]='{val}' nbs='{neighbors}'")
          if val == 'L' and neighbors.count('#') == 0:
            world2[y][x] = '#'
          elif val == '#' and neighbors.count('#') >= 4:
            world2[y][x] = 'L'
      self.iter_num += 1
      self.set_world(world2)
      if self.store_hashes:
        self.world_hashes.append(self.get_hash())

  def find_cycle(self, max_iter=1_000):
    """This may only be called at initial state, before any previous iterations."""
    seen = [world.repr]
    for i in range(max_iter):
      if i % 1_000 == 0:
        log.debug(f"iter# {i}, still running")
      world.iterate()
      world_repr = world.repr()
      if world_repr in seen:
        start_idx = seen.index(world_repr)
        log.info(f"found cycle @ iter={i+1}, seen-idx={start_idx}")
        return([start_idx, i+1])
      else:
        seen.append(world_repr)
    raise Exception("no world iter cycle found")

  def find_stable(self, max_iter=1_000):
    last_hash = self.get_hash()
    #log.info(f"cworld initial state: (hash={last_hash}).")
    #log.debug("world-repr=\n{cworld.repr()}")
    for i in range(1, max_iter+1):
      self.iterate()
      this_hash = self.get_hash()
      #log.debug(f"cworld state after iter#{i}, hash={this_hash}") #":\n{self.repr()}")
      if this_hash == last_hash:
        log.info(f"[CellularWorld:find_stable] BREAK on stable beginning @{i-1}")
        return True
      else:
        last_hash = this_hash
    raise Exception(f"[CellularWorld:find_stable] NO stable world iter found, after break on {max_iter} steps")


# In[ ]:


tests = """
L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL
""".strip().split("\n")

tests = mapl(list, tests)
cworld = CellularWorld(tests) #, store_hashes=True)
cworld.find_stable()
seats_occ = cworld.repr().count('#')
log.info(f"test stable occupied-seats={seats_occ}")
  


# In[ ]:


ins = aoc.read_file_to_list('./in/day11.in')
ins = mapl(list, ins)
cworld = CellularWorld(ins)
cworld.find_stable()
seats_occ = cworld.repr().count('#')
log.info(f"Day 11 a solution: stable occupied-seats={seats_occ} after {cworld.iter_num} iterations")


# In[ ]:


print("Day 11 b")

class CellularWorldDirected(CellularWorld):
  def iterate(self, steps=1):
    for i in range(steps):
      world2 = copy.deepcopy(self.world)
      for y in range(self.dim[1]):
        for x in range(self.dim[0]):
          val = self.world[y][x]
          neighbors = self.get_seen_occuppied_seats(x, y)
          if val == 'L' and neighbors == 0:
            world2[y][x] = '#'
          elif val == '#' and neighbors >= 5:
            world2[y][x] = 'L'
      self.iter_num += 1
      self.set_world(world2)
      if self.store_hashes:
        self.world_hashes.append(self.get_hash())

  def get_seen_occuppied_seats(self, x, y):
    directions = [
      [1,0], [-1,0], [0,1], [0,-1],
      [1,1], [-1,1], [1,-1], [-1,-1],
    ]
    seen = 0
    for d in directions:
      #dseen = 0
      dx, dy = d  # directions
      nx, ny = [x, y] # startpoint
      while(True):  # loop handling one direction vector
        nx, ny = [nx+dx, ny+dy]
        if nx < 0 or ny < 0 or nx >= self.dim[0] or ny >= self.dim[1]:
          break
        if "#" == self.world[ny][nx]:
          #dseen += 1
          seen += 1
          break  # in each direction, only 1 occupied can bee seen
        elif "L" == self.world[ny][nx]:
          break  # empty seats block view
    return seen

  def find_cell(self, val):
    """Find first cell containing given value, return it's `[x, y]` coordinates."""
    for y in range(self.dim[1]):
      for x in range(self.dim[0]):
        if self.world[y][x] == val:
          return [x, y]


# In[ ]:


tests = """
.......#.
...#.....
.#.......
.........
..#L....#
....#....
.........
#........
...#.....
""".strip().split("\n")

tests = mapl(list, tests)
cworld = CellularWorldDirected(tests)
log.info(f"world repr:\n{cworld.repr()}")
c = cworld.find_cell('L')
n = cworld.get_seen_occuppied_seats(c[0], c[1])
log.info(f"  empty spectator cell={c}, neib-#={n}")
assert( 8 == n )


# In[ ]:


tests = """
.............
.L.L.#.#.#.#.
.............
""".strip().split("\n")
tests = mapl(list, tests)
cworld = CellularWorldDirected(tests)
c = cworld.find_cell('L')
assert( 0 == cworld.get_seen_occuppied_seats(c[0], c[1]) )


# In[ ]:


tests = """
.##.##.
#.#.#.#
##...##
...L...
##...##
#.#.#.#
.##.##.
""".strip().split("\n")
tests = mapl(list, tests)
cworld = CellularWorldDirected(tests)
c = cworld.find_cell('L')
assert( 0 == cworld.get_seen_occuppied_seats(c[0], c[1]) )


# In[ ]:


tests = """
L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL
""".strip().split("\n")
tests = mapl(list, tests)
cworld = CellularWorldDirected(tests)

#for i in range(12):
#  log.info(f"before: 0,0 val={cworld.world[0][0]} seen-occupied-#={cworld.get_seen_occuppied_seats(0,0)}")
#  cworld.iterate()
#  log.info(f"after {cworld.iter_num} iters, hash={cworld.get_hash()}: repr:\n{cworld.repr()}")
cworld.find_stable()
log.info(f"world stable after {cworld.iter_num} iters.") #": repr:\n{cworld.repr()}")
seats_occ = cworld.repr().count('#')
assert(26 == seats_occ)
log.info(f"test stable occupied-seats={seats_occ}")


# In[ ]:


cworld = CellularWorldDirected(ins)
cworld.find_stable()
log.info(f"world stable after {cworld.iter_num} iters.") #": repr:\n{cworld.repr()}")
seats_occ = cworld.repr().count('#')
log.info(f"Day 11 b solution: stable occupied-seats={seats_occ} after {cworld.iter_num} iters")


# ### Day 12: Rain Risks

# In[ ]:


directions = ['N', 'W', 'S', 'E']
direct_vecs = {'N': [0, 1], 'W': [-1, 0], 'S': [0, -1], 'E': [1, 0]}

def dist_manhattan(pos, pos_ref):
  return abs(pos[0]-pos_ref[0]) + abs(pos[1]-pos_ref[1])

def move_ship(los):
  ship_direct = 'E'
  ship_vec = direct_vecs[ship_direct]
  pos_ref = [0, 0]
  pos = pos_ref.copy()
  for cmd_str in los:
    cmd, val = [cmd_str[0], int(cmd_str[1:])]
    log.debug(f"cmd={[cmd, val]}")
    if cmd in directions:
      vec = direct_vecs[cmd]
      pos[0] += val * vec[0]
      pos[1] += val * vec[1]
      log.debug(f"  new pos: {pos}")
    elif cmd == 'F':
      pos[0] += val * ship_vec[0]
      pos[1] += val * ship_vec[1]
      log.debug(f"  new pos: {pos}; ship_direct={ship_direct}")
    elif cmd == 'R' or cmd == 'L':
      turns = val//90
      if cmd == 'R':
        new_direct_idx = directions.index(ship_direct)-turns
      elif cmd == 'L':
        new_direct_idx = (directions.index(ship_direct)+turns) % len(directions)
      log.debug(f"cur_direct={ship_direct}:{directions.index(ship_direct)}, new_direct_idx={new_direct_idx}; cmd={cmd_str}; turns={turns}")
      ship_direct = directions[new_direct_idx]
      ship_vec = direct_vecs[ship_direct]
      log.debug(f"  new ship_direct: {ship_direct}; from turn:{cmd}")
  return dist_manhattan(pos, pos_ref)
      


# In[ ]:


tests = """
F10
N3
F7
R90
F11
""".strip().split("\n")

assert( 25 == move_ship(tests) )


# In[ ]:


ins = aoc.read_file_to_list('./in/day12.in')
res = move_ship(ins)
log.info(f"Day 12 a solution: {res}")


# In[ ]:


print("Day 12 b")

def move_ship_by_waypoint(los):
  pos_ref = [0, 0]
  waypt_pos = [10, 1]
  pos = pos_ref.copy()
  for cmd_str in los:
    cmd, val = [cmd_str[0], int(cmd_str[1:])]
    log.debug(f"cmd={[cmd, val]}")
    if cmd in directions:
      vec = direct_vecs[cmd]
      dpos = [val * vec[0], val * vec[1]]
      waypt_pos[0] += dpos[0]
      waypt_pos[1] += dpos[1]
      log.debug(f"  new waypt-rpos: {waypt_pos}")
    elif cmd == 'F':
      dpos = [val * waypt_pos[0], val * waypt_pos[1]]
      pos[0] += dpos[0]
      pos[1] += dpos[1]
      log.debug(f"  new pos: {pos}; waypt-rpos={waypt_pos}")
    elif cmd == 'R' or cmd == 'L': # rotate cartesian coordinates around origin in 90 degrees steps
      if cmd_str in ['R90', 'L270']: # rotate RIGHT
        cx, cy = waypt_pos
        waypt_pos = [cy, -cx]
      elif cmd_str in ['L90', 'R270']: # rotate LEFT
        cx, cy = waypt_pos
        waypt_pos = [-cy, cx]
      elif cmd_str in ['L180', 'R180']: # invert 180
        cx, cy = waypt_pos
        waypt_pos = [-cx, -cy]
      elif cmd_str in ['L180', 'R180']:
        cx, cy = waypt_pos
        waypt_pos = [-cx, -cy]
      else:
        raise Exception(f"unknown cmd_str={cmd_str}")
      log.debug(f"  new waypt-rpos={waypt_pos} from {[cx, cy]}")
  dist = dist_manhattan(pos, pos_ref)
  log.info(f"dist={dist}")
  return dist


# In[ ]:


assert( 286 == move_ship_by_waypoint(tests) )


# In[ ]:


log.setLevel( logging.INFO )
res = move_ship_by_waypoint(ins)
log.info(f"Day 12 b solution: {res}")


# ### Day 13: Shuttle search

# In[ ]:


tests = """
939
7,13,x,x,59,x,31,19
""".strip().split("\n")


# In[ ]:


def find_shuttle(los):
  min_wait_tm, min_bus = [99_999_999, -1]

  start_tm = int(los[0])
  shuttles = los[1].split(',')
  log.info(f"[find_shuttle] {start_tm} {shuttles}")
  for bus in shuttles:
    if bus == 'x':
      continue
    bus = int(bus)
    remainder = start_tm % bus
    if remainder == 0:
      wait_tm = 0
    else:
      wait_tm = bus - remainder
    if wait_tm < min_wait_tm:
      min_wait_tm, min_bus = [wait_tm, bus]
      log.info(f"new_min: wait_tm={wait_tm}, 4bus={bus}, rmd={remainder}, res={wait_tm * bus}")
    if wait_tm == 0:
      break
    log.debug(f"wait_tm={wait_tm}, 4bus={bus}, rmd={remainder}, res={wait_tm * bus}")
  res = min_wait_tm * min_bus
  log.info(f"MIN: wait_tm={min_wait_tm}, 4bus={min_bus}, res={res}")
  return res


# In[ ]:


find_shuttle(tests)


# In[ ]:


ins = aoc.read_file_to_list('./in/day13.in')
find_shuttle(ins)


# In[ ]:


print("Day 13 b")

def find_shuttle_offsetted(s):
  """Semi-optimized brute-force algorithm implementation."""
  start_tm = int(time.time())
  log.info(f"[find_shuttle_offsetted] {s}")
  offsets = {}
  values = {}
  for idx, val in enumerate(s.split(',')):
    if val == 'x':
      continue
    val = int(val)
    values[idx] =val  # by offset
    offsets[val] = idx  # by value
  srtvalues = list(reversed(sorted(list(values.values()))))
  max_iterator = max(srtvalues)
  max_iterator_offset = offsets[max_iterator]
  log.info(f"max_it={max_iterator}->ofst={max_iterator_offset}; srtvalues={srtvalues}, offsets={offsets}, values={values}")
  
  #values_len = len(srtvalues)
  iterator2 = srtvalues[1]
  iterator2_offset = offsets[iterator2]
  iterator3 = srtvalues[2]
  iterator3_offset = offsets[iterator3]
  print_mod_interval = 100_000_000_000
  next_print_mod = print_mod_interval
  for t in map(lambda it: it * max_iterator -max_iterator_offset, range(1, 9_000_000_000_000_000//max_iterator)):
    if (t + iterator2_offset) % iterator2 != 0        or (t + iterator3_offset) % iterator3 != 0:
      continue # "FAST EXIT" this loop-item
    if t >= next_print_mod: #idx >= next_print_mod:
      log.info(f"  calculating @{int(time.time())-start_tm:,}s ...: t#={t:,}")
      next_print_mod += print_mod_interval
    loop_ok = True
    for val in srtvalues[3:]:
      if (t + offsets[val]) % val != 0:
        loop_ok = False
        break
    if loop_ok:
      log.info(f"loop-OK for t#={t:,} @{int(time.time())-start_tm:,}s")
      return t
  raise Exception(f"No matching shuttle found after step t={t}")


# In[ ]:


test = "7,13,x,x,59,x,31,19"
assert( 1068781 == find_shuttle_offsetted(test) )


# In[ ]:


test = "17,x,13,19"
assert( 3417 == find_shuttle_offsetted(test) )


# In[ ]:


test = "67,7,59,61"
assert( 754018 == find_shuttle_offsetted(test) )


# In[ ]:


test = "67,x,7,59,61"
assert( 779210 == find_shuttle_offsetted(test) )


# In[ ]:


test = "67,7,x,59,61"
assert( 1261476 == find_shuttle_offsetted(test) )


# In[ ]:


test = "1789,37,47,1889"
assert( 1202161486 == find_shuttle_offsetted(test) )


# In[ ]:


print(f"known: solution larger than {100000000000000:,} <= 100000000000000")


# In[ ]:


def find_shuttle_offsetted6(s):
  """Semi-optimized brute-force algorithm implementation."""
  start_tm = int(time.time())
  log.info(f"[find_shuttle_offsetted] {s}")
  offsets = {}
  values = {}
  for idx, val in enumerate(s.split(',')):
    if val == 'x':
      continue
    val = int(val)
    values[idx] = val  # by offset
    offsets[val] = idx  # by value
  srtvalues = list(reversed(sorted(list(values.values()))))
  iterator1 = max(srtvalues)
  iterator1_offset = offsets[iterator1]
  log.info(f"max_it={iterator1}->ofst={iterator1_offset}; srtvalues={srtvalues}, offsets={offsets}, values={values}")
  
  #values_len = len(srtvalues)
  iterator2 = srtvalues[1]
  iterator2_offset = offsets[iterator2]
  iterator3 = srtvalues[2]
  iterator3_offset = offsets[iterator3]
  iterator4 = srtvalues[3]
  iterator4_offset = offsets[iterator4]
  iterator5 = srtvalues[4]
  iterator5_offset = offsets[iterator5]
  iterator6 = srtvalues[5]
  iterator6_offset = offsets[iterator6]
  print_mod_interval = 100_000_000_000
  next_print_mod = print_mod_interval
  for idx in range(1, 9_000_000_000_000_000//iterator1):
    t = idx * iterator1 - iterator1_offset

    if (t + iterator2_offset) % iterator2 != 0:
      continue # "FAST EXIT" this loop-item
    elif (t + iterator3_offset) % iterator3 != 0:
      continue # "FAST EXIT" this loop-item
    elif (t + iterator4_offset) % iterator4 != 0:
      continue # "FAST EXIT" this loop-item
    elif (t + iterator5_offset) % iterator5 != 0:
      continue # "FAST EXIT" this loop-item
    elif (t + iterator6_offset) % iterator6 != 0:
      continue # "FAST EXIT" this loop-item
    else:
      if t >= next_print_mod: #idx >= next_print_mod:
        log.info(f"  calculating @{int(time.time())-start_tm:,}s ...: t#={t:,}; {t//(int(time.time())-start_tm):,} Ts/s")
        next_print_mod += print_mod_interval
      loop_ok = True
      for val in srtvalues[6:]:
        if (t + offsets[val]) % val != 0:
          loop_ok = False
          break
      if loop_ok:
        log.info(f"loop-OK for t#={t:,} @{int(time.time())-start_tm:,}s")
        return t
  raise Exception(f"No matching shuttle found after step t={t}")


# In[ ]:


in13b = ins[1]
#EXEC_RESOURCE_HOGS = True
if EXEC_RESOURCE_HOGS:
  res = find_shuttle_offsetted6(in13b)
  print(f"Day 13 b solution={res}")
  # 2,448,348,017 Ts/s
  # 3,163,888,049 Ts/s explicit t calc
else:
  print("Omitting day 13 b resource expensive solution")


# In[ ]:


# Inspiration base: [- 2020 Day 13 Solutions - : adventofcode](https://www.reddit.com/r/adventofcode/comments/kc4njx/2020_day_13_solutions/)
# One solution: [adventofcode2020/main.py at master · r0f1/adventofcode2020](https://github.com/r0f1/adventofcode2020/blob/master/day13/main.py)
# Math Explanation: [Chinese Remainder Theorem | Brilliant Math & Science Wiki](https://brilliant.org/wiki/chinese-remainder-theorem/)
# a wonderful walk-through: [aoc/README.md at master · mebeim/aoc](https://github.com/mebeim/aoc/blob/master/2020/README.md#day-13---shuttle-search)

import numpy as np
#from math import prod # python 3.8 ?

def egcd(a, b):
  if a == 0:
    return (b, 0, 1)
  else:
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

#def egcd(a, b):
#  if a == 0:
#    return (b, 0, 1)
#  else:
#    g, y, x = egcd(b % a, a)
#    return (g, x - (b // a) * y, y)

#def modinv(a, m):
#  g, x, y = egcd(a, m)
#  if g != 1:
#    raise Exception('modular inverse does not exist')
#  else:
#    return x % m

def modinv(x, m):
  g, inv, y = egcd(x, m)
  assert g == 1, 'modular inverse does not exist'
  return inv % m

def pow38(g,w,p):
  #log.info(f"pow38({g},{w},{p}) called")
  if w >= 0:
    return pow(g, w ,p)
  else:
    return modinv(g, p) #, -w, p


with open('./in/day13.in') as f:
  lines = [x.strip() for x in f]

arrival = int(lines[0])
buses = [(i, int(e)) for i, e in enumerate(lines[1].split(",")) if e.isdigit()]

times = [t for _, t in buses]
b = [e - (arrival % e) for e in times]
res = np.min(b) * times[np.argmin(b)]
print("Day 13 a solution:", res)

# Python-3.7 ERROR: pow() 2nd argument cannot be negative when 3rd argument specified
def crt(ns, bs):
  """Solve: Chinese Remainder "problem" using Chinese Remainder Theorem."""
  # Chinese Remainder Theorem
  # https://brilliant.org/wiki/chinese-remainder-theorem/
  #N = prod(ns)
  N = np.prod(ns).item()
  #x = sum(b * (N // n) * pow(N // n, -1, n) for b, n in zip(bs, ns))
  x = sum(b * (N // n) * pow38(N // n, -1, n) for b, n in zip(bs, ns))
  return x % N

offsets = [time-idx for idx, time in buses]
res = crt(times, offsets)
print(f"Day 13 b solution: {res:,} <-- {res}")


# In[ ]:


# cool solution from user Rtchaik; this is my preferred!:
#  at: [- 2020 Day 13 Solutions - : adventofcode](https://www.reddit.com/r/adventofcode/comments/kc4njx/2020_day_13_solutions/)

from itertools import count

def solve_day13_part2(buses):
  log.info(f"[solve_day13_part2] {buses}")
  start_idx, steps = 0, 1
  log.info(f"  initial startid={start_idx}, steps-delta={steps}")
  for bus, offset in sorted(buses.items(), reverse=True):
    for tstamp in count(start_idx, steps):
      if not (tstamp + offset) % bus:
        start_idx = tstamp
        steps *= bus
        log.info(f"  new startid={start_idx}, steps-delta={steps}, tstamp={tstamp}")
        break
  log.info(f"found-OK: {tstamp}")
  return tstamp

def prepare_buses(s):
  buses = {}
  for idx, val in enumerate(s.split(',')):
    if val == 'x':
      continue
    val = int(val)
    buses[val] = idx
  return buses


# In[ ]:


test = "1789,37,47,1889"
assert( 1202161486 == solve_day13_part2(prepare_buses(test)) )


# In[ ]:


#ins = aoc.read_file_to_list('./in/day13.in')
res = solve_day13_part2(prepare_buses(ins[1]))
log.info(f"Day 13 b solution: {res:,} <-- {res}")


# ### Day 14: Docking Data

# In[ ]:


def solve_day14_a(los):
  log.info(f"[solve_day14_a] #-instructions={len(los)}")
  addrs = {}
  for line in los:
    if line.startswith('mask'):
      mask = line.split(' ')[-1]
      mask_or = mask.replace('0','X').replace('X','0')
      mask_and = mask.replace('1','X').replace('X','1')
      num_or = int(mask_or, 2)
      num_and = int(mask_and, 2)
      log.debug(f"mask={mask}")
      log.trace(f"  mask_or ={mask_or }; num_or ={num_or}")
      log.trace(f"  mask_and={mask_and}; num_and={num_and}")
    else:
      addr, val = mapl(int, filterl(lambda it: it != '', re.split(r'[^\d]', line)))
      new_val = (val | num_or) & num_and
      addrs[addr] = new_val
      log.debug(f"instruct={[addr, val]} new_val={new_val}")
  res = sum(addrs.values())
  log.info(f"[solve_day14_a] value-sum={res} from num-addrs={len(addrs.keys())} addrs[#1-#3]={list(addrs.items())[0:3]}")
  return res


# In[ ]:


tests = """
mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X
mem[8] = 11
mem[7] = 101
mem[8] = 0
""".strip().split("\n")

#log.setLevel(logging.DEBUG)
solve_day14_a(tests)


# In[ ]:


log.setLevel(logging.INFO)
ins = aoc.read_file_to_list('./in/day14.in')
solve_day14_a(ins)


# In[ ]:


print("Day 14 b")


# In[ ]:


import itertools
# this function by reddit User semicolonator
#  @ [adventofcode2020/main.py at master · r0f1/adventofcode2020](https://github.com/r0f1/adventofcode2020/blob/master/day14/main.py)
def get_possible_addrs(mask, addr):
  mask2 = "".join(v if m == "0" else m for m, v in zip(mask, f"{addr:036b}"))
  res = []
  for t in itertools.product("01", repeat=mask2.count("X")):
    it = iter(t)
    res.append(int("".join(next(it) if c == "X" else c for c in mask2), 2))
  return res


# In[ ]:


#def find_nth(haystack, needle, n):
#  parts= haystack.split(needle, n+1)
#  if len(parts)<=n+1:
#    return -1
#  return len(haystack)-len(parts[-1])-len(needle)

#def replace_at(instr, pos, new_char):
#  l = list(instr)
#  l[pos] = new_char
#  return str.join('', l)

#def get_variations(mf):
#  varias = []
#  mf2 = mf.replace('0','.')
#  varct = mf.count('X')
#  varias.append(mf2)
#  log.debug(f"[get_variations({mf})] ct={varct} repr={mf2}")
#  #for n in range(0, varct):
#  #  pos = find_nth(mf, 'X', n)
#  #  log.debug(f"#{n} at {pos}")
#  lct = 0
#  while('X' in varias[0]):
#    lct += 1
#    if lct > 99_999:
#      raise Exception(f"get_variations FAILSAFE on in={mf}")
#    log.trace(f"partial-varias={varias}")
#    next_varias = varias.copy()
#    pos = varias[0].find('X')
#    for mfit in varias:
#      next_varias.remove(mfit)
#      mf0 = replace_at(mfit, pos, '0')
#      mf1 = replace_at(mfit, pos, '1')
#      next_varias.append([mf0, mf1])
#    varias = aoc.flatten(next_varias)
#  log.trace(f"final varias={varias}")
#  return varias

def solve_day14_b(los):
  log.info(f"[solve_day14_b] #-instructions={len(los)}")
  addrs = {}
  for line in los:
    if line.startswith('mask'):
      mask = line.split(' ')[-1]
      mask_float = mask.replace('1','0')
      mask_or = mask.replace('X','0') #mask.replace('0','X').replace('X','0')
      num_or = int(mask_or, 2)
      log.debug(f"mask={mask}")
      log.trace(f"  mask_float={mask_float}")
      log.trace(f"  mask_or   ={mask_or }; num_or ={num_or}")
    else:
      new_addrs = {}
      addr, val = mapl(int, filterl(lambda it: it != '', re.split(r'[^\d]', line)))
      #new_val = (val | num_or) & num_and
      # NOP?: If the bitmask bit is 0, the corresponding memory address bit is unchanged.
      #  OR!: If the bitmask bit is 1, the corresponding memory address bit is overwritten with 1.
      new_addr = addr | num_or
      log.trace(f"        addr={    addr:>8b} ; := {    addr}")
      #log.trace(f"      num_or={num_or:>8b} ; := {num_or}")
      ##log.trace(f"    addr-ORd={new_addr:>8b}")
      log.trace(f"    new-addr={new_addr:>8b} ; := {new_addr}")
      #mf = re.search(r'X.*', mask_float)[0]
      #log.trace(f"  mask_float={mf:>8}")
      #for varia in get_variations(mask_float):
      #  #['0....0', '0....1', '1....0', '1....1']
      #  mask_and = varia.replace('.', '1')
      #  mask_or =  varia.replace('.', '0')
      #  num_and = int(mask_and, 2)
      #  num_or = int(mask_or, 2)
      #  idx = (new_addr | num_or) & num_and
      #  new_addrs[idx] = 1
      #log.debug(f"instruct={[addr, val]} new_addr={new_addr}, new_addrs-#={len(new_addrs.keys())}")
      #for addr2 in new_addrs.keys():
      #  addrs[addr2] = val
      for addr2 in get_possible_addrs(mask, addr):
        addrs[addr2] = val
  res = sum(addrs.values())
  log.info(f"[solve_day14_b] value-sum={res} from addrs-#={len(addrs.keys())} addrs[#1-#3]={list(addrs.items())[0:3]}")
  log.trace(f"  {addrs}")
  return res


# In[ ]:


tests = """
mask = 000000000000000000000000000000X1001X
mem[42] = 100
mask = 00000000000000000000000000000000X0XX
mem[26] = 1
""".strip().split("\n")

#log.setLevel(aoc.LOGLEVEL_TRACE) # logging.DEBUG
#log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)
solve_day14_b(tests)


# In[ ]:


solve_day14_b(ins)


# ### Day 15: Rambunctious Recitation

# In[ ]:


def solve15a(l, steps=10):
  log.debug(f"[solve15a(l)] called with l={l}")
  seen = {}
  last_spoken = None
  for idx, n in enumerate(l):
    last_spoken = n
    if n in seen:
      seen[n].append(idx+1)
    else:
      seen[n] = [idx+1]
    log.debug(f"idx#{idx+1}, n={n}, * seen[n]={seen[n]}")
  #log.trace(f"  seen={seen}")
  for idx in range(idx+2, steps+len(l)-idx):
    #log.debug(f"idx#{idx}, last_spoken={last_spoken}, seen-len={len(seen)}")
    #log.trace(f"  seen={seen}")
    if len(seen[last_spoken])==1:
      n = 0
    else:
      n = seen[last_spoken][-1] - seen[last_spoken][-2]
    if n in seen:
      seen[n].append(idx)
    else:
      seen[n] = [idx]
    log.trace(f"  new n={n}; seen={seen}")
    log.debug(f"idx#{idx}, n={n}, last_spoken={last_spoken}, seen-len={len(seen)}")
    last_spoken = n
  log.info(f"[solve15a] idx#{idx}, n={n}, last_spoken={last_spoken}, seen-len={len(seen)}")
  return n


# In[ ]:


tests = "0,3,6"
#log.setLevel(aoc.LOGLEVEL_TRACE)
#log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)
res = solve15a(mapl(int, tests.split(',')), steps=10)
# 0*, 3*, 6*, 0, 3, 3, 1, 0, 4, 0
log.info(f"testing result={0}")


# In[ ]:


res = solve15a([1, 3, 2], steps=2020)
assert( 1 == res )

res = solve15a([2, 1, 3], steps=2020)
assert( 10 == res )

res = solve15a([1, 2, 3], steps=2020)
assert( 27 == res )

res = solve15a([2, 3, 1], steps=2020)
assert( 78 == res )

res = solve15a([3, 2, 1], steps=2020)
assert( 438 == res )

res = solve15a([3, 1, 2], steps=2020)
assert( 1836 == res )


# In[ ]:


ins = aoc.read_file_to_str('./in/day15.in').strip().split(',')
ins = mapl(int, ins)
res = solve15a(ins, steps=2020)
#log.setLevel(logging.DEBUG)
log.info(f"Day 15 a solution: {res} from {ins}")


# In[ ]:


def solve15b(l, steps=10):
  log.info(f"[solve15b(l)] called with list-len={len(l)}, steps={steps:,}")
  seen = {}
  last_spoken = None
  for idx, n in enumerate(l):
    last_spoken = n
    if n in seen:
      #seen[n].append(idx+1)
      seen[n] = [seen[n][-1], idx+1]
    else:
      seen[n] = [idx+1]
    #log.debug(f"idx#{idx+1}, n={n}, * seen[n]={seen[n]}")
  seen_lens = {}
  for n in seen:
    seen_lens[n] = len(seen[n])  
  for idx in range(idx+2, steps+len(l)-idx):
    if idx % 10_000_000 == 0 and idx < steps:
      log.info(f"  calculating, @ idx={idx:,}")
    if seen_lens[last_spoken] == 1: #len(seen[last_spoken]) == 1:
      n = 0
    else:
      n = seen[last_spoken][-1] - seen[last_spoken][-2]
    if n in seen:
      #seen[n].append(idx)
      seen[n] = [seen[n][-1], idx]
      seen_lens[n] = 2
    else:
      seen[n] = [idx]
      seen_lens[n] = 1
    #log.debug(f"idx#{idx}, n={n}, last_spoken={last_spoken}, seen-len={len(seen)}")
    last_spoken = n
  log.info(f"[solve15b] idx#{idx:,}, n={n}, last_spoken={last_spoken}, seen-len={len(seen)}")
  return n


# In[ ]:


# Part a soltions still valid !
res = solve15b([1, 3, 2], steps=2020)
assert( 1 == res )
res = solve15b([2, 1, 3], steps=2020)
assert( 10 == res )
res = solve15b([1, 2, 3], steps=2020)
assert( 27 == res )
res = solve15b([2, 3, 1], steps=2020)
assert( 78 == res )
res = solve15b([3, 2, 1], steps=2020)
assert( 438 == res )
res = solve15b([3, 1, 2], steps=2020)
assert( 1836 == res )


# In[ ]:


#nsteps = 30000000
nsteps = 30_000_000

def run15b(l, steps, cond):
  if cond is not None and not EXEC_RESOURCE_HOGS:
    # omit resource intensive tests
    return
  start_tm = int(time.time())
  res = solve15b(l, steps=nsteps)
  if cond is not None:
    assert( cond == res )
  took_tm = int(time.time()) - start_tm
  log.info(f"result={res} took {took_tm}s")


# In[ ]:


# Given 0,3,6, the 30000000th number spoken is 175594.
run15b([0, 3, 6], nsteps, 175594)


# In[ ]:


# Given 1,3,2, the 30000000th number spoken is 2578.
run15b([1, 3, 2], nsteps, 2578)


# In[ ]:


# Given 2,1,3, the 30000000th number spoken is 3544142.
run15b([2, 1, 3], nsteps, 3544142)


# In[ ]:


# Given 1,2,3, the 30000000th number spoken is 261214.
run15b([1, 2, 3], nsteps, 261214)


# In[ ]:


# Given 2,3,1, the 30000000th number spoken is 6895259.
run15b([2, 3, 1], nsteps, 6895259)


# In[ ]:


# Given 3,2,1, the 30000000th number spoken is 18.
run15b([3, 2, 1], nsteps, 18)


# In[ ]:


# Given 3,1,2, the 30000000th number spoken is 362.
run15b([3, 1, 2], nsteps, 362)


# In[ ]:


if EXEC_RESOURCE_HOGS:
  log.info("Day 15 b solution:")
  run15b(ins, nsteps, None)
else:
  log.info("Day 15 b solution: [[already solved]] - omitting")  


# In[ ]:




