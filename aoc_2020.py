#!/usr/bin/env python
# coding: utf-8

# # Advent of Code 2020
# 
# This solution (Jupyter notebook; python 3.7) by kannix68, @ 2020-12.  \
# Using anaconda distro, conda v4.9.2. installation on MacOS v10.14.6 "Mojave".

# ## Generic AoC code

# In[ ]:


import sys
import logging
import itertools
#from operator import mul
import re

import numpy as np

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


### Day 16: Ticket Translation


# In[ ]:


tests = """
class: 1-3 or 5-7
row: 6-11 or 33-44
seat: 13-40 or 45-50

your ticket:
7,1,14

nearby tickets:
7,3,47
40,4,50
55,2,20
38,6,12
""".strip()


# In[ ]:


def parse_day16_input(s):
  los = s.split("\n")
  md = 'fields'
  myticket = []
  other_tickets = []
  fields = {}
  for line in los:
    if line == '':
      continue
    if line == 'your ticket:':
      md = 'my_ticket'
      continue
    elif line == 'nearby tickets:':
      md = 'other_tickets'
      continue
    if md == 'fields':
      fld, vals = line.split(':')
      avals = mapl(lambda it: it.strip() , vals.split(' or '))
      for idx, aval in enumerate(avals):
        aval = mapl(int, aval.split('-'))
        avals[idx] = aval
      fields[fld] = avals
    elif md == 'my_ticket' or md == 'other_tickets':
      this_ticket = mapl(int, line.split(','))
      if md == 'my_ticket':
        my_ticket = this_ticket
      else:
        other_tickets.append(this_ticket)
  return {'fields':fields, 'my_ticket':my_ticket, 'other_tickets':other_tickets}

def solve16a(ticket_info):
  #log.info(f"ticket_info={ticket_info}")
  valid_nums = []
  for field in ticket_info['fields'].keys():
    for entry in ticket_info['fields'][field]:
      min, max = entry
      for n in range(min, max+1):
        valid_nums.append(n)
  valid_nums = sorted(set(valid_nums))
  #log.info(f"valid_nums={valid_nums}")
  invalid_nums = []
  for this_ticket in ticket_info['other_tickets']:
    for n in this_ticket:
      if not n in valid_nums:
        invalid_nums.append(n)
  ticket_error_rate = sum(invalid_nums)
  log.info(f"ticket_error_rate={ticket_error_rate} invalid_nums={invalid_nums}")
  return ticket_error_rate


# In[ ]:


ticket_info = parse_day16_input(tests)
solve16a(ticket_info)


# In[ ]:


ins = aoc.read_file_to_str('./in/day16.in')
ticket_info = parse_day16_input(ins)
solve16a(ticket_info)


# In[ ]:


print("Day 16 b")

tests2 = """
class: 0-1 or 4-19
row: 0-5 or 8-19
seat: 0-13 or 16-19

your ticket:
11,12,13

nearby tickets:
3,9,18
15,1,5
5,14,9
""".strip()


# In[ ]:


def solve16b(ticket_info):
  #log.info(f"ticket_info={ticket_info}")
  fields = ticket_info['fields']
  my_ticket = ticket_info['my_ticket']
  other_tickets = ticket_info['other_tickets']
  all_tickets = other_tickets.copy()
  all_tickets.append(my_ticket)
  log.info(f"[solve16b] start all_tickets_len={len(all_tickets)}")
  all_valid_nums = []
  valid_nums = {}
  for field in fields.keys():
    valid_nums[field] = []
    for entry in fields[field]:
      min, max = entry
      for n in range(min, max+1):
        valid_nums[field].append(n)
        all_valid_nums.append(n)
  for field in valid_nums.keys():
    valid_nums[field] = sorted(set(valid_nums[field]))
  all_valid_nums = sorted(set(all_valid_nums))
  log.trace(f"valid_nums={valid_nums}")
  invalid_tickets = []
  for this_ticket in all_tickets:
    for n in this_ticket:
      if not n in all_valid_nums:
        invalid_tickets.append(this_ticket)
        break
  for this_ticket in invalid_tickets:
    log.debug(f"removing invalid ticket {this_ticket}")
    other_tickets.remove(this_ticket)
    all_tickets.remove(this_ticket)
  log.info(f"[solve16b] weedd all_tickets_len={len(all_tickets)}")
  
  num_fields = len(ticket_info['fields'])
  log.info(f"[solve16b] num_fields={num_fields}")
  assert( len(my_ticket) == num_fields)
  
  idx_maybe_field = {}
  for idx in range(num_fields):
    idx_maybe_field[idx] = []
    ticket_nums_at_idx = mapl(lambda it: it[idx], all_tickets)
    for field in fields:
      if set(ticket_nums_at_idx).issubset(set(valid_nums[field])):
        log.debug(f"idx={idx} field={field} OK for values={ticket_nums_at_idx}")
        idx_maybe_field[idx].append(field)

  idx_map = {}
  for i in range(1, 1001):
    lens = mapl(lambda it: len(it[1]), idx_maybe_field.items()) # index-order is implcit
    log.trace(lens)
    found_this_loop = []
    for idx, l in enumerate(lens):
      if l == 0:
        continue
      #if not idx in idx_maybe_field.keys(): # already found
      #  continue
      if l == 1:
        fieldnm = idx_maybe_field[idx][0]
        found_this_loop.append(fieldnm)
        idx_map[fieldnm] = idx
        idx_maybe_field[idx] = []
    log.debug(f"loop {i} idx_map={idx_map}")
    for f in found_this_loop:
      for k in idx_maybe_field.keys():
        if f in idx_maybe_field[k]:
          idx_maybe_field[k].remove(f)
    if len(idx_map.keys()) >= num_fields:
      break
    if i >= 1000:
      raise Exception("FAILSAFE")
  return idx_map


# In[ ]:


ticket_info = parse_day16_input(tests)
solve16b(ticket_info)


# In[ ]:


ticket_info = parse_day16_input(tests2)
solve16b(ticket_info)


# In[ ]:


ticket_info = parse_day16_input(ins)
idx_map = solve16b(ticket_info)
my_ticket = ticket_info['my_ticket']
f = 1
for k,v in idx_map.items():
  if k.startswith('departure'):
    log.info(f"field-idx={[k, v]} myticket-val={my_ticket[v]}")
    f *= my_ticket[v]
log.info(f"Day 16 b solution: {f}") # not 930240


# ### Day 17: Conway Cubes

# In[ ]:


tests = """
.#.
..#
###
""".strip()
tests = mapl(list, tests.split("\n"))
log.info(tests)


# In[ ]:


# solution TODO


# In[ ]:


class Grid3d:
  """Grid of 3d-cells, discrete 3d space, each cell represents a cube."""

  def __init__(self):
    log.debug("[Grid3d] constructor.")

  def initialize(self, pattern):
    self.pattern0 = pattern
    self.points = []
    z = 0
    for y in range(len(pattern)):
      for x in range(len(pattern[0])):
        if pattern[y][x] == '#':
          self.points.append( (x, y, z) )

  def report(self):
    return f"#pts={len(self.points)} {self.points}"

  def get_layer(self, z):
    return filterl(lambda it: z == it[2], self.points)

  def get_zrange(self):
    zs = mapl(lambda it: it[2], self.points)
    return range(min(zs), max(zs)-min(zs)+1)

  def get_layer_repr(self, z):
    xs = mapl(lambda it: it[0], self.points)
    ys = mapl(lambda it: it[1], self.points)
    extent2d = [[min(xs), max(xs)], [min(ys), max(ys)]]
    dim_x, dim_y = [ max(xs) - min(xs) + 1, max(ys) - min(ys) + 1 ]
    x_ofst = -min(xs)
    y_ofst = -min(ys)
    rows = []
    for y in range(0, max(ys)+y_ofst+1):
      s = ''
      for x in range(0, max(xs)+x_ofst+1):
        if (x-x_ofst, y-y_ofst, z) in self.points:
          s += '#'
        else:
          s += '.'
      rows.append(s)
    return f"grid-lvl@z={z} dims={[dim_x, dim_y]} extents={self.get_extents()} x-ofst={-x_ofst} y-ofst={-y_ofst}\n" +str.join("\n", rows)  

  def get_num_neighbors(self, pt):
    xp, yp, zp = pt
    num_neighbors = 0
    for z in range(zp-1, zp+2):
      for y in range(yp-1, yp+2):
        for x in range(xp-1, xp+2):
          if (x, y, z) == pt:  # identity, given point itself
            continue
          if (x, y, z) in self.points:
            num_neighbors += 1
    return num_neighbors

  def get_extents(self):
    xs = mapl(lambda it: it[0], self.points)
    ys = mapl(lambda it: it[1], self.points)
    zs = mapl(lambda it: it[2], self.points)
    return [[min(xs), max(xs)], [min(ys), max(ys)], [min(zs), max(zs)]]


class ConwayCubeGrid(Grid3d):
  """Conway cellular automaton in 3d, inheriting from class Grid3d."""

  def __init__(self):
    log.debug("[ConwayCubeGrid] constructor.")
    self.t = 0
  
  def iterate(self, steps=1):
    for i in range(steps):
      exts = self.get_extents()
      new_pts = copy.deepcopy(self.points)
      for x in range(exts[0][0]-1, exts[0][1]+2):
        for y in range(exts[1][0]-1, exts[1][1]+2):
          #if x == 0:
          #  log.trace(f"iter-row {y}")
          for z in range(exts[2][0]-1, exts[2][1]+2):
            pt = (x, y, z)
            is_active = pt in self.points
            #if is_active:
            #  log.info(f"iterate: pt={pt} was active")
            nn = self.get_num_neighbors(pt)
            if is_active:
              if not (nn in [2, 3]):
                #log.trace(f"iter-remove {pt}")
                new_pts.remove( pt )
            else:
              if nn == 3:
                #log.trace(f"iter-append {pt}")
                new_pts.append( pt )
      self.points = new_pts
      self.t += 1


# In[ ]:


grid = Grid3d()
log.info(f"grid={grid}")
grid.initialize(tests)
log.info(f"grid rpt:\n{grid.report()}")
assert 1 == grid.get_num_neighbors( (0,0,0) )
assert 2 == grid.get_num_neighbors( (2,0,0) )
assert 5 == grid.get_num_neighbors( (1,1,0) )
assert 0 == grid.get_num_neighbors( (-2,-1,0) )

grid.get_extents()


# In[ ]:


grid = ConwayCubeGrid()
log.info(f"grid={grid}")
grid.initialize(tests)
#log.info(f"grid rpt:\n{grid.report()}")
assert 1 == grid.get_num_neighbors( (0,0,0) )
assert 2 == grid.get_num_neighbors( (2,0,0) )
assert 5 == grid.get_num_neighbors( (1,1,0) )
assert 0 == grid.get_num_neighbors( (-2,-1,0) )

grid.get_extents()
log.info(f"grid @ t={grid.t} extents={grid.get_extents()} numpts={len(grid.points)}")
log.info(grid.get_layer_repr(0))
#res = grid.get_layer(0)

for i in range(1, 7):
  grid.iterate()
  log.info(f"Iterated: grid @ t={grid.t} extents={grid.get_extents()} numpts={len(grid.points)}")
  for z in grid.get_zrange():
    ##log.info(f"grid @ t={grid.t} pts@z=0 {res}")
    #log.info(grid.get_layer_repr(z))
    True


# In[ ]:


grid = ConwayCubeGrid()
grid.initialize(tests)
grid.iterate(steps=6)
assert( 6 == grid.t )
assert( 112 == len(grid.points) )


# In[ ]:


ins = aoc.read_file_to_str('in/day17.in').strip()
log.info(f"pattern=\n{ins}")
ins = mapl(list, ins.split("\n"))
grid = ConwayCubeGrid()
grid.initialize(ins)
grid.iterate(steps=6)
assert( 6 == grid.t )
res = len(grid.points)
log.info(f"Day 18 a solution: num points after 6 iterations: {res}")


# In[ ]:


class Grid4d:
  """Grid of 4d-cells, each cell represents a 4d-cube, a hypercube, a tesseract."""

  def __init__(self):
    log.debug("[Grid4d] constructor.")

  def initialize(self, pattern):
    self.pattern0 = pattern
    self.points = []
    z, w = 0, 0
    for y in range(len(pattern)):
      for x in range(len(pattern[0])):
        if pattern[y][x] == '#':
          self.points.append( (x, y, z, w) )

  def report(self):
    return f"#pts={len(self.points)} {self.points}"

  def get_layer(self, z, w):
    return filterl(lambda it: z == it[2] and w == it[3], self.points)

  def get_zrange(self):
    zs = mapl(lambda it: it[2], self.points)
    return range(min(zs), max(zs)+1)

  def get_wrange(self):
    ws = mapl(lambda it: it[3], self.points)
    return range(min(ws), max(ws)+1)

  def get_layer_repr(self, z, w):
    xs = mapl(lambda it: it[0], self.points)
    ys = mapl(lambda it: it[1], self.points)
    extent2d = [[min(xs), max(xs)], [min(ys), max(ys)]]
    dim_x, dim_y = [ max(xs) - min(xs) + 1, max(ys) - min(ys) + 1 ]
    x_ofst = -min(xs)
    y_ofst = -min(ys)
    rows = []
    for y in range(0, max(ys)+y_ofst+1):
      s = ''
      for x in range(0, max(xs)+x_ofst+1):
        if (x-x_ofst, y-y_ofst, z, w) in self.points:
          s += '#'
        else:
          s += '.'
      rows.append(s)
    return f"grid-lvl@[z,w]={[z,w]} dims={[dim_x, dim_y]} extents={self.get_extents()}"       + f"x-ofst={-x_ofst} y-ofst={-y_ofst}\n" +str.join("\n", rows)  

  def get_num_neighbors(self, pt):
    xp, yp, zp, wp = pt
    num_neighbors = 0
    for w in range(wp-1, wp+2):
      for z in range(zp-1, zp+2):
        for y in range(yp-1, yp+2):
          for x in range(xp-1, xp+2):
            if (x, y, z, w) == pt:  # identity, given point itself
              continue
            if (x, y, z, w) in self.points:
              num_neighbors += 1
    return num_neighbors

  def get_extents(self):
    xs = mapl(lambda it: it[0], self.points)
    ys = mapl(lambda it: it[1], self.points)
    zs = mapl(lambda it: it[2], self.points)
    ws = mapl(lambda it: it[3], self.points)
    return [[min(xs), max(xs)], [min(ys), max(ys)], [min(zs), max(zs)], [min(ws), max(ws)]]


class ConwayTesseractGrid(Grid4d):
  """Conway cellular automaton in 4d, inheriting from class Grid4d."""

  def __init__(self):
    log.debug("[ConwayTesseractGrid] constructor.")
    self.t = 0
  
  def iterate(self, steps=1):
    for i in range(steps):
      exts = self.get_extents()
      new_pts = copy.deepcopy(self.points)
      for x in range(exts[0][0]-1, exts[0][1]+2):
        for y in range(exts[1][0]-1, exts[1][1]+2):
          #if x == 0:
          #  log.trace(f"iter-row {y}")
          for w in range(exts[3][0]-1, exts[3][1]+2):
            for z in range(exts[2][0]-1, exts[2][1]+2):
              pt = (x, y, z, w)
              is_active = pt in self.points
              #if is_active:
              #  log.info(f"iterate: pt={pt} was active")
              nn = self.get_num_neighbors(pt)
              if is_active:
                if not (nn in [2, 3]):
                  #log.trace(f"iter-remove {pt}")
                  new_pts.remove( pt )
              else:
                if nn == 3:
                  #log.trace(f"iter-append {pt}")
                  new_pts.append( pt )
      self.points = new_pts
      self.t += 1


# In[ ]:


grid = ConwayTesseractGrid()
log.info(f"grid={grid}")
grid.initialize(tests)
#log.info(f"grid rpt:\n{grid.report()}")
assert 1 == grid.get_num_neighbors( (0,0,0,0) )
assert 2 == grid.get_num_neighbors( (2,0,0,0) )
assert 5 == grid.get_num_neighbors( (1,1,0,0) )
assert 0 == grid.get_num_neighbors( (-2,-1,0,0) )

grid.get_extents()
log.info(f"grid @ t={grid.t} extents={grid.get_extents()} numpts={len(grid.points)}")
log.info(grid.get_layer_repr(0, 0))
#res = grid.get_layer(0)

grid.iterate()
log.info(grid.get_layer_repr(-1, -1))
log.info(grid.get_layer_repr(0, 0))
log.info(grid.get_layer_repr(1, 1))

grid.iterate()
log.info(grid.get_layer_repr(-2, -2))
log.info(grid.get_layer_repr(0, 0))
log.info(grid.get_layer_repr(2, 0))

grid.iterate(steps=4)
assert( 6 == grid.t )
assert( 848 == len(grid.points) )


# In[ ]:


if EXEC_RESOURCE_HOGS: # took 226 seconds on my notebook
  grid = ConwayTesseractGrid()
  grid.initialize(ins)
  start_tm = int(time.time())
  for i in range(1, 7):
    grid.iterate(steps=1)
    npts = len(grid.points)
    took_tm = int(time.time()) - start_tm
    log.info(f"after grid iteration {i}: num-points={npts:,} after {took_tm}s")
  assert( 6 == grid.t )
  res = len(grid.points)
  log.info(f"Day 18 b solution: num points after 6 iterations: {res}")


# ### Day 18: : Operation Order

# In[ ]:


# This definitely is/would be LISP territory !


# In[ ]:


def parse_equation18a(s):
  """Parse / tokenize a single "equation"."""
  l = re.split(r'(?=[\+\-\*\/\(\)])|(?<=[\+\-\*\/\(\)])', s)
  l = filterl(lambda it: it != '', mapl(lambda it: it.strip(), l))
  l = mapl(lambda it: int(it) if not (it in ['+','-','*','/','(',')']) else it, l)
  log.debug(f"[parse_equation18a] returns={l}")
  return l

def rindex_list(elem, l):
  """Return the index of the rightmost element in list."""
  return len(l) - list(reversed(l)).index(elem) - 1

def find_matching_close_paren_idx(lst):
  """Assumes input list starting with '(', finds matching ')' and returns it's index.
    If not found, returns -1."""
  tgtcount = 0
  tgtidx = -1
  for idx in range(len(lst)):
    if lst[idx] == ')':
      tgtcount -= 1
    elif lst[idx] == '(':
      tgtcount += 1
    if tgtcount < 1:
      tgtidx = idx
      break
  return tgtidx
  
def calc18a(l):
  log.debug(f"[calc18a] l={l}")
  rest = l
  ict = 0
  while( len(rest)>1 ):
    ict += 1
    lval, rest = [rest[0], rest[1:]]
    log.trace(f"  in [lval, rest]={[lval, rest]} rest-len={len(rest)}")
    if lval == '(':
      rest = [lval] + rest # re-assemble
      ridx = find_matching_close_paren_idx(rest)
      sublst = rest[1:ridx] # last/rightmost index of closing parens
      new_rest = rest[ridx+1:]
      log.trace(f"calcparen lval={lval} sublst={sublst} new-rest={new_rest} from={rest}")
      lval = calc18a(sublst.copy())
      rest = [lval] + new_rest
    else:
      op, rest = [rest[0], rest[1:]]
      rval = rest[0]
      log.trace(f"  op-mode {[op, rest]} lval={lval} op={op} rval={rval} all-rest={rest}")
      if rval == '(':
        idx = find_matching_close_paren_idx(rest)
        sublst = rest[1:idx]
        new_rest = rest[idx+1:]
        log.trace(f"calcparen (lval={lval}) rval sublst={sublst} new-rest={new_rest} from {rest}")
        rval = calc18a(sublst.copy())
        rest = [op] + new_rest
        log.trace(f"  calcparen rval={rval} sublst={sublst} new-rest={new_rest} from {rest}")
      if op == '+':
        lval += rval
        rest = [lval] + rest[1:]
      elif op == '*':
        lval *= rval
        rest = [lval] + rest[1:]
      else:
        raise Exception(f"unhandled operator {op}")
    log.trace(f"  loop-end: lval={lval}; new-list={rest}")
    if len(rest)>1 and rest[1] == ')': # found result of parns in val
      log.debug("  next is ')' group closing, break")
      break
  log.debug(f"  returning val={lval}; from={l}")
  return lval


# In[ ]:


#log.setLevel(aoc.LOGLEVEL_TRACE)
#log.setLevel(logging.INFO)

test = """
1 + 2 * 3 + 4 * 5 + 6
""".strip()
testlst = parse_equation18a(test)
res = calc18a(testlst)
print("test result", res)


# In[ ]:


test = """
1 + (2 * 3) + (4 * (5 + 6))
""".strip()
assert( 51 == calc18a(parse_equation18a(test)))


# In[ ]:


test = """
2 * 3 + (4 * 5)
""".strip()
res = calc18a(parse_equation18a(test))
assert( 26 == calc18a(parse_equation18a(test)))


# In[ ]:


test = """
5 + (8 * 3 + 9 + 3 * 4 * 3)
""".strip()
expectd = 437
assert( expectd == calc18a(parse_equation18a(test)))


# In[ ]:


test = """
5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))
""".strip()
expectd = 12240
assert( expectd == calc18a(parse_equation18a(test)))


# In[ ]:


test = """
(1 + 2)
""".strip()
expectd = 3
assert( expectd == calc18a(parse_equation18a(test)))


# In[ ]:


test = """
((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2
""".strip()
expectd = 13632
assert( expectd == calc18a(parse_equation18a(test)))


# In[ ]:


ins = aoc.read_file_to_list('./in/day18.in')
csum = 0
for eqstr in ins:
  csum += calc18a(parse_equation18a(eqstr))
log.info(f"Day 18 a solution: equations cumsum={csum}")


# In[ ]:


print("Day 18 b")
def calc18b(l):
  log.debug(f"[calc18b] l={l}")
  rest = l
  ict = 0
  while( len(rest)>1 ):
    ict += 1
    lval, rest = [rest[0], rest[1:]]
    log.trace(f"  >in [lval, rest]={[lval, rest]} rest-len={len(rest)}")
    if lval == '(':
      rest = [lval] + rest # re-assemble
      ridx = find_matching_close_paren_idx(rest)
      sublst = rest[1:ridx] # last/rightmost index of closing parens
      new_rest = rest[ridx+1:]
      log.trace(f"calcparen lval={lval} sublst={sublst} new-rest={new_rest} from={rest}")
      lval = calc18b(sublst.copy())
      rest = [lval] + new_rest
      log.trace(f"  cprv new-rest={rest}")
    else:
      op, rest = [rest[0], rest[1:]]
      rval = rest[0]
      log.trace(f"  op-mode {[op, rest]} lval={lval} op={op} rval={rval} all-rest={rest}")
      if rval == '(':
        idx = find_matching_close_paren_idx(rest)
        sublst = rest[1:idx]
        new_rest = rest[idx+1:]
        log.trace(f"calcparen (lval={lval}) rval sublst={sublst} new-rest={new_rest} from {rest}")
        rval = calc18b(sublst.copy())
        rest = [rval] + new_rest
        log.trace(f"  calcparen rval={rval} sublst={sublst} new-rest={new_rest} from {rest}")
      if op == '+':
        lval += rval
        rest = [lval] + rest[1:]
        log.debug(f"    (+)=> rval={rval}, lval={lval}, new rest={rest}")
      elif op == '*':
        # postpone multiplication ! Rather, recurse fun-call for r-value
        log.debug(f"  PROD in [lval, op, rest]={[lval, op, rest]} rest-len={len(rest)}")
        if len(rest) > 1:
          rval = calc18b(rest.copy())          
        lval *= rval
        rest = []
        log.debug(f"    (*)=> rval={rval}, lval={lval}, new rest={rest}")
      else:
        raise Exception(f"unhandled operator {op}")
    log.trace(f"  loop-end: lval={lval}; new-list={rest}")
    if len(rest)>1 and rest[1] == ')': # found result of parens in val
      log.debug("  next is ')' group closing, break")
      break
  log.debug(f"[calc18b] RC={lval} from {l}")
  return lval


# In[ ]:


test = """
1 + 2 * 3 + 4 * 5 + 6
""".strip()
testlst = parse_equation18a(test)
res = calc18b(testlst)
print("test result", res)


# In[ ]:


test = """
1 + (2 * 3) + (4 * (5 + 6))
""".strip()
expectd = 51
res = calc18b(parse_equation18a(test))
assert( expectd == res)
log.info(f"test result={res}")


# In[ ]:


test = """
2 * 3 + (4 * 5)
""".strip()
expectd = 46
res = calc18b(parse_equation18a(test))
assert( expectd == res)
log.info(f"test result={res}")


# In[ ]:


test = """
5 + (8 * 3 + 9 + 3 * 4 * 3)
""".strip()
expectd = 1445
res = calc18b(parse_equation18a(test))
assert( expectd == res)
log.info(f"test result={res}")


# In[ ]:


test = """
5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))
""".strip()
expectd = 669060
res = calc18b(parse_equation18a(test))
assert( expectd == res)
log.info(f"test result={res}")


# In[ ]:


test = """
((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2
""".strip()
expectd = 23340
res = calc18b(parse_equation18a(test))
assert( expectd == res)
log.info(f"test result={res}")


# In[ ]:


ins = aoc.read_file_to_list('./in/day18.in')
csum = 0
for eqstr in ins:
  csum += calc18b(parse_equation18a(eqstr))
log.info(f"Day 18 b solution: equations cumsum={csum}")


# ### Day 19: Monster Messages
# 
# The most simple/elegant would be to create a grammar for this problem and parse the rules (lexx/yacc) etc.
# But as a shortcut today I did a fallback on only using/constructing text regular expressions.

# In[ ]:


def parse_day19_rules(s):
  rules = s.split("\n")
  rules = mapl(lambda it: it.split(': '), rules)
  return rules

def parse_day19(s):
  rules, samples = s.strip().split("\n\n")
  rules = parse_day19_rules(rules)
  samples = samples.split("\n")
  log.debug(f"parsed:\n  rules=\n{rules}\n  samples=\n{samples}")
  return rules, samples

def solve_day19(rules, max_depth=30, part=1):
  log.debug(f"[solve19b] started")
  pd = {}
  rules_keys = []
  for rule in rules:
    rule_num, rule_expr = rule
    rules_keys.append(rule_num)
    if rule_expr.startswith('"') and rule_expr.endswith(''):
      log.debug(f"  added key={rule_num} rule={rule_expr}")
      pd[rule_num] = rule_expr.replace('"', '')
  missing_rules_keys = rules_keys.copy()
  for k in pd.keys():
    missing_rules_keys.remove(k)
    
  for i in range(1, max_depth+2):
    log.debug(f"  loop#={i}")
    found_new_key = False
    for rule in rules:
      rule_num, rule_expr = rule

      if part == 2: # apply part 2 conditions:
        if rule_num == '8':
          rule_expr = '42 | 42 8'
        elif rule_num == '11':
          rule_expr = '42 31 | 42 11 31'

      if not rule_num in pd.keys():
        ree = rule_expr.split(' ')
        rules_contained = set(filterl(lambda it: it != '|', ree))
        log.trace(f"unparsed rule {rule}, rules_contained={rules_contained}")
        if set(rules_contained).issubset(pd.keys()):
          log.trace(f"can add {ree}")
          r = str.join('', mapl(lambda it: pd[it] if it in pd.keys() else it, ree))
          pd[rule_num] = '(' + r + ')'
          found_new_key = True
          missing_rules_keys.remove(rule_num)
          log.debug(f"  added key={rule_num} rule={r}")
        else:
          log.trace(f"can't add {ree}")
    if not found_new_key:
      if not '0' in pd.keys():
        log.debug(f"rule0 not found after {i} loops, rules-found={sorted(pd.keys())}")
        log.debug(f"  rules_missing={sorted(missing_rules_keys)}")
        if part == 2:
          log.debug(f"  rules[42]={pd['42']}")
          log.debug(f"  rules[31]={pd['31']}")
          # THIS is the re secret sauce expressing ca. conditions:
          # > rule_expr = '42 | 42 8' :: 1..n of pattern 42
          pd['8'] = f"({pd['42']})+"
          # > rule_expr = '42 31 | 42 11 31' :: 1..n of pattern 42 followd by 31
          #pd['11'] = f"({pd['42']})+({pd['31']})+" # the first and second + repeat count have to be same
          ors = []
          for i in range(1, 6):
            pl = pd['42']
            pr = pd['31']
            ors.append(pl*i+pr*i)
          pd['11'] = f"({str.join('|', ors)})"
          # > 8 11
          pd['0'] = f"{pd['8']}{pd['11']}"
          log.debug(f"  rules[8]={pd['8']}")
          log.debug(f"  len(rules[11])={len(pd['11'])}")
          log.debug(f"  len(rules[0])={len(pd['0'])}")
      break
  log.debug(f"[solve19b] parsed-dict={pd}")
  return pd


# In[ ]:


tests = """
0: 1 2
1: "a"
2: 1 3 | 3 1
3: "b"
""".strip()


# In[ ]:


log.setLevel(logging.INFO)
rules = parse_day19_rules(tests)
log.info(f"test: parsed rules\n{rules}")
pd = solve_day19(rules)
rule0 = pd['0']
assert( re.match(rule0, "aab") )
assert( re.match(rule0, "aba") )
assert( not re.match(rule0, "bba") )


# In[ ]:


tests = """
0: 4 1 5
1: 2 3 | 3 2
2: 4 4 | 5 5
3: 4 5 | 5 4
4: "a"
5: "b"
""".strip()

rules = parse_day19_rules(tests)
log.info(f"test: parsed rules\n{rules}")
pd = solve_day19(rules)
log.info(f"tests parse-dir={pd}")
rule0='^' + pd['0'] + '$'

samples = "aaaabb,aaabab,abbabb,abbbab,aabaab,aabbbb,abaaab,ababbb".split(',')
for sample in samples:
  assert( re.match(rule0, sample) )
assert( not re.match(rule0, "baaabb") )
assert( not re.match(rule0, "ababba") )


# In[ ]:


tests = """
0: 4 1 5
1: 2 3 | 3 2
2: 4 4 | 5 5
3: 4 5 | 5 4
4: "a"
5: "b"

ababbb
bababa
abbbab
aaabbb
aaaabbb
""".strip()


# In[ ]:


#ababbb and abbbab match
#bababa, aaabbb, and aaaabbb

rules, samples = parse_day19(tests)
pd = solve_day19(rules)
log.info(f"rule0={pd['0']}")
rule0='^' + pd['0'] + '$'
smatching = 0
for sample in samples:
  if re.match(rule0, sample):
    #log.info(f"{sample} matches {rule0}")
    smatching +=1
  else:
    #log.info(f"{sample} NOmatch {rule0}")
    True
log.info(f"matching-samples-#={smatching}")
assert ( smatching == 2)


# In[ ]:


ins = aoc.read_file_to_str('./in/day19.in')
rules, samples = parse_day19(ins)
pd = solve_day19(rules)
log.debug(f"rule0={pd['0']}")
rule0='^' + pd['0'] + '$'
log.info(f"parsed-rules, len(rule0)={len(rule0)}")
smatching = 0
for sample in samples:
  if re.match(rule0, sample):
    smatching +=1
    #log.info(f"{sample} matches {rule0}")
  #else:
  #  log.info(f"{sample} NOmatch {rule0}")
    
log.info(f"matching-samples-#={smatching}")


# In[ ]:


print("Day 19 b")

tests = """
42: 9 14 | 10 1
9: 14 27 | 1 26
10: 23 14 | 28 1
1: "a"
11: 42 31
5: 1 14 | 15 1
19: 14 1 | 14 14
12: 24 14 | 19 1
16: 15 1 | 14 14
31: 14 17 | 1 13
6: 14 14 | 1 14
2: 1 24 | 14 4
0: 8 11
13: 14 3 | 1 12
15: 1 | 14
17: 14 2 | 1 7
23: 25 1 | 22 14
28: 16 1
4: 1 1
20: 14 14 | 1 15
3: 5 14 | 16 1
27: 1 6 | 14 18
14: "b"
21: 14 1 | 1 14
25: 1 1 | 1 14
22: 14 14
8: 42
26: 14 22 | 1 20
18: 15 15
7: 14 5 | 1 21
24: 14 1

abbbbbabbbaaaababbaabbbbabababbbabbbbbbabaaaa
bbabbbbaabaabba
babbbbaabbbbbabbbbbbaabaaabaaa
aaabbbbbbaaaabaababaabababbabaaabbababababaaa
bbbbbbbaaaabbbbaaabbabaaa
bbbababbbbaaaaaaaabbababaaababaabab
ababaaaaaabaaab
ababaaaaabbbaba
baabbaaaabbaaaababbaababb
abbbbabbbbaaaababbbbbbaaaababb
aaaaabbaabaaaaababaa
aaaabbaaaabbaaa
aaaabbaabbaaaaaaabbbabbbaaabbaabaaa
babaaabbbaaabaababbaabababaaab
aabbbbbaabbbaaaaaabbbbbababaaaaabbaaabba
""".strip()

log.setLevel(logging.INFO)
rules, samples = parse_day19(tests)
max_samples_len = max(mapl(len, samples))
log.debug(f"max_samples_len={max_samples_len}")
pd = solve_day19(rules, part=2, max_depth=max_samples_len)
log.debug(f"rule0={pd['0']}")
rule0='^' + pd['0'] + '$'
log.info(f"parsed-rules, len(rule0)={len(rule0)}")

smatching = 0
for sample in samples:
  if re.match(rule0, sample):
    smatching +=1
log.info(f"matching-samples-#={smatching}")
assert( 12 == smatching )


# In[ ]:


log.setLevel(logging.INFO)
rules, samples = parse_day19(ins)
max_samples_len = max(mapl(len, samples))
log.debug(f"max_samples_len={max_samples_len}")
pd = solve_day19(rules, part=2, max_depth=max_samples_len)
log.debug(f"rule0={pd['0']}")
rule0='^' + pd['0'] + '$'
log.info(f"parsed-rules, len(rule0)={len(rule0)}")

smatching = 0
for sample in samples:
  if re.match(rule0, sample):
    smatching +=1
log.info(f"matching-samples-#={smatching}")


# ### Day 20: Jurassic Jigsaw

# In[ ]:


tests = """
Tile 2311:
..##.#..#.
##..#.....
#...##..#.
####.#...#
##.##.###.
##...#.###
.#.#.#..##
..#....#..
###...#.#.
..###..###

Tile 1951:
#.##...##.
#.####...#
.....#..##
#...######
.##.#....#
.###.#####
###.##.##.
.###....#.
..#.#..#.#
#...##.#..

Tile 1171:
####...##.
#..##.#..#
##.#..#.#.
.###.####.
..###.####
.##....##.
.#...####.
#.##.####.
####..#...
.....##...

Tile 1427:
###.##.#..
.#..#.##..
.#.##.#..#
#.#.#.##.#
....#...##
...##..##.
...#.#####
.#.####.#.
..#..###.#
..##.#..#.

Tile 1489:
##.#.#....
..##...#..
.##..##...
..#...#...
#####...#.
#..#.#.#.#
...#.#.#..
##.#...##.
..##.##.##
###.##.#..

Tile 2473:
#....####.
#..#.##...
#.##..#...
######.#.#
.#...#.#.#
.#########
.###.#..#.
########.#
##...##.#.
..###.#.#.

Tile 2971:
..#.#....#
#...###...
#.#.###...
##.##..#..
.#####..##
.#..####.#
#..#.#..#.
..####.###
..#.#.###.
...#.#.#.#

Tile 2729:
...#.#.#.#
####.#....
..#.#.....
....#..#.#
.##..##.#.
.#.####...
####.#.#..
##.####...
##..#.##..
#.##...##.

Tile 3079:
#.#.#####.
.#..######
..#.......
######....
####.#..#.
.#...#.##.
#.#####.##
..#.###...
..#.......
..#.###...
""".strip()


# In[ ]:


def get_dimens(num_tiles):
  for gridx in range(1, num_tiles+1):
    for gridy in range(1, num_tiles+1):
      if gridx * gridy == num_tiles:
        if gridx > 1 and gridy > 1:
          log.info(f"[get_dimens] {gridx}x{gridy} dimen possible.")

def get_borders(tile):
  borders = set()
  rows = tile.split("\n")
  borders.add(rows[0])
  borders.add(rows[0][::-1])  # reversed
  borders.add(rows[-1])
  borders.add(rows[-1][::-1])  # reversed
  col0 = str.join('', mapl(lambda it: it[0], rows))
  col_last = str.join('', mapl(lambda it: it[-1], rows) )
  borders.add(col0)
  borders.add(col0[::-1])  # reversed
  borders.add(col_last)
  borders.add(col_last[::-1])  # reversed
  return borders

def find_corner_tiles(tiles):
  tile_keys = tiles.keys()
  borders = {}
  bsects = {}
  for key in tile_keys:
    borders[key] = get_borders(tiles[key])
    bsects[key] = []
  for combi in itertools.permutations(tile_keys, 2):
    key1, key2 = combi
    b1 = borders[key1]
    b2 = borders[key2]
    bsects[key1].append( len( b1 & b2 ) )
  corner_tiles = set()
  for key in tile_keys:
    #log.info(f"key: {key} {bsects[key]}")
    bct = len( filterl(lambda it: it > 0,  bsects[key]) )
    if bct < 3:
      #log.info(f"border-tile: {key}")
      corner_tiles.add(key)
    #elif bct == 4:
    #  log.info(f"middle-tile: {key}")
  return corner_tiles

def find_border_tiles(tiles):
  tile_keys = tiles.keys()
  borders = {}
  bsects = {}
  for key in tile_keys:
    borders[key] = get_borders(tiles[key])
    bsects[key] = []
  for combi in itertools.permutations(tile_keys, 2):
    key1, key2 = combi
    b1 = borders[key1]
    b2 = borders[key2]
    bsects[key1].append( len( b1 & b2 ) )
  border_tiles = set()
  for key in tile_keys:
    bct = len( filterl(lambda it: it > 0,  bsects[key]) )
    if bct == 3:
      border_tiles.add(key)
  return border_tiles

def parse_tiles(s):
  d = {}
  for tile_str in s.split("\n\n"):
    tile_repr = ''
    for idx, line in enumerate(tile_str.split("\n")):
      if idx == 0:
        tile_id = int( line.replace('Tile ','').replace(':','') )
      else:
        tile_repr += line + "\n"
    d[tile_id] = tile_repr.strip()
  return d


# In[ ]:


tiles = parse_tiles(tests)
num_tiles = len(tiles.keys())
tile_keys = tiles.keys()
log.info(f"tests num-tiles={num_tiles}")
get_dimens(num_tiles)
find_corner_tiles(tiles)


# In[ ]:


ins = aoc.read_file_to_str('in/day20.in').strip()
tiles = parse_tiles(ins)
num_tiles = len(tiles.keys())
log.info(f"input num-tiles={num_tiles}")
res = find_corner_tiles(tiles)
log.info(f"ins corner-tiles={res}")
res = np.prod(list(res))
log.info(f"Day 20 a solution: border-tiles-product={res}")


# In[ ]:


print("Day 20 b")


# In[ ]:


from math import sqrt

def flip_vert_tile(s):
  """Flip a tile vertically, return str repr."""
  return str.join("\n", list(reversed(s.split("\n"))))

def flip_horiz_tile(s):
  """Flip a tile horizontally, return str repr."""
  new_los = []
  for line in s.split("\n"):
    new_los.append(str.join('', reversed(line)))
  return str.join("\n", new_los)
  
def rotate_tile(s):
  """Left-rotate of tile representation, return str repr."""
  lol = mapl(lambda it: list(it), s.split("\n"))
  new_los = []
  for islice in reversed(range(len(lol))):
    line = str.join('', mapl(lambda it: it[islice], lol))
    new_los.append(line)
  log.trace("rot-repr=\n"+str.join("\n", new_los))
  return str.join("\n", new_los)

def get_tile_transforms(s):
  """Provide all transforms of a tile as list, including identity."""
  transforms = [s]  # start with identity as first elem
  current_repr = s
  for rot_num in range(3):
    current_repr = rotate_tile(current_repr)
    transforms.append(current_repr)

  current_repr = flip_vert_tile(s)
  transforms.append(current_repr)
  for rot_num in range(3):
    current_repr = rotate_tile(current_repr)
    transforms.append(current_repr)

  current_repr = flip_horiz_tile(s)
  transforms.append(current_repr)
  for rot_num in range(3):
    current_repr = rotate_tile(current_repr)
    transforms.append(current_repr)

  return set(transforms)

def fits_horiz(lefts, rights):
  lhs = str.join('', mapl(lambda it: it[-1], lefts.split("\n")))
  rhs = str.join('', mapl(lambda it: it[0], rights.split("\n")))
  return lhs == rhs

def fits_vert(tops, bottoms):
  lhs = tops.split("\n")[-1]
  rhs =  bottoms.split("\n")[0]
  return lhs == rhs

def get_next_coord(coord, image_width):
  x, y = coord
  nx = (x+1) % image_width
  if x == image_width-1:
    ny = y+1
  else:
    ny = y
  log.trace(f"next-coord={(nx, ny)}")
  return (nx, ny)

def is_corner(coord, image_width):
  b = (coord[0] in [0, image_width-1]) and (coord[1] in [0, image_width-1])
  if b:
    log.trace(f"{coord} is corner; image_width={image_width}")
  return b

def is_border(coord, image_width):
  log.trace(f"{coord} is border image_width={image_width}")
  b = not is_corner(coord, image_width) and     ((coord[0] in [0, image_width-1]) or (coord[1] in [0, image_width-1]))
  if b:
    log.info(f"{coord} is border; image_width={image_width}")
  return b

def create_image(tiles, tilekeys_left, img, imgidx, coord, corner_tiles, border_tiles, image_width):
  x, y = coord
  log.debug(f"[create_image] tks-left={len(tilekeys_left)}, @{coord}")
  if x >= image_width or y >= image_width:
    log.debug(f"FOUND\n{np.array(imgidx)}")
    return True, img, imgidx
  if y > 0 and x > 0:
    #log.info(f" check h+v")
    #if is_corner(coord, image_width):  # @ corner
    #  tkl2 = tilekeys_left & corner_tiles
    #elif is_border(coord, image_width):  # @border
    #  tkl2 = tilekeys_left & border_tiles
    #else:
    #  tkl2 = tilekeys_left
    #for tk in tkl2:
    for tk in tilekeys_left:
      for tvari in get_tile_transforms( tiles[tk] ):
        if fits_horiz(img[y][x-1], tvari) and fits_vert(img[y-1][x], tvari):
          tkl_new = tilekeys_left.copy(); tkl_new.remove(tk)
          img_new = copy.deepcopy(img); img_new[y][x] = tvari
          log.debug(f"found h+v match for tilekey={tk} @{coord}")
          imgidx_new = copy.deepcopy(imgidx); imgidx_new[y][x] = tk
          return create_image(tiles, tkl_new, img_new, imgidx_new, get_next_coord(coord, image_width), corner_tiles, border_tiles, image_width)
  elif y > 0:
    #log.info(f" check   v")
    #if is_corner(coord, image_width):  # @ corner
    #  tkl2 = tilekeys_left & corner_tiles
    #else:  # @border
    #  tkl2 = tilekeys_left & border_tiles
    #for tk in tkl2:
    for tk in tilekeys_left:
      for tvari in get_tile_transforms( tiles[tk] ):
        if fits_vert(img[y-1][x], tvari):
          tkl_new = tilekeys_left.copy(); tkl_new.remove(tk)
          img_new = copy.deepcopy(img); img_new[y][x] = tvari
          imgidx_new = copy.deepcopy(imgidx); imgidx_new[y][x] = tk
          log.debug(f"found h+v match for tilekey={tk} @{coord}")
          return create_image(tiles, tkl_new, img_new, imgidx_new, get_next_coord(coord, image_width), corner_tiles, border_tiles, image_width)
  elif x > 0:
    #log.info(f" check h")
    #if is_corner(coord, image_width):  # @ corner
    #  tkl2 = tilekeys_left & corner_tiles
    #else:  # @border
    #  tkl2 = tilekeys_left & border_tiles
    #for tk in tkl2:
    for tk in tilekeys_left:
      for tvari in get_tile_transforms( tiles[tk] ):
        if fits_horiz(img[y][x-1], tvari):
          tkl_new = tilekeys_left.copy(); tkl_new.remove(tk)
          img_new = copy.deepcopy(img); img_new[y][x] = tvari
          imgidx_new = copy.deepcopy(imgidx); imgidx_new[y][x] = tk
          log.debug(f"found h+v match for tilekey={tk} @{coord}")
          return create_image(tiles, tkl_new, img_new, imgidx_new, get_next_coord(coord, image_width), corner_tiles, border_tiles, image_width)
  log.trace("[create_image] fall-out")
  return False, img, imgidx

def assemble_image(tiles):
  tiles_keys = tiles.keys()
  num_tiles = len(tiles)
  image_width = int(sqrt(num_tiles))
  corner_tiles = find_corner_tiles(tiles)
  log.info(f"[assemble_image] corner-tiles-#={len(corner_tiles)}")
  assert( 4 == len(corner_tiles) )
  border_tiles = find_border_tiles(tiles)
  log.info(f"[assemble_image] border-tiles-#={len(border_tiles)}; image_width={image_width}")
  assert( 4*(image_width-2) == len(border_tiles) )
  start_tile = list(corner_tiles)[0]
  log.info(f"[assemble_image] starting; tiles_set={set(tiles_keys)}")
  tilekeys_left = set(tiles_keys) - set([start_tile])
  for vari in get_tile_transforms( tiles[start_tile] ):
    img = [[None for x in range(image_width)] for y in range(image_width)]
    imgidx = [[None for x in range(image_width)] for y in range(image_width)]
    img[0][0] = vari
    imgidx[0][0] = start_tile
    log.debug(f"first corner tile img=\n{vari}")
    img_found, img_final, imgidx_final = create_image(tiles, tilekeys_left, img, imgidx, get_next_coord((0,0), image_width), corner_tiles, border_tiles, image_width)
    if img_found:
      log.info(f"IMG found, idxs=\n{imgidx_final}")
      break
  assert( img_found )
  return img_found, img_final, imgidx_final

def get_image_repr(img):
  img_len = len(img)
  tile_len = len(img[0][0].split("\n"))
  log.debug(f"[get_image_repr] num-tiles={img_len}^2={img_len**2} cells-per-tile={tile_len**2}")
  images = copy.deepcopy(img)
  for img_y in range(img_len):
    for img_x in range(img_len):
      images[img_y][img_x] = img[img_y][img_x].split("\n")  # split each tile line-wise
  img_rows = []
  for img_rowidx in range(img_len):
    tiles_rows = []
    for tile_rowidx in range(tile_len):
      tiles_row = ""
      for img_colidx in range(img_len):
        tiles_row += images[img_rowidx][img_colidx][tile_rowidx]
      tiles_rows.append(tiles_row)
    img_rows.append(str.join("\n", tiles_rows))
  img_repr = str.join("\n", img_rows)
  return img_repr

def show_image(img):
  img_len = len(img)
  tile_len = len(img[0][0].split("\n"))
  log.info(f"[show_image] num-tiles={img_len}^2={img_len**2} cells-per-tile={tile_len**2}")
  log.info("\n"+get_image_repr(img))

def cut_tile_borders(tile):
  los = tile.split("\n")
  tile_len = len(los)
  new_los = []
  for idx, line in enumerate(los):
    if idx in [0, tile_len-1]:
      continue
    new_line = line[1:-1]
    assert(len(new_line) == tile_len-2)
    new_los.append( new_line )
  assert(len(new_los) == tile_len-2)
  return str.join("\n", new_los)

def cut_image_borders(img):
  img_len = len(img)
  for y in range(img_len):
    for x in range(img_len):
      tile = img[y][x]
      tile = cut_tile_borders(tile)
      img[y][x] = tile
  return img


# In[ ]:


sea_monster = """
                  # 
#    ##    ##    ###
 #  #  #  #  #  #   
"""

def tiles_to_sea_npar(sea_los):
  """Convert original tiles representation to a 'sea' numpy-array of 0s and 1s."""
  tiles = parse_tiles(sea_los)
  img_found, img, imgidx = assemble_image(tiles)
  #show_image(test_img)
  img_cut = cut_image_borders(img)
  #show_image(test_img_cut)
  img_cut = get_image_repr(img_cut)  # from x*x matrix to 1 str
  image_los = img_cut.replace(".", "0 ").replace("#", "1 ").split("\n")
  image_ar = np.array([[int(c) for c in seamst_line.strip().split(" ")] for seamst_line in image_los])
  return image_ar

# Thanks github user JesperDramsch:
def variations_of(npar):
  """Return identity and all rotation and flip-horiz flip-vert variations of np-array."""
  varias = []
  for i in range(4):
    tfar = np.rot90(npar, i)
    varias.append(tfar)
    varias.append(np.flip(tfar, 0))
    varias.append(np.flip(tfar, 1))
  return varias

# Inspired
# Thanks github user JesperDramsch, via reddit aoc 2020 day 20 solutions/discussion:
#   https://github.com/JesperDramsch/advent-of-code-1
def eliminate_monsters(sea, seamst):
  """Given 'sea' and 'seamonster' input numpy-arrays,
  eliminate all variations of seamonster (rots, flips) from the sea,
  return sea without monsters (np-array)."""

  seamst_cct = seamst.sum()
  seamst_varias = variations_of(seamst)

  monsters_num = 0
  while monsters_num == 0:
    monster = seamst_varias.pop()
    mst_y, mst_x = monster.shape
    for y, x in np.ndindex(sea.shape):
      sub_arr = sea[y : y + mst_y, x : x + mst_x].copy()
      if not sub_arr.shape == monster.shape:
        continue
      sub_arr *= monster  # <= sea & monster
      if np.sum(sub_arr) == seamst_cct:
        monsters_num += 1
        sea[y : y + mst_y, x : x + mst_x] -= monster  # => sea - monster
  return sea

sea_monster = sea_monster.strip("\n")
#print(f">{sea_monster}<")
# Thanks github user JesperDramsch:
sea_monster_los = sea_monster.replace(" ", "0 ").replace("#", "1 ").split("\n")
#log.info(f"\n{sea_monster_los}")
seamst = np.array([[int(c) for c in seamst_line.strip().split(" ")] for seamst_line in sea_monster_los])
seamst_cct = seamst.sum()
log.info(f"Seamonster cell-count={seamst_cct}")
log.info(f"\n{seamst}")

sea_ar = tiles_to_sea_npar(tests)
log.info(f"sea-nparray, shape={sea_ar.shape}::\n{sea_ar}")
res = eliminate_monsters(sea_ar, seamst).sum()
log.info(f"Day 21 b tests: rough-sea-count={res}")
assert( 273 == res )


# In[ ]:


sea_ar = tiles_to_sea_npar(ins)
log.info(f"sea-nparray, shape={sea_ar.shape}::\n{sea_ar}")
res = eliminate_monsters(sea_ar, seamst).sum()
log.info(f"Day 21 b final solution: rough-sea-count={res}")


# ### Day 21: Allergen Assessment

# In[ ]:


tests = """
mxmxvkd kfcds sqjhc nhms (contains dairy, fish)
trh fvjkl sbzzf mxmxvkd (contains dairy)
sqjhc fvjkl (contains soy)
sqjhc mxmxvkd sbzzf (contains fish)
""".strip().split("\n")


# In[ ]:


def solve_day21(los, part=1):
  ingreds_all = set()
  log.info(f"[solve21a] num-lines={len(los)}")
  allerg_assoc = {}
  recips = []
  for line in los:
    ingreds, allergs = line.split(' (contains ')
    ingreds = set(ingreds.strip().split(' '))
    allergs = allergs.strip().replace(')','').split(', ')
    log.debug(f"  ingreds={ingreds}; allergs={allergs}")
    ingreds_all |= ingreds
    recips.append({'ingreds':ingreds, 'allergs':allergs})
    for allerg in allergs:
      if not allerg in allerg_assoc:
        allerg_assoc[allerg] = set(ingreds)
      else:
        allerg_assoc[allerg] &= set(ingreds)
  for i in range(len(allerg_assoc.keys())):  # loop and weed max n times
    found_allergs = filterl(lambda it: len(allerg_assoc[it]) == 1, allerg_assoc.keys())
    found_ingreds = mapl(lambda it: list(allerg_assoc[it])[0], found_allergs)
    for allerg in allerg_assoc.keys():
      if allerg in found_allergs:
        continue
      allerg_assoc[allerg] -= set(found_ingreds)
    if 1 == max( mapl(lambda it: len(allerg_assoc[it]), allerg_assoc.keys()) ):
      break
  allerg_assoc = {k:list(v)[0] for k,v in allerg_assoc.items()} # get rid of wrapping set per values
  log.info(f"allerg_assoc={allerg_assoc}")
  ingreds_pure = ingreds_all.copy()
  for ingred_allergic in allerg_assoc.values():
    ingred_allergic = ingred_allergic
    ingreds_pure.remove(ingred_allergic)
  log.info(f"ingreds-pure={ingreds_pure}")
  ct = 0
  for ingred_pure in ingreds_pure:
    for recip in recips:
      if ingred_pure in recip['ingreds']:
        ct += 1
  log.info(f"day 21 part 1: count of pure ingredients occurences={ct}")
  if part == 1:
    return ct
  vals_ordered = []
  for k in sorted(allerg_assoc.keys()):
    vals_ordered.append(allerg_assoc[k])
  vals_str = str.join(',', vals_ordered)
  log.info(f"vals_str=>{vals_str}<")
  return vals_str


# In[ ]:


#log.setLevel(aoc.LOGLEVEL_TRACE)
log.setLevel(logging.INFO)
res = solve_day21(tests, part=1)
assert( 5 == res )


# In[ ]:


ins = aoc.read_file_to_list('./in/day21.in')
res = solve_day21(ins, part=1)
logging.info(f"Day 21 a solution: {res}")


# In[ ]:


print("Day 21 b")
#log.setLevel(aoc.LOGLEVEL_TRACE)
#log.setLevel(logging.INFO)
res = solve_day21(tests, part=2)
assert( "mxmxvkd,sqjhc,fvjkl" == res )


# In[ ]:


res = solve_day21(ins, part=2)
log.info(f"Day 21 b solution:\n>{res}<")


# ### Day 22: Crab Combat

# In[ ]:


def parse_day22(s):
  players = {}
  players_str = s.split("\n\n")
  for player_str in players_str:
    for line in player_str.split("\n"):
      if line.startswith('Player'):
        player_id = int(line.replace('Player ', '').replace(':',''))
        players[player_id] = []
      else:
        players[player_id].append(int(line))
  log.debug(f"[parse_day22] {players}")
  return players

def play_crabcardgame(players):
  t = 0
  player_keys = list(players.keys())
  while(
    min( mapl(lambda it: len(players[it]), player_keys) ) > 0
  ):
    draw = mapl(lambda it: players[it].pop(0), player_keys)
    winner_idx = draw.index(max(draw))
    #players[player_keys[winner_idx]] += sorted(draw, reverse=True)
    loser_idx = (0 if winner_idx == 1 else 1)
    players[player_keys[winner_idx]] += [draw[winner_idx], draw[loser_idx]] # winner's card first
    t += 1
    log.debug(f"[play_ccg] t={t} draw={draw} {players}")
    if t > 1_000:
      raise Exception("failsafe")
  players['t'] = t
  players['winner'] = player_keys[winner_idx]
  return players

def score_crabcardgame(players):
  cardstack = players[players['winner']]
  log.debug(f"[score_crabcardgame] cardstack={cardstack}")
  cardstack = list(reversed(cardstack))
  score = 0
  for idx in range(len(cardstack)):
    score += (idx+1) * cardstack[idx]
  return score


# In[ ]:


tests = """
Player 1:
9
2
6
3
1

Player 2:
5
8
4
7
10
""".strip()


# In[ ]:


players = parse_day22(tests)
players = play_crabcardgame(players)
res = score_crabcardgame(players)
assert( 306 == res)


# In[ ]:


ins = aoc.read_file_to_str('in/day22.in').strip()
players = parse_day22(ins)
players = play_crabcardgame(players)
res = score_crabcardgame(players)
log.info(f"Day 22 part 1 solution: winning score={res}")


# In[ ]:


print("Day 22 b")

def hashrep_of(player):
  repres = str(player)
  return hashlib.sha1(repres.encode()).hexdigest()

def play_recursivecombat(players):#
  t = 0
  player_keys = list(players.keys())
  player_seen_handhashes = set()
  plcardnums = [len(players[1]), len(players[2])]
  log.debug(f"[play_recursivecombat] plcard#={plcardnums} t={t} {players}")
  for t in range(1, 100_000):
    log.debug(f"t={t} init={players}")

    # NOTE: The hands-already-seen condition had to be read VERY CAREFULLY !!!
    player1_hashrep = hashrep_of(players[1])
    player2_hashrep = hashrep_of(players[2])
    if player1_hashrep in player_seen_handhashes and player2_hashrep in player_seen_handhashes:
      ###                             NOTE THE **AND** in above condition !!!
      log.debug(f"  current hands already seen")
      hand_seen = True
    else:
      player_seen_handhashes.add(player1_hashrep)
      player_seen_handhashes.add(player2_hashrep)
      hand_seen = False

    if hand_seen:
      players['t'] = t
      players['winner'] = player_keys[0]
      players['win-cond'] = 'hand_already_seen'
      log.debug(f"win-cond plcard#={plcardnums} already-played players={players}")
      return players
      
    draw = mapl(lambda it: players[it].pop(0), player_keys)
    log.debug(f"  t={t} draw={draw} keeping {players}")
    if draw[0] <= len(players[1]) and draw[1] <= len(players[2]):
      # both players have enough cards left
      log.debug(f"  recursing")
      recursed_players = copy.deepcopy(players)
      # the quantity of cards copied is equal to the number on the card they drew to trigger the sub-game
      if draw[0] < len(players[1]):
        recursed_players[1] = recursed_players[1][:draw[0]] # cut the stack to size for recursion
      if draw[1] < len(players[2]):
        recursed_players[2] = recursed_players[2][:draw[1]] # cut the stack to size for recursion
      recursed_players = play_recursivecombat(recursed_players)
      winner = recursed_players['winner']
    else:
      winner = draw.index(max(draw)) + 1
    winner_idx = winner - 1
    loser_idx = (0 if winner_idx == 1 else 1)
    players[winner] += [draw[winner_idx], draw[loser_idx]] # winner's card first
    if min( mapl(lambda it: len(players[it]), player_keys) ) <= 0:
      players['t'] = t
      players['winner'] = winner
      players['win-cond'] = '1player_out_of_cards'
      log.debug(f"win-cond plcard#={plcardnums} 1-player-run-outof-cards players={players}")
      return players
  raise Exception("failsafe")
  


# In[ ]:


players = play_recursivecombat(parse_day22(tests))
res = score_crabcardgame(players)
assert( 291 == res )


# In[ ]:


tests_loop = """
Player 1:
43
19

Player 2:
2
29
14
""".strip()
res = play_recursivecombat(parse_day22(tests_loop))
assert( res['win-cond'] == 'hand_already_seen' )


# In[ ]:


#log.setLevel(logging.INFO)
players = play_recursivecombat(parse_day22(ins))
log.info(f"recursive-combat result for ins: {players}")
res = score_crabcardgame(players)
log.info(f"Day 22 part 2 solution: recursive-combat winner-score={res}")


# ### Day 23: Crab Cups

# In[ ]:


def play_crabcups_round(l):
  #orig_lst = l.copy()
  list_len = len(l)
  current = l[0]
  taken = [l.pop(1), l.pop(1), l.pop(1)] # take 3
  next_val = current - 1
  while(True):
    if next_val in l:
      next_idx = l.index(next_val)
      break
    else:
      next_val -= 1
      if next_val <= 0:
        next_val = max(l)
  log.debug(f"[play_crabcups_round] head={current}, taken={taken}, dest={next_val}")
  new_list = [next_val]
  new_list = new_list + taken
  appending = False
  for val in itertools.cycle(l):
    if not appending:
      if val == next_val:
        appending = True
    else:
      new_list.append(val)
      if len(new_list) >= list_len:
        break
  log.debug(f"  new_list={new_list}")
  tgt_idx = (new_list.index(current)+1) % list_len
  new_list2 = new_list[tgt_idx:] + new_list[:tgt_idx]
  log.debug(f"  new_list2={new_list2}")
  return new_list2

def play_crabcups_game(l, rounds=1):
  log.info(f"[play_crabcups_game] started: l={l}, rounds={rounds}")
  lst = l.copy()
  for i in range(1, rounds+1):
    lst = play_crabcups_round(lst)
    log.debug(f" round={i} l={lst}")
  return lst

def score_crabcups_game(l):
  tgt_idx = (l.index(1)+1) % len(l)
  if tgt_idx == 0:
      outlst = l[tgt_idx, len(l)-1]
  else:
    outlst = l[tgt_idx:] + l[:tgt_idx-1]
  return int( str.join('', mapl(str,outlst)) )


# In[ ]:


tests = "389125467"
test_lst = mapl(int, list(tests))
res = play_crabcups_game(test_lst, rounds=10)
log.info(f"test result={res}")
score = score_crabcups_game(res)
log.info(f"test result  10rds score={score}")
assert( 92658374 == score )

res = play_crabcups_game(test_lst, rounds=100)
score = score_crabcups_game(res)
log.info(f"test result 100rds score={score}")
assert( 67384529 == score)


# In[ ]:


ins = aoc.read_file_to_str('in/day23.in').strip()
ins_lst = mapl(int, list(ins))
res = play_crabcups_game(ins_lst, rounds=100)
log.info(f"Day 23 part 1 result={res}")
score = score_crabcups_game(res)
log.info(f"Day 23 part 1 solution: result 100rds score={score}")


# In[ ]:


print("Day 23 b")

def assemble_crabcups2_list(l, num_cups = 1_000_000):
  """Get a cups-list according to part 2 requirements (1mio cups)."""
  out_lst = l.copy()
  max_val = max(l)
  num_new_cups = num_cups - len(out_lst)
  out_lst += list(range(max_val+1, num_cups+1))
  assert( num_cups == len(out_lst) )
  return out_lst

def play_crabcups_round_opt(l, rounds=1):
  """Optimize play of crabcups for n rounds, using cycling LinkedList instead of list."""
  start_tm = int(time.time())
  list_len = len(l)
  lkl = {}
  #firstval = l[0]
  #lastval = l[-1]
  curval = l[0]
  for idx, val in enumerate(l):
    next_idx = idx+1
    if next_idx == list_len:
      next_idx = 0
    lkl[val] = l[next_idx]
  for rd in range(rounds):
    # The crab picks up the three cups that are immediately clockwise of the current cup.
    # They are removed from the circle;
    # cup spacing is adjusted as necessary to maintain the circle.
    n1 = lkl[curval]
    n2 = lkl[n1]
    n3 = lkl[n2]
    lkl[curval] = lkl[n3]
    #log.trace(f"  re-chained from current={curval} to={lkl[n3]}, taken={[n1, n2, n3]}")
    
    # The crab selects a destination cup:
    # the cup with a label equal to the current cup's label minus one.
    # If this would select one of the cups that was just picked up,
    # the crab will keep subtracting one until it finds a cup
    # that wasn't just picked up.
    # If at any point in this process the value goes below
    # the lowest value on any cup's label, it wraps around
    # to the highest value on any cup's label instead.
    for _ in range(list_len):
      if _ == 0:
        nextval = curval
      nextval -= 1
      #log.trace(f"    chknextval={nextval}")
      if nextval in [n1, n2, n3]:
        #log.trace(f"      is in outtakes")
        continue
      if nextval <= 0:
        nextval = max(lkl.keys())+1
        continue
      else:
        break
    #log.trace(f"  current={curval} picked={[n1, n2, n3]}, dest={nextval}")
    # The crab places the cups it just picked up
    # so that they are immediately clockwise of the destination cup.
    # They keep the same order as when they were picked up.
    next_end_val = lkl[nextval] # store end value
    lkl[nextval] = n1 # break open the chain
    # lkl[n1] == n2
    # lkl[n2] == n3
    lkl[n3] = next_end_val # close the chain again

    # The crab selects a new current cup:
    #  the cup which is immediately clockwise of the current cup
    curval = lkl[curval]
    if rd % 1_000_000 == 0:
      took_tm = int(time.time()) - start_tm
      log.info(f"round={rd:,} time_taken sofar {took_tm}s")

  out_lst = []
  for i in range(list_len):
    if i == 0:
      #last_val = 1
      last_val = curval
    out_lst.append(last_val)
    last_val = lkl[last_val]
  
  return out_lst

def play_crabcups_game_opt(l, rounds=1):
  log.info(f"[play_crabcups_game] started: l={l}, rounds={rounds}")
  #lst = l.copy()
  return play_crabcups_round_opt(l, rounds)

def score_crabcups_game_part2(l):
  lst_len = len(l)
  tgt_idx = (l.index(1)+1) % len(l)
  if tgt_idx < lst_len - 2:
    subl = l[tgt_idx : tgt_idx+2]
    #log.info(subl)
  else:
    tgtidx1 = (tgt_idx+1) % lst_len
    tgtidx2 = (tgt_idx+2) % lst_len
    subl = [l[tgtidx1], l[tgtidx2]]
  assert( 2 == len(subl) )
  return subl[0] * subl[1]


# In[ ]:


# check part 1 game results and scores still valid...
tests = "389125467"
test_lst = mapl(int, list(tests))
res = play_crabcups_game_opt(test_lst, rounds=10)
log.info(f"test result={res}")
score1 = score_crabcups_game(res)
log.info(f"test result  10rds score part 1={score1}")
log.info(f"test result  10rds score part 2={score}")
assert( 92658374 == score1 )
score = score_crabcups_game_part2(res)

# still valid...
ins = aoc.read_file_to_str('in/day23.in').strip()
ins_lst = mapl(int, list(ins))
res = play_crabcups_game_opt(ins_lst, rounds=100)
log.info(f"Day 23 part 1 result={res}")
score1 = score_crabcups_game(res)
log.info(f"Day 23 part 1 solution: result 100rds score={score1}")
score = score_crabcups_game_part2(res)
log.info(f"Day 23 part 2 check: result 100rds score2={score}")
assert( 74698532 == score1 )


# In[ ]:


# test with long list for part 2
test2_lst = assemble_crabcups2_list(test_lst, num_cups = 1_000_000)
log.info("done")
assert( 1_000_000 == len(test2_lst) )

res = play_crabcups_game_opt(test2_lst, rounds=10_000_000)
log.info("done2")
score2 = score_crabcups_game_part2(res)
log.info(f"score2={score2}")

assert( 1_000_000 == len(res) )
assert( 149245887792 == score2 )


# In[ ]:


ins2_lst = assemble_crabcups2_list(ins_lst, num_cups = 1_000_000)
res = play_crabcups_game_opt(ins2_lst, rounds=10_000_000)
log.info("done2")
score2 = score_crabcups_game_part2(res)
log.info(f"Day 23 part 2 solution: score2={score2}")


# ### Day 24: Lobby Layout
# 
# Hexagonal geometry and hexagonal 2d-coordinates.
# 
# See red blob games site [Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/)
# for thorough explanations.
# Thanks to colleague P S for the hint!  \
# Last used in Advent of Code 2017, day 11.  \
# Todays aoc hint: [Hexagonal tiling - Wikipedia](https://en.wikipedia.org/wiki/Hexagonal_tiling)
# 

# In[ ]:


def cl(l):
  """Return compact list str representation."""
  return str(l).replace(', ',',')

#  Using pointy topped grid/geometry and axial coordinates.
#  Using axis notation [q,r] here, q is west>east and r is south>north
hex2d_axial_pt_translations = {'e':[1,0], 'w':[-1,0], 'se':[0,1], 'sw':[-1,1], 'ne':[+1,-1], 'nw':[0,-1]}

def hex_axial_distance(a, b):
  return int((abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) / 2)

# east, southeast, southwest, west, northwest, and northeast
# => e, se, sw, w, nw, and ne
def parse_day24_line(s):
  log.debug(f"parse_day24_line in={s}")
  out_trs = []
  while len(s) > 0:
    log.trace(f"out_trs={out_trs} rest={s}")
    if len(s)>= 2 and s[:2] in ['se','sw','nw','ne']:
      out_trs.append(s[:2])
      s = s[2:]
    elif len(s)>= 1 and s[:1] in ['e','w']:
      out_trs.append(s[:1])
      s = s[1:]
    else:
      raise Exception(f"unforeseen: {s}")
  log.debug(f"parse_day24_line returns {cl(out_trs)}")
  return out_trs
  
def parse_day24(los):
  return mapl(lambda it: parse_day24_line(it), los)

def flip_day24_line(steps):
  #flips = defaultdict(int)
  c = (0,0)
  for step in steps:
    trans = hex2d_axial_pt_translations[step]
    c = (c[0]+trans[0], c[1]+trans[1])
  #flips[c] += 1
  #return flips
  return c

def flip_day24_lines(steps_lol):
  flips = defaultdict(int)
  c = (0,0)
  for steps in steps_lol:
    c = flip_day24_line(steps)
    flips[c] += 1
  return flips


# In[ ]:


test1 = 'esew'
flip_day24_line( parse_day24_line(test1) )


# In[ ]:


test2 = 'nwwswee'
flip_day24_line( parse_day24_line(test2) ) 


# In[ ]:


tests = """
sesenwnenenewseeswwswswwnenewsewsw
neeenesenwnwwswnenewnwwsewnenwseswesw
seswneswswsenwwnwse
nwnwneseeswswnenewneswwnewseswneseene
swweswneswnenwsewnwneneseenw
eesenwseswswnenwswnwnwsewwnwsene
sewnenenenesenwsewnenwwwse
wenwwweseeeweswwwnwwe
wsweesenenewnwwnwsenewsenwwsesesenwne
neeswseenwwswnwswswnw
nenwswwsewswnenenewsenwsenwnesesenew
enewnwewneswsewnwswenweswnenwsenwsw
sweneswneswneneenwnewenewwneswswnese
swwesenesewenwneswnwwneseswwne
enesenwswwswneneswsenwnewswseenwsese
wnwnesenesenenwwnenwsewesewsesesew
nenewswnwewswnenesenwnesewesw
eneswnwswnwsenenwnwnwwseeswneewsenese
neswnwewnwnwseenwseesewsenwsweewe
wseweeenwnesenwwwswnew
""".strip().split("\n")
flips = flip_day24_lines( parse_day24(tests) )
tiles_black = filterl(lambda it: flips[it] % 2 == 1, flips.keys())
log.info(f"Day 24 part 1 tests solutions: black tiles#={len(tiles_black)}") #" from {tiles_black}")
assert( 10 == len(tiles_black))


# In[ ]:


ins = aoc.read_file_to_list('in/day24.in')
flips = flip_day24_lines( parse_day24(ins) )
tiles_black = filterl(lambda it: flips[it] % 2 == 1, flips.keys())
log.info(f"Day 24 part 1 solution: black tiles#={len(tiles_black)}") #" from {tiles_black}")


# In[ ]:


print("Day 24 b")

# cellular automaton on this hexagonal tile geometry space

def get_extents(tiles_black):
  qs = mapl(lambda it: it[0], tiles_black)
  rs = mapl(lambda it: it[1], tiles_black)
  return [[min(qs), max(qs)], [min(rs), max(rs)]]
  

def num_neighbors(c, tiles_black):
  nsum = 0
  for tilec in tiles_black:
    #if c != tilec and hex_axial_distance(c, tilec) == 1:
    if hex_axial_distance(c, tilec) == 1:
      log.trace(f"{tilec} is neib of {c}")
      nsum += 1
  assert( nsum <= 6 )
  return nsum

def cell_automate(tiles_black, rounds = 1):
  exts = get_extents(tiles_black)
  log.info(f"[cell_automate] at round 0: num-tiles-black={len(tiles_black)}; extents={exts}") #" from {sorted(tiles_black)}")
  start_tm = int(time.time())
  for rnd in range(1, rounds+1):
    new_tiles_black = tiles_black.copy()
    exts = get_extents(tiles_black)
    log.debug(f"round {rnd}: extents found={exts}")
    q_min, q_max = exts[0]
    r_min, r_max = exts[1]

    for q in range(q_min-1, q_max+1+1):
      for r in range(r_min-1, r_max+1+1):
        c = (q, r)
        nneibs = num_neighbors(c, tiles_black)
        if c in tiles_black:
          if nneibs == 0 or nneibs > 2:
            log.debug(f"flip-to-white {c} nneibs={nneibs}")
            new_tiles_black.remove(c)
        else:
          if nneibs == 2:
            log.debug(f"flip-to-black {c} nneibs={nneibs}")
            new_tiles_black.append(c)
    tiles_black = new_tiles_black
    took_tm = int(time.time()) - start_tm
    log.info(f"  after round {rnd} @{took_tm:>5}s: num-tiles-black={len(tiles_black)}; extents={exts}") #" from {sorted(tiles_black)}")
  log.info(f"[cell_automate] finished round {rnd}: num-tiles-black={len(tiles_black)}; extents={exts}") #" from {sorted(tiles_black)}")
  return tiles_black

flips = flip_day24_lines( parse_day24(tests) )
tiles_black = filterl(lambda it: flips[it] % 2 == 1, flips.keys())
assert 10 == len(tiles_black)

tiles_black2 = cell_automate(tiles_black, rounds=1)
assert 15 == len(tiles_black2)

tiles_black2 = cell_automate(tiles_black, rounds=2)
assert 12 == len(tiles_black2)

tiles_black2 = cell_automate(tiles_black, rounds=10)
assert 37 == len(tiles_black2)

tiles_black2 = cell_automate(tiles_black, rounds=20)
assert 132 == len(tiles_black2)

if EXEC_RESOURCE_HOGS:
  tiles_black2 = cell_automate(tiles_black, rounds=100)
  assert 2208 == len(tiles_black2)


# In[ ]:


if EXEC_RESOURCE_HOGS:
  flips = flip_day24_lines( parse_day24(ins) )
  tiles_black = filterl(lambda it: flips[it] % 2 == 1, flips.keys())
  log.info(f"Day 24 part 1 solution: black tiles#={len(tiles_black)}") #" from {tiles_black}")
  tiles_black2 = cell_automate(tiles_black, rounds=100)
  log.info(f"Day 24 part 2 solution: black tiles#={len(tiles_black2)}") #" from {tiles_black}")
  # took 1496 seconds!


# ### Day 24: Combo Breaker

# In[ ]:


def find_loopsize(pubkey, max_iter=100_000):
  subjectnum = 7
  val = 1
  for i in range(1, max_iter+1):
    val = (val * subjectnum) % 20201227
    if val == pubkey:
      break
  if i == max_iter:
    raise Exception("failsafe")
  return i

def encrypt_day25(subjectnum=7, loopsize=None):
  log.info(f"[encrypt_day25] subject#={subjectnum}, loopsize={loopsize}")
  val = 1
  for i in range(loopsize):
    val = (val * subjectnum) % 20201227
  return val


# In[ ]:


tests = """
5764801
17807724
""".strip()


# In[ ]:


card_pubkey, door_pubkey = mapl(int, tests.split("\n"))
log.info("tests card-pubkey={card_pubkey}, door pubkey=(door_pubkey)")

card_loopsize = find_loopsize(card_pubkey)
door_loopsize = find_loopsize(door_pubkey)
log.info(f"tests result: card-loopsize={card_loopsize}, door_loopsize={door_loopsize}")
t1 = encrypt_day25(subjectnum=door_pubkey, loopsize=card_loopsize)
t2 = encrypt_day25(subjectnum=card_pubkey, loopsize=door_loopsize)
log.info(f"tests result: encryption key={t1} : encrypted {t1} =? {t2}")
assert( t1 == t2 )


# In[ ]:


ins = aoc.read_file_to_list('in/day25.in')
card_pubkey, door_pubkey = mapl(int, ins)
log.info(f"card-pubkey={card_pubkey}, door pubkey={door_pubkey}")
card_loopsize = find_loopsize(card_pubkey, max_iter=10_000_000)
door_loopsize = find_loopsize(door_pubkey, max_iter=10_000_000)
log.info(f"intermed result: card-loopsize={card_loopsize:,}, door_loopsize={door_loopsize:,}")
t1 = encrypt_day25(subjectnum=door_pubkey, loopsize=card_loopsize)
t2 = encrypt_day25(subjectnum=card_pubkey, loopsize=door_loopsize)
log.info(f"Day 25 solution: encryption key={t1} : encrypted {t1} =? {t2}")


# In[ ]:




