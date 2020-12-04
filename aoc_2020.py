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
    v = np.array(v)
    #print(v)
    if v.sum() == THIS_YEAR:
      print(f"found {v}")
      p = v.prod()
      print(f"product={p}")


# In[ ]:


solve01a(tests)


# In[ ]:


ins = list(map(int, read_file_to_list('./in/day01.in')))
#ins


# In[ ]:


solve01a(ins)


# In[ ]:


def solve01b(l):
  for v in itertools.combinations(l, 3):
    v = np.array(v)
    #print(v)
    if v.sum() == THIS_YEAR:
      print(f"found {v}")
      p = v.prod() #np.prod(np.array(v))
      print(f"product={p}")
  return p


# In[ ]:


print("Day 1 b")
print("test results:", solve01b(tests))


# In[ ]:


print("solution:", solve01b(ins))


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


def solve2a(l):
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
  print(f"num of valid passwords={ct}")


# In[ ]:


print("tests:", solve2a(tests))


# In[ ]:


ins = read_file_to_list('./in/day02.in')
print("solution:", solve2a(ins))


# In[ ]:


def solve2b(l):
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
  print(f"num of valid passwords={ct}")
  return ct


# In[ ]:


print("Day 2 b")
print("assert day 2 b test conditions")
assert( 1 == solve2b([tests[0]]) )
assert( 0 == solve2b([tests[1]]) )
assert( 0 == solve2b([tests[2]]) )
print("assertions were ok.")


# In[ ]:


solve2b(tests)


# In[ ]:


solve2b(ins)


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


def solve3a(l2d):
  num_rows = len(l2d)
  num_cols = len(l2d[0])
  print(f"num rows={num_rows}, cols={num_cols}")
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
      print(f"break at iter#={iter}")
      break
    else:
      iter += 1
  outstr = f"encountered {ct} trees."
  if DEBUG_FLAG > 0:
    outstr += f"Path={tpath}"
  print(outstr)


# In[ ]:


solve3a(tests)


# In[ ]:


ins = prepare_input(read_file_to_list('./in/day03.in'))


# In[ ]:


solve3a(ins)


# In[ ]:


def solve3b(l2d, vec):
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
assert( 2 == solve3b(tests, [1, 1]))
assert( 7 == solve3b(tests, [3, 1]))
assert( 3 == solve3b(tests, [5, 1]))
assert( 4 == solve3b(tests, [7, 1]))
assert( 2 == solve3b(tests, [1, 2]))
print("assertions were ok.")


# In[ ]:


p = solve3b(tests, [1, 1]) * solve3b(tests, [3, 1]) * solve3b(tests, [5, 1])     * solve3b(tests, [7, 1]) * solve3b(tests, [1, 2])
print("day 3 b test result (product):", p)


# In[ ]:


p = solve3b(ins, [1, 1]) * solve3b(ins, [3, 1]) * solve3b(ins, [5, 1])       * solve3b(ins, [7, 1]) * solve3b(ins, [1, 2])
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


def solve4a(passports):
  ct = 0
  for passport in passports:
    if passport_valid(passport):
      ct +=1
  log_debug("valid-count:", ct)
  return ct


# In[ ]:


print("tests valid-count:", solve4a(tests))


# In[ ]:


ins = read_file_to_str('./in/day04.in').split("\n\n")
print("solution 4 a valid-count:", solve4a(ins))


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


for passport in tests_invalid:
  print(passport)
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


for passport in tests_valid:
  print(passport)
  print("valid?:", passport_valid2(passport))
  print()


# In[ ]:


def solve4b(passports):
  ct = 0
  for passport in passports:
    log_debug(passport)
    if passport_valid2(passport):
      ct +=1
  log_debug("valid-count:", ct)
  return ct


# In[ ]:


assert( 0 == solve4b(tests_invalid) )


# In[ ]:


assert( 4 == solve4b(tests_valid) )


# In[ ]:


result = solve4b(ins)
print("Day 4 b result:", result)


# In[ ]:




