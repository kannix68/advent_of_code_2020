#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Advent of Code 2020

## This solution (Jupyter notebook; python3.7) by kannix68, @ 2020-12.
## Using anaconda distro, conda v4.9.2. installation on MacOS v10.14.6 "Mojave".


# In[ ]:


import sys
print("Python version:", sys.version)
print("Version info:", sys.version_info)


# In[ ]:


DEBUG_FLAG = 0


# In[ ]:


# Generic AoC code

def assert_msg(msg, assertion):
  assert assertion, "ERROR on assert: {}".format(msg)
  print("assert-OK: {}".format(msg))

def expect_msg(msg, expected, inputs):
  assert_msg(msg.format(inputs, expected), expected == solve(inputs))

def log_debug(*args,**kwargs):
  if DEBUG_FLAG > 0:
    print(*args,**kwargs)

def read_file_to_str(filename):
  """Read a file into one string."""
  with open(filename, 'r') as inputfile:
    data = inputfile.read()
  return data

def read_file_to_list(filename):
  """Read a file into a list of strings (per line), each ending whitespace stripped."""
  with open(filename, 'r') as inputfile:
    lines_list = inputfile.readlines()
  #lines_list = [line.rstrip('\n') for line in lines_list] # via list comprehension
  lines_list = list(map(lambda it: it.rstrip(), lines_list)) # via map
  return lines_list


# In[ ]:


# Problem domain code


# In[ ]:


# Day 1
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
    #print(v)
    if sum(v) == THIS_YEAR:
      print(f"found {v}")
      p = np.prod(np.array(v))
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
    #print(v)
    if sum(v) == THIS_YEAR:
      print(f"found {v}")
      p = np.prod(np.array(v))
      print(f"product={p}")


# In[ ]:


print("Day 1 b")
print("tests:", solve01b(tests))


# In[ ]:


print("solution:", solve01b(ins))


# In[ ]:


# Day 2
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


# In[ ]:


# Day 3
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


# In[ ]:




