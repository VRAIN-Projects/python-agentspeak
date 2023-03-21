
me(1).
concern__(X) :- X = 1 & me(2) | X = 5 & me(1).
concern__(X) :- X = 2 & me(2) | X = 9 & me(1).

