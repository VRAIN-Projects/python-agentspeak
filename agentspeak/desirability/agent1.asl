!start.

me(3).

+!start : me(3) <- -me(3); me(2).

+!start : me(2) <- ?X ; .print(X); -me(2); me(1).

concern__(X):- me(3) & X = 1 | me(2) & X = 2 | me(1) & X = 4.