child(X, Y, Z) :- parent(Y, X) & Z = 1 | parent(X, X) & Z = 2.
 
me(3).
me(4).
parent(bob, jane).



!start.

+!start <- .print("Comienza a ejecutar el plan"); .wait(10); +parent(bob, jane); .print("parent(bob, jane) add"); .wait(2); !prueba.

+!prueba : parent(bob, jane) & child(jane, bob, 1)<- .print("Ha ido bien").

concern__(X):- parent(bob,jane) & X = 1 | parent(jane,bob) & X = 2.