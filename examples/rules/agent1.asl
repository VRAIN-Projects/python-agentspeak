child(X, Y) :- parent(Y, X).
 
me(3).
me(4).
parent(bob, jane).



!start.

+!start <- .print("Comienza a ejecutar el plan"); .wait(10); +parent(bob, jane); .print("parent(bob, jane) add"); .wait(2); !prueba.

+!prueba : parent(bob, jane) & child(jane, bob)<- .print("Ha ido bien").

concern__(X):- parent(bob,jane) & X = 1 | parent(jane,bob) & X = 2.