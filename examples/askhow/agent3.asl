!start.

+!start : true 
    <-
        .print("Preguntar Plan");
        .send(agent4, tellHow, "+!hola(N,C) <- .print(C, \" saluda a \", N).");
        .send(agent4, askHow, "+!hola(Q,F)");
        .print("Plan Añadido ...");
        !hola(1,2).