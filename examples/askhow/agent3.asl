!start.

+!start : true 
    <-
        .print("Preguntar Plan");
        .send(agent4, tellHow, "@pred[me(3)] +!hola(N,C)[me(A)] <- .print(C, \" saluda a \", N).");
        .send(agent4, askHow, "+!hola(Q,F)");
        .print("Plan Añadido ...");
        !hola(1,2)[me(10)].