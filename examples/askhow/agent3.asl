!start.

+!start : true 
    <-
        .print("Preguntar Plan");
        .send(agent4, tellHow, "+!hola(N,M) <- .print(\"Hola a \", N, \" y a \", M).");
        .send(agent4, askHow, "+!hola(T,R)");
        .print("Plan Añadido ...");
        !hola(manolo, rodolfo).