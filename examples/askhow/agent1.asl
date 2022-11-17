me(100).

!start.


+!start : true
    <-
        .print("Preguntar Plan");
        .send(agent2,askHow,"+!hello");
        .print("Plan Añadido");
        .wait(5000);
        !hello;
        -me(100);
        !hello;
        !hello
.