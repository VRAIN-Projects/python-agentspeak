!start.

+!start : true
    <-
        .print("Preguntar Plan");
        .send(agent2,askHow,"+!hello");
        .print("Plan Añadido");
        .wait(1000);
        !hello;
        !hello;
        !hello
.