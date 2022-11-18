me(100).

!hello.


@etiqueta[macarron(45)]
+!hello : me(100) <-
    .print("Este es el plan2");
    .wait(2000);
    !hello.

@etiqueta
+!hello : true <-
    .print("Este es el plan");
    .wait(3000);
    !hello.

