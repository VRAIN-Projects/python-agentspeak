
me(100).

@etiqueta[macarron(45), esperanza(gracia)]
+!hello : true <-
    .print("Este es el plan");
    .print("Este es el plan").


@etiqueta[macarron(45)]
+!hello : me(100) <-
    .print("Este es el plan2");
    .print("Este es el plan2");
    -me(100).

@etiqueta[macarron(45)]
-!hello <-
    .print("Este ya no es el plan").