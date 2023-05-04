!start.

me(2).

+!start : me(2) <- .print("Hello World!");
                    .send(agent2, tell, me(3));
                    +me(4);
                    .print("me(2) sended").