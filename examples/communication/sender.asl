!start.
+!start <-
  /*.print("about to broadcast ...");
  .broadcast(achieve, hello(42,56));
  .print("broadcasted.");
  .print("sending individual message ...");
  .send(receiver, achieve, hello(23,3));*/
  .send(receiver,askHow, "+!hello(M,N)");
  .wait(2000);
  !hello(23,34);
  .print("sent.").
