# Python Global Interpreter Lock (GIL)

a type of process lock which is used by python whenever it deals with processes. Generally, Python only uses only one thread to execute the set of written statements. This means that in python only one thread will be executed at a time. 

**The performance of the single-threaded process and the multi-threaded process will be the same in python and this is because of GIL in python**. We can not achieve multithreading in python because we have global interpreter lock which restricts the threads and works as a single thread.

## why GIL in python

Avoid memory leaked and deadlock problem. For the variable in Python, we use reference counter to handle the memory of value, if the count reach to zero the varible will be realeased automatically. If multiple threads access the count value, may lead to memory leaked and deadlock.