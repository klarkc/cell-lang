# The Cell Programming Language
Here I introduce **Cell** which purports to be a programming language that also serves as a foundation of mathematics.
It uses Mathematica-like syntax.

In the below, `>>>` is a REPL prompt.

## Natural numbers
All types are to be built up within the language. The most important type is the Natural numbers (0, 1, 2, ...). Hash sign begin a comment until end of line.

We define naturals with Peano arithmetic.
```
Zero             # 0
Succ[Zero]       # 1
Succ[Succ[Zero]] # 2
# ... etc ...
```
I am now going to use apostrophe + natural number as syntactical sugar for a number in Peano artihetmic. E.g. `'0` is equal to `Zero`.

## Bind and recurse
We define symbols via assignment, like this:
```
>>> Name = Expression[A, B]
>>> Name
Expression[A, B]
```
Bind is a special form that defines bind (lambda) expressions.
Note that we need to use it with Evaluate to work as expected.
```
>>> A = Bind[x, y, Expr[x, y]]
>>> A[a, b]
A[a, b]
>>> Evaluate[A[a, b]]
Expr[a, b]
```
Recurse is the construct used to create recursive functions.
It takes a natural number and returns a recursive expression.
We also need to use Evaluate here in order in order to get the expected result.
```
>>> B = Recurse[x, Succ[Self], Zero]
>>> B[Zero]
B[Zero]
>>> Evaluate[B['0]]
'0
>>> Evaluate[B['2]]
'3
```
Recurse takes, in order, the arguments: variable, recursive case, base case.
`Self` is a special identifier used in the recursive case.

If the argument to Recurse is zero, then we return the base case immediately.
Otherwise, the base case is equal to `Self` in the recursive case.
We continue replace `Self` by the last recursive case until we have iterated N times, where N is the argument passed to Recurse.

## Logical operators
And, Or, Not, Equal are logical operators that work just as expected.

## Definition of LessThan
We can now define LessThan on naturals:
```
LessThan = Bind[x, y, Recurse[z, Or[Self, Equal[Succ[x], z]], And[Equal[x, Zero], Not[Equal[x, y]]]][y]]
```
which will evaluate to
```
>>> Evaluate[LessThan['0, '0]]
And[Not[Equal['0, 0]], Equal['0, '0]]  # False
```

## Reduce
Reduce is another type of evaluation.

Evaluate is a function that takes an expression and returns an expression.
Reduce takes an expression and reduces it "to a point".

Evaluate[LessThan['3, '4]] gives a long expression:
```
Or[Or[Or[Equal[.... ]], And[Equal[...
```
whereas Reduce[LessThan['3, '4]] simply returns `True`:
```
>>> Reduce[LessThan['3, '4]]
True
```

## Integers and rationals
There is a bijection between the set of all pairs (a, b) where a,b and the naturals. We're going to use the bijection Cantor diagonalization which maps
0 -> 0,0 -- 1 -> 1,0 -- 2 -> 1,1 -- etc.

The integers can be defined as all such pairs and three equivalence classes:
Positive (a < b), Zero (a = b), Negative (b < a). In other words, we can define these classes in terms of LessThan and Equal above.

The class `Integer.Positive[3]` is a representative in class Positive.
So `a - b = 3` and `b < a`. We chose the representative corresponding to pair `3,0`.

Each class "starts" a new ordering a fresh (a copy of the naturals) + carries explicit type information. I'll try to explain as clearly as I can:

`Integer.Positive[3]`: This class correspond to `(3,0)` in a Cantor diagonal mapping `f`. Take the inverse `f^-1` to get to the naturals, `6`.
To get from 6, we need to add the type information `Integer.Positive`.

This will be used in proofs checkable by a computer.

We can define rationals similarly.
```
Rational = Pair[Integer, Not[Integer.Zero]]
```
Syntax is flawed here, but a rational is a pair: two natural numbers really + type information. Hopefully this makes sense.

## Proofs
Proofs, e.g. by induction
```
ForAll[x,y
  Not[LessThan[x, y]] AND Not[Equal[x, y]] \implies LessThan[y, x]
]
```
is by expanding LessThan where the induction hypothesis is set to Self.
And then comparing the AST trees for a notion of *logical equivalence*.
